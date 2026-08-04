"""
Platform Adapter Registry

Allows platform adapters (built-in and plugin) to self-register so the gateway
can discover and instantiate them without hardcoded if/elif chains.

Built-in adapters continue to use the existing if/elif in _create_adapter()
for now.  Plugin adapters register here via PluginContext.register_platform()
and are looked up first -- if nothing is found the gateway falls through to
the legacy code path.

Usage (plugin side):

    from gateway.platform_registry import platform_registry, PlatformEntry

    platform_registry.register(PlatformEntry(
        name="irc",
        label="IRC",
        adapter_factory=lambda cfg: IRCAdapter(cfg),
        check_fn=check_requirements,
        validate_config=lambda cfg: bool(cfg.extra.get("server")),
        required_env=["IRC_SERVER"],
        install_hint="pip install irc",
    ))

Usage (gateway side):

    adapter = platform_registry.create_adapter("irc", platform_config)
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from registry_transaction import RegistryTransactionConflict

logger = logging.getLogger(__name__)


@dataclass
class PlatformEntry:
    """Metadata and factory for a single platform adapter."""

    # Identifier used in config.yaml (e.g. "irc", "viber").
    name: str

    # Human-readable label (e.g. "IRC", "Viber").
    label: str

    # Factory callable: receives a PlatformConfig, returns an adapter instance.
    # Using a factory instead of a bare class lets plugins do custom init
    # (e.g. passing extra kwargs, wrapping in try/except).
    adapter_factory: Callable[[Any], Any]

    # Returns True when the platform's dependencies are available.
    check_fn: Callable[[], bool]

    # Optional: given a PlatformConfig, is it properly configured?
    # If None, the registry skips config validation and lets the adapter
    # fail at connect() time with a descriptive error.
    validate_config: Optional[Callable[[Any], bool]] = None

    # Optional: given a PlatformConfig, is the platform connected/enabled?
    # Used by ``GatewayConfig.get_connected_platforms()`` and setup UI status.
    # If None, falls back to ``validate_config`` or ``check_fn``.
    is_connected: Optional[Callable[[Any], bool]] = None

    # Env vars this platform needs (for ``hermes setup`` display).
    required_env: list = field(default_factory=list)

    # Hint shown when check_fn returns False.
    install_hint: str = ""

    # Optional setup function for interactive configuration.
    # Signature: () -> None (prompts user, saves env vars).
    # If None, falls back to _setup_standard_platform (needs token_var + vars)
    # or a generic "set these env vars" display.
    setup_fn: Optional[Callable[[], None]] = None

    # "builtin" or "plugin"
    source: str = "plugin"

    # Name of the plugin manifest that registered this entry (empty for
    # built-ins).  Used by ``hermes gateway setup`` to auto-enable the
    # owning plugin when the user configures its platform.
    plugin_name: str = ""

    # ── Auth env var names (for _is_user_authorized integration) ──
    # E.g. "IRC_ALLOWED_USERS" — checked for comma-separated user IDs.
    allowed_users_env: str = ""
    # E.g. "IRC_ALLOW_ALL_USERS" — if truthy, all users authorized.
    allow_all_env: str = ""

    # ── Message limits ──
    # Max message length for smart-chunking.  0 = no limit.
    max_message_length: int = 0

    # ── Privacy ──
    # If True, session descriptions redact PII (phone numbers, etc.)
    pii_safe: bool = False

    # ── Display ──
    # Emoji for CLI/gateway display (e.g. "💬")
    emoji: str = "🔌"

    # Whether this platform should appear in _UPDATE_ALLOWED_PLATFORMS
    # (allows /update command from this platform).
    allow_update_command: bool = True

    # ── LLM guidance ──
    # Platform hint injected into the system prompt (e.g. "You are on IRC.
    # Do not use markdown.").  Empty string = no hint.
    platform_hint: str = ""

    # ── Env-driven auto-configuration ──
    # Optional: read env vars, return a dict of ``PlatformConfig.extra`` fields
    # to seed when the platform is auto-enabled.  Called during
    # ``_apply_env_overrides`` BEFORE the adapter is constructed, so
    # ``gateway status`` etc. can reflect env-only configuration without
    # instantiating the adapter.  Return ``None`` (or an empty dict) to skip.
    # Signature: () -> Optional[dict[str, Any]]
    env_enablement_fn: Optional[Callable[[], Optional[dict]]] = None

    # ── YAML→env config bridge ──
    # Optional: translate this platform's ``config.yaml`` keys into env vars
    # and/or seed ``PlatformConfig.extra`` directly.  Lets a plugin own its
    # YAML config translation instead of forcing core ``gateway/config.py``
    # to know every platform's schema.
    #
    # Signature: (yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]
    # Called from ``load_gateway_config()`` after the generic shared-key loop
    # and before ``_apply_env_overrides``.  Mutating ``os.environ`` is allowed
    # (use ``not os.getenv(...)`` guards to preserve env > YAML precedence);
    # any returned dict is merged into ``PlatformConfig.extra``.  Exceptions
    # are caught and logged at debug level.
    # See website/docs/developer-guide/adding-platform-adapters.md for the
    # full contract and a worked example.
    apply_yaml_config_fn: Optional[Callable[[dict, dict], Optional[dict]]] = None

    # Optional: home-channel env var name for cron/notification delivery
    # (e.g. ``"IRC_HOME_CHANNEL"``).  When set, ``cron.scheduler`` treats this
    # platform as a valid ``deliver=<name>`` target and reads the env var to
    # resolve the default chat/room ID.  Empty = no cron home-channel support.
    cron_deliver_env_var: str = ""

    # ── Standalone (out-of-process) sending ──
    # Optional: async coroutine that delivers a message without a live
    # gateway adapter.  Called by ``tools/send_message_tool._send_via_adapter``
    # when ``cron`` runs in a separate process from the gateway and the
    # in-process adapter weakref is therefore ``None``.
    #
    # Signature:
    #     async (pconfig, chat_id, message, *, thread_id=None,
    #            media_files=None, force_document=False) -> dict
    #
    # Returns ``{"success": True, "message_id": ...}`` on success or
    # ``{"error": str}`` on failure.  Plugin authors typically open an
    # ephemeral connection / acquire a fresh OAuth token, send, and close.
    # Without this hook, plugin platforms cannot serve as cron ``deliver=``
    # targets when the gateway is not co-resident with the cron process.
    standalone_sender_fn: Optional[Callable[..., Awaitable[dict]]] = None


class _PlatformSnapshot:
    __slots__ = ("_owner", "_entries", "_deferred", "_resolving", "_generation")

    def __init__(
        self,
        owner: "PlatformRegistry",
        entries: tuple,
        deferred: tuple,
        resolving: tuple,
        generation: int,
    ) -> None:
        self._owner = owner
        self._entries = entries
        self._deferred = deferred
        self._resolving = resolving
        self._generation = generation


class _PreparedPlatformState:
    __slots__ = ("_owner", "_snapshot", "_entries", "_deferred", "_generation")

    def __init__(
        self,
        owner: "PlatformRegistry",
        snapshot: _PlatformSnapshot,
        entries: dict,
        deferred: dict,
        generation: int,
    ) -> None:
        self._owner = owner
        self._snapshot = snapshot
        self._entries = entries
        self._deferred = deferred
        self._generation = generation


class _DeferredResolution:
    __slots__ = ("event", "owner_thread")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.owner_thread = threading.get_ident()


class PlatformRegistry:
    """Thread-safe central registry of platform adapters."""

    surface = "platform"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, PlatformEntry] = {}
        # Deferred platform loaders: name -> zero-arg callable that imports the
        # owning plugin module (which calls register() and populates _entries).
        #
        # Why this exists: platform adapter modules import heavy, platform-
        # specific SDKs at module level (lark_oapi, microsoft_teams, discord.py,
        # slack_bolt, ...). Eagerly loading all ~20 bundled platform plugins at
        # plugin-discovery time added several seconds to *every* `hermes`
        # invocation -- including plain `hermes chat`, which never touches any
        # gateway platform. Discovery now registers a cheap deferred loader per
        # platform; the real module is imported only when a registry lookup
        # actually asks for that platform (gateway start, cron delivery,
        # `hermes setup`/`gateway status`, send_message).
        self._deferred: dict[str, Callable[[], None]] = {}
        self._resolving: dict[str, _DeferredResolution] = {}
        # Current cross-thread deferred-loader waits.  Tracking this graph lets
        # re-entrant loaders break only cyclic waits while ordinary readers
        # still wait for the owning loader's final entry (or failure).
        self._waiting: dict[int, _DeferredResolution] = {}
        self._generation = 0
        self._frozen = False
        self._transaction_owner: Optional[PlatformRegistry] = None
        self._transaction_snapshot: Optional[_PlatformSnapshot] = None

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    # -- opaque transaction API -------------------------------------------

    def take_snapshot(self) -> _PlatformSnapshot:
        with self._lock:
            return _PlatformSnapshot(
                self,
                tuple(self._entries.items()),
                tuple(self._deferred.items()),
                tuple(self._resolving.items()),
                self._generation,
            )

    def _validate_snapshot(self, snapshot: _PlatformSnapshot) -> None:
        if not isinstance(snapshot, _PlatformSnapshot) or snapshot._owner is not self:
            raise ValueError(
                "transaction snapshot belongs to a different PlatformRegistry"
            )

    def create_view(
        self,
        snapshot: _PlatformSnapshot,
        *,
        remove_keys=(),
    ) -> "PlatformRegistry":
        self._validate_snapshot(snapshot)
        removed = frozenset(remove_keys)
        staged = PlatformRegistry()
        staged._entries = {
            name: entry
            for name, entry in snapshot._entries
            if name not in removed
        }
        staged._deferred = {
            name: loader
            for name, loader in snapshot._deferred
            if name not in removed
        }
        staged._generation = snapshot._generation
        staged._transaction_owner = self
        staged._transaction_snapshot = snapshot
        return staged

    def prepare(
        self,
        snapshot: _PlatformSnapshot,
        staged: "PlatformRegistry",
    ) -> _PreparedPlatformState:
        self._validate_snapshot(snapshot)
        if (
            not isinstance(staged, PlatformRegistry)
            or staged._transaction_owner is not self
            or staged._transaction_snapshot is not snapshot
        ):
            raise ValueError(
                "transaction view belongs to a different PlatformRegistry"
            )
        with staged._lock:
            prepared = _PreparedPlatformState(
                self,
                snapshot,
                dict(staged._entries),
                dict(staged._deferred),
                staged._generation,
            )
            staged._frozen = True
            return prepared

    def _matches_snapshot_locked(self, snapshot: _PlatformSnapshot) -> bool:
        return (
            self._generation == snapshot._generation
            and len(self._entries) == len(snapshot._entries)
            and all(
                name in self._entries and self._entries[name] is entry
                for name, entry in snapshot._entries
            )
            and len(self._deferred) == len(snapshot._deferred)
            and all(
                name in self._deferred and self._deferred[name] is loader
                for name, loader in snapshot._deferred
            )
            and len(self._resolving) == len(snapshot._resolving)
            and all(
                name in self._resolving
                and self._resolving[name] is resolution
                for name, resolution in snapshot._resolving
            )
        )

    def validate_prepared_locked(
        self,
        snapshot: _PlatformSnapshot,
        prepared: _PreparedPlatformState,
    ) -> None:
        self._validate_snapshot(snapshot)
        if (
            not isinstance(prepared, _PreparedPlatformState)
            or prepared._owner is not self
            or prepared._snapshot is not snapshot
        ):
            raise ValueError(
                "prepared state belongs to a different PlatformRegistry"
            )
        current_thread = threading.get_ident()
        foreign_resolution = any(
            resolution.owner_thread != current_thread
            for _name, resolution in snapshot._resolving
        )
        if foreign_resolution or not self._matches_snapshot_locked(snapshot):
            raise RegistryTransactionConflict(
                self.surface,
                snapshot._generation,
                self._generation,
            )

    def install_prepared_locked(
        self,
        snapshot: _PlatformSnapshot,
        prepared: _PreparedPlatformState,
    ) -> None:
        self.validate_prepared_locked(snapshot, prepared)
        self._entries = prepared._entries
        self._deferred = prepared._deferred
        self._generation = max(self._generation, prepared._generation) + 1

    def restore_snapshot_locked(self, snapshot: _PlatformSnapshot) -> None:
        """Restore an opaque snapshot exactly while the registry lock is held."""
        self._validate_snapshot(snapshot)
        self._entries = dict(snapshot._entries)
        self._deferred = dict(snapshot._deferred)
        self._resolving = dict(snapshot._resolving)
        self._generation = snapshot._generation

    # -- deferred loading ----------------------------------------------------

    def register_deferred(self, name: str, loader: Callable[[], None]) -> None:
        """Register a lazy loader for a platform that hasn't been imported yet.

        *loader* is a zero-arg callable that imports the owning plugin module,
        which is expected to call :meth:`register` with the real entry for
        *name*.  The loader runs at most once, the first time *name* is looked
        up (or when the full entry list is materialized).  A real entry that is
        registered directly (e.g. a built-in) takes precedence -- the deferred
        loader is then dropped.
        """
        with self._lock:
            if self._frozen:
                raise RuntimeError("platform transaction view is frozen")
            if name in self._entries:
                return
            self._deferred[name] = loader
            self._generation += 1

    def _resolve(self, name: str) -> None:
        """Run the deferred loader for *name* if one is pending."""
        resolution: Optional[_DeferredResolution] = None
        loader: Optional[Callable[[], None]] = None
        while True:
            with self._lock:
                if name in self._entries:
                    return
                active = self._resolving.get(name)
                if active is not None:
                    if active.owner_thread == threading.get_ident():
                        return
                    wait_resolution = active
                else:
                    loader = self._deferred.pop(name, None)
                    if loader is None:
                        return
                    resolution = _DeferredResolution()
                    self._resolving[name] = resolution
                    self._generation += 1
                    wait_resolution = None
            if wait_resolution is None:
                break
            if not self._wait_for_resolution(wait_resolution):
                return

        assert loader is not None
        assert resolution is not None
        try:
            loader()
        except Exception as e:
            logger.warning(
                "Deferred load of platform '%s' failed: %s",
                name,
                e,
                exc_info=True,
            )
        finally:
            with self._lock:
                if self._resolving.get(name) is resolution:
                    self._resolving.pop(name, None)
                resolution.event.set()

    def _wait_for_resolution(self, resolution: _DeferredResolution) -> bool:
        """Wait for *resolution*, unless that wait would close a loader cycle.

        Returns ``False`` only when the caller must treat the still-provisional
        resolution as unavailable so its own loader can finish and unblock the
        cycle.  The registry lock is never held while waiting.
        """
        current_thread = threading.get_ident()
        with self._lock:
            if resolution.event.is_set():
                return True

            owner_thread = resolution.owner_thread
            seen = set()
            while owner_thread not in seen:
                if owner_thread == current_thread:
                    return False
                seen.add(owner_thread)
                owner_wait = self._waiting.get(owner_thread)
                if owner_wait is None:
                    self._waiting[current_thread] = resolution
                    break
                owner_thread = owner_wait.owner_thread
            else:
                # Defensive fallback for an already-cyclic dependency graph.
                return False

        try:
            resolution.event.wait()
        finally:
            with self._lock:
                if self._waiting.get(current_thread) is resolution:
                    self._waiting.pop(current_thread, None)
        return True

    def _resolve_all(self) -> None:
        """Run every pending deferred loader.

        Used by the iterate-all accessors (``all_entries``/``plugin_entries``),
        which are only called by paths that genuinely need every adapter:
        gateway startup, ``hermes setup``/``gateway status``, channel
        directory.  CLI chat never iterates the full set.
        """
        while True:
            current_thread = threading.get_ident()
            with self._lock:
                pending = list(self._deferred)
                active = [
                    resolution
                    for resolution in self._resolving.values()
                    if resolution.owner_thread != current_thread
                ]
            if pending:
                for name in pending:
                    self._resolve(name)
                continue
            if active:
                waited = False
                for resolution in active:
                    waited = self._wait_for_resolution(resolution) or waited
                if waited:
                    continue
                # Every remaining foreign resolution depends (possibly
                # transitively) on this loader.  Returning exposes only already
                # committed entries and lets this loader release the cycle.
                return
            return

    def register(self, entry: PlatformEntry) -> None:
        """Register a platform adapter entry.

        If an entry with the same name exists, it is replaced (last writer
        wins -- this lets plugins override built-in adapters if desired).
        """
        name = entry.name
        source = entry.source
        with self._lock:
            if self._frozen:
                raise RuntimeError("platform transaction view is frozen")
            self._deferred.pop(name, None)
            prev = self._entries.get(name)
            self._entries[name] = entry
            self._generation += 1
        if prev is not None:
            logger.info(
                "Platform '%s' re-registered (was %s, now %s)",
                name,
                prev.source,
                source,
            )
        logger.debug("Registered platform adapter: %s (%s)", name, source)

    def unregister(self, name: str) -> bool:
        """Remove a platform entry.  Returns True if it existed."""
        with self._lock:
            if self._frozen:
                raise RuntimeError("platform transaction view is frozen")
            deferred = self._deferred.pop(name, None)
            entry = self._entries.pop(name, None)
            if deferred is not None or entry is not None:
                self._generation += 1
            return entry is not None

    def get(self, name: str) -> Optional[PlatformEntry]:
        """Look up a platform entry by name."""
        self._resolve(name)
        with self._lock:
            return self._entries.get(name)

    def all_entries(self) -> list[PlatformEntry]:
        """Return all registered platform entries."""
        self._resolve_all()
        with self._lock:
            return list(self._entries.values())

    def plugin_entries(self) -> list[PlatformEntry]:
        """Return only plugin-registered platform entries."""
        self._resolve_all()
        with self._lock:
            entries = list(self._entries.values())
        return [entry for entry in entries if entry.source == "plugin"]

    def is_registered(self, name: str) -> bool:
        # A deferred (not-yet-imported) platform still counts as registered --
        # the loader will materialize it on first real use.  This keeps cheap
        # membership checks (toolset resolution, webhook deliver-target checks)
        # from triggering a heavy import.
        with self._lock:
            return (
                name in self._entries
                or name in self._deferred
                or name in self._resolving
            )

    def create_adapter(self, name: str, config: Any) -> Optional[Any]:
        """Create an adapter instance for the given platform name.

        Returns None if:
        - No entry registered for *name*
        - check_fn() returns False (missing deps)
        - validate_config() returns False (misconfigured)
        - The factory raises an exception
        """
        entry = self.get(name)
        if entry is None:
            return None

        if not entry.check_fn():
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning(
                "Platform '%s' requirements not met%s",
                entry.label,
                hint,
            )
            return None

        if entry.validate_config is not None:
            try:
                if not entry.validate_config(config):
                    logger.warning(
                        "Platform '%s' config validation failed",
                        entry.label,
                    )
                    return None
            except Exception as e:
                logger.warning(
                    "Platform '%s' config validation error: %s",
                    entry.label,
                    e,
                )
                return None

        try:
            adapter = entry.adapter_factory(config)
            return adapter
        except Exception as e:
            logger.error(
                "Failed to create adapter for platform '%s': %s",
                entry.label,
                e,
                exc_info=True,
            )
            return None


# Module-level singleton
platform_registry = PlatformRegistry()
