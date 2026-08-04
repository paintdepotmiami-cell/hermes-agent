"""
Hermes Plugin System
====================

Discovers, loads, and manages plugins from four sources:

1. **Bundled plugins** – ``<repo>/plugins/<name>/`` (shipped with hermes-agent;
   ``memory/`` and ``context_engine/`` subdirs are excluded — they have their
   own discovery paths)
2. **User plugins**   – ``~/.hermes/plugins/<name>/``
3. **Project plugins** – ``./.hermes/plugins/<name>/`` (opt-in via
   ``HERMES_ENABLE_PROJECT_PLUGINS``)
4. **Pip plugins**     – packages that expose the ``hermes_agent.plugins``
   entry-point group.

Later sources override earlier ones on name collision, so a user or project
plugin with the same name as a bundled plugin replaces it.

Each directory plugin must contain a ``plugin.yaml`` manifest **and** an
``__init__.py`` with a ``register(ctx)`` function.

Lifecycle hooks
---------------
Plugins may register callbacks for any of the hooks in ``VALID_HOOKS``.
The agent core calls ``invoke_hook(name, **kwargs)`` at the appropriate
points.

Tool registration
-----------------
``PluginContext.register_tool()`` delegates to ``tools.registry.register()``
so plugin-defined tools appear alongside the built-in tools.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

from hermes_constants import get_hermes_home
from private_secret_policy import (
    PrivateSecretPolicyError,
    prepare_private_secret_names,
    replace_private_secret_names,
    restore_private_secret_policy,
    snapshot_private_secret_policy,
)
from utils import env_var_enabled, fast_safe_load
from hermes_cli.config import cfg_get
from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION, VALID_MIDDLEWARE


def get_bundled_plugins_dir() -> Path:
    """Locate the bundled ``plugins/`` directory.

    Honours ``HERMES_BUNDLED_PLUGINS`` (set by the Nix wrapper / packaged
    installs) so read-only store paths are consulted first.  Falls back to
    the in-repo path used during development.
    """
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"

from registry_transaction import RegistryTransactionConflict

try:
    import yaml
except ImportError:  # pragma: no cover – yaml is optional at import time
    yaml = None  # type: ignore[assignment]


class PluginToolOverrideError(PermissionError):
    """Raised when a plugin attempts to override a built-in tool without
    operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override``.
    """


class MCPGovernanceRegistrationError(PermissionError):
    """Raised when a plugin cannot claim one exact MCP server safely."""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin developer debug logging
# ---------------------------------------------------------------------------
#
# Set ``HERMES_PLUGINS_DEBUG=1`` to surface verbose plugin-discovery logs to
# stderr in addition to ~/.hermes/logs/agent.log. Aimed at plugin authors
# trying to figure out why their plugin isn't showing up: which directories
# were scanned, which manifests parsed, which plugins were skipped (and why),
# what each ``register(ctx)`` call registered, and full tracebacks on load
# failure.
#
# The env var is read once at import time; tests that need to flip it
# mid-process can call ``_install_plugin_debug_handler(force=True)``.

_PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on",
}
_DEBUG_HANDLER_INSTALLED = False


def _install_plugin_debug_handler(force: bool = False) -> None:
    """When HERMES_PLUGINS_DEBUG is on, tee plugin logs to stderr at DEBUG.

    Idempotent: only attaches the handler once per process unless ``force``
    is passed. Does not touch the root logger or other Hermes loggers.
    """
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Don't double-emit through the root logger when the central logging
    # config also writes to stderr. agent.log still captures everything.
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug(
        "HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled"
    )


_install_plugin_debug_handler()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_HOOKS: Set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    # Transform LLM output before it's returned to the user.
    # Plugins return a string to replace the response text, or None/empty to leave unchanged.
    # First non-None string wins. Useful for vocabulary/personality transformation.
    "transform_llm_output",
    "pre_llm_call",
    "post_llm_call",
    # Verification-loop gate. Fired once per turn when the agent has edited code
    # and is about to verify/finish (after the verify-on-stop guard). A callback
    # may keep the agent going — run a check, defer it, tidy the diff — instead
    # of stopping by returning:
    #   {"action": "continue", "message": "<follow-up instruction>"}
    # The Claude-Code Stop shape {"decision": "block", "reason": "..."} (block
    # the stop == keep going) is accepted too. Anything else lets the turn
    # finish. Hermes' shipped guidance lives in the evidence-based
    # verification-stop nudge; this hook is for user/plugin policy and is
    # bounded by agent.max_verify_nudges.
    "pre_verify",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "subagent_start",
    "subagent_stop",
    # Gateway pre-dispatch hook. Fired once per incoming MessageEvent
    # after the internal-event guard but BEFORE auth/pairing and agent
    # dispatch. Plugins may return a dict to influence flow:
    #   {"action": "skip",    "reason": "..."}  -> drop message (no reply)
    #   {"action": "rewrite", "text": "..."}    -> replace event.text, continue
    #   {"action": "allow"}  /  None             -> normal dispatch
    # Kwargs: event: MessageEvent, gateway: GatewayRunner, session_store.
    "pre_gateway_dispatch",
    # Approval lifecycle hooks. Fired by tools/approval.py when a dangerous
    # command needs an approval decision -- fires for CLI-interactive prompts,
    # gateway/ACP approvals, and smart-mode auxiliary-LLM decisions.
    # Observers only: return values are ignored. Plugins cannot veto or
    # pre-answer an approval from these hooks (use pre_tool_call to block
    # a tool before it reaches approval).
    #
    # Kwargs for pre_approval_request:
    #   command: str, description: str, pattern_key: str, pattern_keys: list[str],
    #   session_key: str, surface: "cli" | "gateway" | "smart"
    # Kwargs for post_approval_response: same as above plus
    #   choice: "once" | "session" | "always" | "deny" | "timeout"
    #           | "smart_approve" | "smart_deny"
    #   decided_by: "aux_llm"  -- only on surface="smart"
    "pre_approval_request",
    "post_approval_response",
    # Kanban task lifecycle hooks. Fired by hermes_cli.kanban_db when a task
    # transitions state, AFTER the change is committed to the board DB (so the
    # hook always sees durable state and a slow plugin can never hold the
    # SQLite write lock). Observers only: return values are ignored.
    #
    # WHICH PROCESS each fires in matters, because kanban workers run as
    # separate `hermes -p <profile> chat -q` subprocesses:
    #   - kanban_task_claimed   -> the DISPATCHER process (gateway-embedded
    #                              dispatcher or `hermes kanban dispatch`),
    #                              right before the worker subprocess spawns.
    #   - kanban_task_completed -> the WORKER process, when it calls
    #                              kanban_complete (or a CLI/manual complete).
    #   - kanban_task_blocked   -> the WORKER process (worker-initiated block)
    #                              or whichever process drove the block.
    # A plugin that needs to observe every transition centrally should hook in
    # the dispatcher; one that needs per-task in-session context should hook in
    # the worker.
    #
    # Common kwargs: task_id: str, board: str | None, assignee: str | None,
    #   run_id: int | None, profile_name: str.
    # kanban_task_completed adds: summary: str | None.
    # kanban_task_blocked adds:   reason: str | None.
    "kanban_task_claimed",
    "kanban_task_completed",
    "kanban_task_blocked",
}

ENTRY_POINTS_GROUP = "hermes_agent.plugins"

_NS_PARENT = "hermes_plugins"


def _snapshot_module_namespace(root: str) -> Dict[str, types.ModuleType]:
    """Capture module identities for *root* and all descendants."""
    prefix = f"{root}."
    return {
        name: module
        for name, module in sys.modules.items()
        if name == root or name.startswith(prefix)
    }


def _restore_module_namespace(
    root: str,
    snapshot: Dict[str, types.ModuleType],
) -> None:
    """Remove modules created in an attempt, then restore prior identities."""
    prefix = f"{root}."
    for name in list(sys.modules):
        if name == root or name.startswith(prefix):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _directory_module_name(manifest: "PluginManifest") -> str:
    key = manifest.key or manifest.name
    slug = key.replace("/", "__").replace("-", "_")
    return f"{_NS_PARENT}.{slug}"


_PROVIDER_REGISTRY_MODULES = {
    "image_gen": "agent.image_gen_registry",
    "video_gen": "agent.video_gen_registry",
    "web": "agent.web_search_registry",
    "browser": "agent.browser_registry",
    "tts": "agent.tts_registry",
    "stt": "agent.transcription_registry",
    "dashboard": "hermes_cli.dashboard_auth.registry",
}
_EXTERNAL_COMMIT_SURFACES = (
    "platform",
    "browser",
    "dashboard",
    "image_gen",
    "secret",
    "stt",
    "tts",
    "video_gen",
    "web",
)


def _external_registry_transactions() -> Dict[str, Any]:
    """Return opaque registry-owned transaction surfaces in commit order."""
    from gateway.platform_registry import platform_registry

    transactions: Dict[str, Any] = {"platform": platform_registry}
    for surface in _EXTERNAL_COMMIT_SURFACES[1:]:
        module_name = (
            "agent.secret_sources.registry"
            if surface == "secret"
            else _PROVIDER_REGISTRY_MODULES[surface]
        )
        module = importlib.import_module(module_name)
        transactions[surface] = module._plugin_transaction
    return transactions


def _env_enabled(name: str) -> bool:
    """Return True when an env var is set to a truthy opt-in value."""
    return env_var_enabled(name)


def _get_disabled_plugins() -> set:
    """Read the disabled plugins list from config.yaml.

    Kept for backward compat and explicit deny-list semantics. A plugin
    name in this set will never load, even if it appears in
    ``plugins.enabled``.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        disabled = cfg_get(config, "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _get_enabled_plugins() -> Optional[set]:
    """Read the enabled-plugins allow-list from config.yaml.

    Plugins are opt-in by default — only plugins whose name appears in
    this set are loaded. Returns:

    * ``None`` — the key is missing or malformed. Callers should treat
      this as "nothing enabled yet" (the opt-in default); the first
      ``migrate_config`` run populates the key with a grandfathered set
      of currently-installed user plugins so existing setups don't
      break on upgrade.
    * ``set()`` — an empty list was explicitly set; nothing loads.
    * ``set(...)`` — the concrete allow-list.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        plugins_cfg = config.get("plugins")
        if not isinstance(plugins_cfg, dict):
            return None
        if "enabled" not in plugins_cfg:
            return None
        enabled = plugins_cfg.get("enabled")
        if not isinstance(enabled, list):
            return None
        return set(enabled)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_VALID_PLUGIN_KINDS: Set[str] = {"standalone", "backend", "exclusive", "platform", "model-provider"}


@dataclass
class PluginManifest:
    """Parsed representation of a plugin.yaml manifest."""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    requires_env: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    source: str = ""        # "user", "project", or "entrypoint"
    path: Optional[str] = None
    # Plugin kind — see plugins.py module docstring for semantics.
    # ``standalone`` (default): hooks/tools of its own; opt-in via
    #                           ``plugins.enabled``.
    # ``backend``: pluggable backend for an existing core tool (e.g.
    #              image_gen). Built-in (bundled) backends auto-load;
    #              user-installed still gated by ``plugins.enabled``.
    # ``exclusive``: category with exactly one active provider (memory).
    #              Selection via ``<category>.provider`` config key; the
    #              category's own discovery system handles loading and the
    #              general scanner skips these.
    # ``platform``: gateway messaging platform adapter (e.g. IRC). Bundled
    #              platform plugins auto-load so every shipped platform is
    #              available out of the box; user-installed platform plugins
    #              in ~/.hermes/plugins/ still gated by ``plugins.enabled``
    #              (untrusted code).
    kind: str = "standalone"
    # Registry key — path-derived, used by ``plugins.enabled``/``disabled``
    # lookups and by ``hermes plugins list``. For a flat plugin at
    # ``plugins/disk-cleanup/`` the key is ``disk-cleanup``; for a nested
    # category plugin at ``plugins/image_gen/openai/`` the key is
    # ``image_gen/openai``. When empty, falls back to ``name``.
    key: str = ""


@dataclass(frozen=True, slots=True)
class MCPGovernorRegistration:
    """Immutable manager-owned attribution for one exact server claim."""

    server_name: str
    plugin_id: str
    policy: Any = field(repr=False, compare=False)
    secret_env_names: frozenset[str] = field(default_factory=frozenset)


@dataclass
class LoadedPlugin:
    """Runtime state for a single loaded plugin."""

    manifest: PluginManifest
    module: Optional[types.ModuleType] = None
    tools_registered: List[str] = field(default_factory=list)
    hooks_registered: List[str] = field(default_factory=list)
    middleware_registered: List[str] = field(default_factory=list)
    commands_registered: List[str] = field(default_factory=list)
    mcp_governors_registered: List[str] = field(default_factory=list)
    enabled: bool = False
    error: Optional[str] = None
    # True for a bundled platform plugin recorded as a deferred (not-yet-
    # imported) loader. The module loads on first real use via the
    # platform_registry; see PluginManager._register_deferred_platform.
    deferred: bool = False


def _manifest_private_secret_names(
    manifest: PluginManifest,
) -> frozenset[str]:
    """Return validated private env names declared by one active manifest.

    Bare-string ``requires_env`` entries and rich entries without
    ``secret: true`` retain their existing setup semantics but do not become
    host-private names.  A malformed attempt to declare a private name raises
    a sanitized typed error without echoing manifest content.
    """
    raw_entries = manifest.requires_env
    if raw_entries is None:
        return frozenset()
    if not isinstance(raw_entries, (list, tuple)):
        if isinstance(raw_entries, dict) and "secret" in raw_entries:
            raise PrivateSecretPolicyError(
                "invalid plugin private-secret declaration"
            )
        return frozenset()

    declared: list[object] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        secret_marker = entry.get("secret", False)
        if secret_marker is False or secret_marker is None:
            continue
        if secret_marker is not True:
            raise PrivateSecretPolicyError(
                "invalid plugin private-secret declaration"
            )
        declared.append(entry.get("name"))
    return prepare_private_secret_names(declared)


def _manifest_is_active_for_general_discovery(
    manifest: PluginManifest,
    disabled: set,
    enabled: Optional[set],
) -> bool:
    """Match the general loader's activation rules without importing code."""
    lookup_key = manifest.key or manifest.name
    if lookup_key in disabled or manifest.name in disabled:
        return False
    if manifest.kind in {"exclusive", "model-provider"}:
        return False
    if manifest.source == "bundled" and manifest.kind in {"backend", "platform"}:
        return True
    return enabled is not None and (
        lookup_key in enabled or manifest.name in enabled
    )


class _PluginManagerState:
    """Host-private, identity-preserving snapshot of manager-owned state."""

    def __init__(self, manager: "PluginManager") -> None:
        self._plugins = dict(manager._plugins)
        self._hooks = {
            name: list(callbacks) for name, callbacks in manager._hooks.items()
        }
        self._middleware = {
            name: list(callbacks)
            for name, callbacks in manager._middleware.items()
        }
        self._plugin_tool_names = set(manager._plugin_tool_names)
        self._plugin_platform_names = set(manager._plugin_platform_names)
        self._plugin_external_names = {
            surface: set(names)
            for surface, names in manager._plugin_external_names.items()
        }
        self._cli_commands = {
            name: dict(entry) for name, entry in manager._cli_commands.items()
        }
        self._context_engine = manager._context_engine
        self._plugin_commands = {
            name: dict(entry) for name, entry in manager._plugin_commands.items()
        }
        self._plugin_skills = {
            name: dict(entry) for name, entry in manager._plugin_skills.items()
        }
        self._aux_tasks = {
            name: {
                **entry,
                "defaults": dict(entry.get("defaults", {})),
            }
            for name, entry in manager._aux_tasks.items()
        }
        self._slack_action_handlers = list(manager._slack_action_handlers)
        self._mcp_governors = dict(manager._mcp_governors)
        self._discovered = manager._discovered
        self._generation = manager._generation
        self._live_context_generation = manager._live_context_generation


class _PluginRegistrationView:
    """Isolated manager facade populated while arbitrary plugin code runs."""

    def __init__(
        self,
        state: _PluginManagerState,
        cli_ref: Any,
    ) -> None:
        # Clone every list/dict again: the rollback checkpoint and staging view
        # must never alias one another.
        self._plugins = dict(state._plugins)
        self._hooks = {
            name: list(callbacks) for name, callbacks in state._hooks.items()
        }
        self._middleware = {
            name: list(callbacks)
            for name, callbacks in state._middleware.items()
        }
        self._plugin_tool_names = set(state._plugin_tool_names)
        self._plugin_platform_names = set(state._plugin_platform_names)
        self._plugin_external_names = {
            surface: set(names)
            for surface, names in state._plugin_external_names.items()
        }
        self._cli_commands = {
            name: dict(entry) for name, entry in state._cli_commands.items()
        }
        self._context_engine = state._context_engine
        self._plugin_commands = {
            name: dict(entry) for name, entry in state._plugin_commands.items()
        }
        self._plugin_skills = {
            name: dict(entry) for name, entry in state._plugin_skills.items()
        }
        self._aux_tasks = {
            name: {
                **entry,
                "defaults": dict(entry.get("defaults", {})),
            }
            for name, entry in state._aux_tasks.items()
        }
        self._slack_action_handlers = list(state._slack_action_handlers)
        self._mcp_governors = dict(state._mcp_governors)
        self._discovered = state._discovered
        self._generation = state._generation
        self._live_context_generation = state._live_context_generation
        self._cli_ref = cli_ref
        self._lock = threading.RLock()
        self._transaction_tools_registered: List[str] = []
        self._transaction_commands_registered: List[str] = []
        self._transaction_mcp_governors_registered: List[str] = []
        self._frozen = False

    def _record_external_registration(self, surface: str, name: str) -> None:
        self._plugin_external_names.setdefault(surface, set()).add(name)
        if surface == "platform":
            self._plugin_platform_names.add(name)
        self._generation += 1

    def _freeze(self) -> None:
        """Make every staged manager container reject late registrations."""
        self._plugins = types.MappingProxyType(dict(self._plugins))
        self._hooks = types.MappingProxyType(
            {name: tuple(callbacks) for name, callbacks in self._hooks.items()}
        )
        self._middleware = types.MappingProxyType(
            {
                name: tuple(callbacks)
                for name, callbacks in self._middleware.items()
            }
        )
        self._plugin_tool_names = frozenset(self._plugin_tool_names)
        self._plugin_platform_names = frozenset(self._plugin_platform_names)
        self._plugin_external_names = types.MappingProxyType(
            {
                surface: frozenset(names)
                for surface, names in self._plugin_external_names.items()
            }
        )
        self._cli_commands = types.MappingProxyType(dict(self._cli_commands))
        self._plugin_commands = types.MappingProxyType(dict(self._plugin_commands))
        self._plugin_skills = types.MappingProxyType(dict(self._plugin_skills))
        self._aux_tasks = types.MappingProxyType(dict(self._aux_tasks))
        self._slack_action_handlers = tuple(self._slack_action_handlers)
        self._mcp_governors = types.MappingProxyType(dict(self._mcp_governors))
        self._frozen = True


class _PluginPublicationCapability:
    """Host-owned, fail-closed authority for one managed publication."""

    __slots__ = ("_published",)

    def __init__(self) -> None:
        object.__setattr__(self, "_published", False)

    def _activate(self) -> None:
        # This is the final, non-failing publication point. Bypass plugin-
        # controlled attribute hooks so activation can never be ambiguous.
        object.__setattr__(self, "_published", True)


def _run_cleanup_actions(
    actions: Iterable[tuple[str, Callable[[], None]]],
    *,
    primary: Optional[BaseException] = None,
    note: str = "Plugin registration cleanup failed",
) -> None:
    """Run every cleanup action, preserving an active primary exception."""
    failures: List[tuple[str, BaseException]] = []
    for label, action in actions:
        try:
            action()
        except BaseException as exc:
            logger.error("%s for %s", note, label, exc_info=True)
            failures.append((label, exc))
    if not failures:
        return
    if primary is not None:
        primary.add_note(
            f"{note}; attempted all cleanup actions. First failure: "
            f"{failures[0][0]}: {type(failures[0][1]).__name__}: {failures[0][1]}"
        )
        for label, exc in failures[1:]:
            primary.add_note(
                f"Additional cleanup failure: {label}: "
                f"{type(exc).__name__}: {exc}"
            )
        return
    first_label, first_failure = failures[0]
    for label, exc in failures[1:]:
        first_failure.add_note(
            f"Additional cleanup failure: {label}: {type(exc).__name__}: {exc}"
        )
    first_failure.add_note(
        f"{note}; attempted all cleanup actions. First failure at {first_label}."
    )
    raise first_failure


@dataclass(frozen=True)
class _PluginContextBinding:
    """One coherent authority target captured by a registration mutation."""

    registry: Any
    manager: Any
    external_registries: Any
    managed_generation: Optional[int]
    publication_capability: Optional[_PluginPublicationCapability] = None


class _RegistrationTransaction:
    """One isolated plugin registration or complete force-reload candidate."""

    def __init__(
        self,
        manager: "PluginManager",
        *,
        replace_owned: bool = False,
    ) -> None:
        from tools.registry import registry as tool_registry

        self.manager = manager
        self.replace_owned = replace_owned
        self.tool_registry = tool_registry
        self.registration_lock = threading.RLock()
        self.external_transactions = _external_registry_transactions()
        locks = [
            manager._lock,
            tool_registry._lock,
            *(
                self.external_transactions[surface].lock
                for surface in _EXTERNAL_COMMIT_SURFACES
            ),
        ]
        acquired_locks: List[Any] = []
        try:
            for lock in locks:
                lock.acquire()
                acquired_locks.append(lock)
            self.manager_before = _PluginManagerState(manager)
            self.tool_snapshot = tool_registry._take_transaction_snapshot()
            self.external_snapshots = {
                surface: self.external_transactions[surface].take_snapshot()
                for surface in _EXTERNAL_COMMIT_SURFACES
            }
        finally:
            _run_cleanup_actions(
                [
                    (f"constructor lock[{index}]", lock.release)
                    for index, lock in enumerate(reversed(acquired_locks))
                ],
                primary=sys.exc_info()[1],
            )

        self.manager_view = _PluginRegistrationView(
            self.manager_before,
            manager._cli_ref,
        )
        if replace_owned:
            self.manager_view._plugins = {}
            self.manager_view._hooks = {}
            self.manager_view._middleware = {}
            self.manager_view._plugin_tool_names = set()
            self.manager_view._plugin_platform_names = set()
            self.manager_view._plugin_external_names = {
                surface: set() for surface in _EXTERNAL_COMMIT_SURFACES
            }
            self.manager_view._cli_commands = {}
            self.manager_view._context_engine = None
            self.manager_view._plugin_commands = {}
            self.manager_view._plugin_skills = {}
            self.manager_view._aux_tasks = {}
            self.manager_view._slack_action_handlers = []
            self.manager_view._mcp_governors = {}
        self.manager_view._discovered = True

        removed_tool_names = (
            self.manager_before._plugin_tool_names if replace_owned else set()
        )
        removed_policy_names = set()
        if replace_owned:
            for loaded in self.manager_before._plugins.values():
                manifest = loaded.manifest
                plugin_id = manifest.key or manifest.name
                if manifest.source in {"user", "project", "bundled"}:
                    removed_policy_names.add(_directory_module_name(manifest))
                else:
                    removed_policy_names.add(plugin_id)
        self.tool_view = tool_registry._create_transaction_view(
            self.tool_snapshot,
            remove_names=removed_tool_names,
            remove_policy_names=removed_policy_names,
        )

        self.external_views = {}
        for surface, transaction in self.external_transactions.items():
            remove_keys = (
                self.manager_before._plugin_external_names.get(surface, set())
                if replace_owned
                else set()
            )
            self.external_views[surface] = transaction.create_view(
                self.external_snapshots[surface],
                remove_keys=remove_keys,
            )
        self.contexts: List["PluginContext"] = []
        self.context_generation = (
            self.manager_before._live_context_generation + 1
            if replace_owned
            else self.manager_before._live_context_generation
        )

    def context_targets(self) -> Dict[str, tuple[Any, Any]]:
        return {
            surface: (transaction, self.external_views[surface])
            for surface, transaction in self.external_transactions.items()
        }

    def commit(self) -> None:
        """Validate every baseline, then install all surfaces under one lock set."""
        prepared_tools = self.tool_registry._prepare_transaction_commit(
            self.tool_snapshot,
            self.tool_view,
        )
        prepared_external = {
            surface: transaction.prepare(
                self.external_snapshots[surface],
                self.external_views[surface],
            )
            for surface, transaction in self.external_transactions.items()
        }
        with self.manager_view._lock:
            self.manager_view._live_context_generation = self.context_generation
            next_manager = _PluginManagerState(self.manager_view)
            self.manager_view._freeze()

        revoke_generation = (
            self.manager_before._live_context_generation
            if self.replace_owned
            else None
        )
        context_locks = [self.manager._context_registration_lock]
        context_locks.extend(
            dict.fromkeys(context._registration_lock for context in self.contexts)
        )
        live_registry_locks = [
            self.manager._lock,
            self.tool_registry._lock,
            *(
                self.external_transactions[surface].lock
                for surface in _EXTERNAL_COMMIT_SURFACES
            ),
        ]
        acquired_locks: List[Any] = []
        revocation_started = False
        publication_started = False
        publication_capability = _PluginPublicationCapability()
        context_bindings_before: List[
            tuple[PluginContext, _PluginContextBinding]
        ] = []
        active_primary: Optional[BaseException] = None

        try:
            # Binding locks always precede every live manager/tool/external
            # registry lock. This blocks retained contexts before publication
            # can swap or restore their single coherent authority reference.
            for lock in context_locks:
                # A registration callback may wait for this commit thread.
                # Never wait back while it owns the shared context lock: abort
                # this candidate and let the live registration complete.
                if not lock.acquire(blocking=False):
                    raise _ForceSweepAbort(
                        "plugin publication conflicted with an active registration"
                    )
                acquired_locks.append(lock)
            if revoke_generation is not None:
                # Claim cleanup before the call: an injected BaseException may
                # occur after the manager adds the marker but before returning.
                revocation_started = True
                self.manager._begin_live_context_revocation(revoke_generation)
            for lock in live_registry_locks:
                lock.acquire()
                acquired_locks.append(lock)

            context_bindings_before = [
                (context, context._registration_binding)
                for context in self.contexts
            ]
            if self.manager._generation != self.manager_before._generation:
                raise RegistryTransactionConflict(
                    "manager",
                    self.manager_before._generation,
                    self.manager._generation,
                )
            self.tool_registry._validate_prepared_transaction_locked(
                self.tool_snapshot,
                prepared_tools,
            )
            for surface in _EXTERNAL_COMMIT_SURFACES:
                transaction = self.external_transactions[surface]
                transaction.validate_prepared_locked(
                    self.external_snapshots[surface],
                    prepared_external[surface],
                )

            publication_started = True
            self.tool_registry._install_prepared_transaction_locked(
                self.tool_snapshot,
                prepared_tools,
            )
            for surface in _EXTERNAL_COMMIT_SURFACES:
                transaction = self.external_transactions[surface]
                transaction.install_prepared_locked(
                    self.external_snapshots[surface],
                    prepared_external[surface],
                )
            for context in self.contexts:
                context._registration_binding = _PluginContextBinding(
                    registry=self.tool_registry,
                    manager=self.manager,
                    external_registries=None,
                    managed_generation=self.context_generation,
                    publication_capability=publication_capability,
                )
            self.manager._install_owned_state_locked(next_manager)
            publication_capability._activate()
        except BaseException as primary:
            active_primary = primary
            if publication_started and not publication_capability._published:
                rollback_errors: List[BaseException] = []
                rollback_steps = [
                    (
                        "tool",
                        lambda: self.tool_registry._restore_transaction_snapshot_exact_locked(
                            self.tool_snapshot,
                        ),
                    ),
                    *(
                        (
                            surface,
                            lambda surface=surface: self.external_transactions[
                                surface
                            ].restore_snapshot_locked(
                                self.external_snapshots[surface],
                            ),
                        )
                        for surface in _EXTERNAL_COMMIT_SURFACES
                    ),
                    *(
                        (
                            f"context[{index}]",
                            lambda binding=binding: setattr(
                                binding[0], "_registration_binding", binding[1]
                            ),
                        )
                        for index, binding in enumerate(context_bindings_before)
                    ),
                    (
                        "manager",
                        lambda: self.manager._restore_owned_state_locked(
                            self.manager_before,
                        ),
                    ),
                ]
                for surface, restore in rollback_steps:
                    try:
                        restore()
                    except BaseException as rollback_exc:
                        logger.critical(
                            "Plugin publication rollback failed for %s after %s",
                            surface,
                            type(primary).__name__,
                            exc_info=True,
                        )
                        rollback_errors.append(rollback_exc)
                if rollback_errors:
                    primary.__notes__ = getattr(primary, "__notes__", []) + [
                        "Plugin publication rollback failed; process-local "
                        "registries may be partially published. Check logs "
                        "for every rollback failure."
                    ]
            raise
        finally:
            cleanup_actions: List[tuple[str, Callable[[], None]]] = [
                (f"commit lock[{index}]", lock.release)
                for index, lock in enumerate(reversed(acquired_locks))
            ]
            if revocation_started:
                cleanup_actions.append(
                    (
                        "live context revocation",
                        lambda: self.manager._cancel_live_context_revocation(
                            revoke_generation
                        ),
                    )
                )
            _run_cleanup_actions(
                cleanup_actions,
                primary=sys.exc_info()[1] or active_primary,
            )


class _ForceSweepAbort(RuntimeError):
    """Internal sentinel: a force candidate failed and must not publish."""


# ---------------------------------------------------------------------------
# PluginContext  – handed to each plugin's ``register()`` function
# ---------------------------------------------------------------------------


class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(
        self,
        manifest: PluginManifest,
        manager: "PluginManager",
        *,
        registration_manager: Any = None,
        registration_registry: Any = None,
        registration_external_registries: Any = None,
        registration_lock: Any = None,
    ):
        self.manifest = manifest
        self._manager = manager
        # Live contexts share the manager lock; staged transaction contexts use
        # a transaction-local lock and join the live ordering only at commit.
        self._registration_lock = registration_lock or manager._context_registration_lock
        # Managed discovery supplies isolated registry transaction views.
        # Direct/runtime contexts leave them unset and target live registries.
        self._registration_binding = _PluginContextBinding(
            registry=registration_registry,
            manager=registration_manager or manager,
            external_registries=registration_external_registries,
            managed_generation=None,
        )
        # Lazy-built host-owned LLM facade — see ctx.llm property below.
        self._llm: Any = None
        self._subagent_lifecycle: Any = None

    def _registration_binding_locked(self) -> _PluginContextBinding:
        """Read and validate the one authority binding under its owner lock."""
        binding = self._registration_binding
        capability = binding.publication_capability
        if capability is not None and not capability._published:
            raise RuntimeError(
                f"Plugin context for {self.manifest.name!r} is not published"
            )
        return binding

    def _ensure_live_locked(
        self,
        binding: _PluginContextBinding,
        target: Any,
    ) -> None:
        if getattr(target, "_frozen", False):
            raise RuntimeError("plugin registration view is frozen")
        generation = binding.managed_generation
        if generation is None or not isinstance(target, PluginManager):
            return
        if target._live_context_generation != generation:
            raise RuntimeError(
                f"Plugin context for {self.manifest.name!r} is no longer live"
            )

    def _acquire_live_lease(
        self,
        binding: _PluginContextBinding,
    ) -> Callable[[], None]:
        target = binding.manager
        generation = binding.managed_generation
        if generation is None or not isinstance(target, PluginManager):
            return lambda: None
        return target._acquire_live_context_lease(self.manifest.name, generation)

    def _record_external_registration(
        self,
        binding: _PluginContextBinding,
        surface: str,
        name: str,
    ) -> None:
        target = binding.manager
        recorder = getattr(target, "_record_external_registration", None)
        if recorder is not None:
            with target._lock:
                self._ensure_live_locked(binding, target)
                recorder(surface, name)

    def _register_external_value(
        self,
        binding: _PluginContextBinding,
        surface: str,
        value: Any,
        live_register: Callable[[Any], Any],
    ) -> Optional[str]:
        targets = binding.external_registries
        if targets is None:
            release_live = self._acquire_live_lease(binding)
            try:
                result = live_register(value)
                if surface == "secret" and not result:
                    return None
                name = value.name
                if name:
                    self._record_external_registration(binding, surface, name)
            finally:
                release_live()
        else:
            transaction, staged = targets[surface]
            name = transaction.register(staged, value)
            if name:
                self._record_external_registration(binding, surface, name)
        return name

    # -- host-owned LLM access ----------------------------------------------

    @property
    def llm(self) -> Any:
        """Return the plugin's :class:`agent.plugin_llm.PluginLlm` facade.

        Lets trusted plugins run host-owned chat or structured completions
        against the user's active model and auth without bringing their
        own provider keys. Override capability (model, agent id, auth
        profile) is fail-closed by default and gated through
        ``plugins.entries.<plugin_id>.llm.*`` config keys.

        See :mod:`agent.plugin_llm` for the full surface."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            plugin_id = self.manifest.key or self.manifest.name
            self._llm = PluginLlm(plugin_id=plugin_id)
        return self._llm

    @property
    def subagent_lifecycle(self) -> Any:
        """Return the public, plugin-safe subagent lifecycle service.

        The service only resolves the active host-owned parent agent when a
        child is launched. Plugins receive serializable handles and immutable
        snapshots; they never receive a live agent or a private registry.
        """
        if self._subagent_lifecycle is None:
            from agent.subagent_lifecycle import (
                SubagentLifecycleService,
                get_active_subagent_parent,
            )
            self._subagent_lifecycle = SubagentLifecycleService(
                get_active_subagent_parent
            )
        return self._subagent_lifecycle

    # -- profile awareness --------------------------------------------------

    @property
    def profile_name(self) -> str:
        """Return the active Hermes profile name (e.g. ``"default"``).

        Derived from ``HERMES_HOME`` via
        :func:`hermes_cli.profiles.get_active_profile_name`, so it works in
        every execution context — interactive CLI, gateway, and
        kanban-spawned worker sessions alike — without depending on
        ``_cli_ref`` (which is ``None`` outside an interactive CLI run).

        Returns ``"default"`` for the default profile, the profile id when
        running under ``~/.hermes/profiles/<name>``, or ``"custom"`` when
        ``HERMES_HOME`` points somewhere unrecognized.
        """
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name()
        except Exception:
            return "default"

    # -- tool registration --------------------------------------------------

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        override: bool = False,
    ) -> None:
        """Register a tool in the global registry **and** track it as plugin-provided.

        Pass ``override=True`` to replace an existing built-in tool with the
        same name (e.g. swap the default ``browser_navigate`` for a custom
        CDP-backed implementation). Without it, attempting to register a name
        already claimed by a different toolset is rejected.

        ``override=True`` against a built-in tool requires the operator to
        opt in via ``plugins.entries.<plugin_id>.allow_tool_override: true``
        in config.yaml — mirrors the trust gate pattern used for
        ``ctx.llm`` provider/model overrides (#23194). Without that gate,
        any enabled plugin could silently replace a privileged built-in
        like ``shell_exec`` or ``write_file`` and exfiltrate everything
        the model invokes through it.
        """
        if override and not self._tool_override_allowed(name):
            plugin_id = self.manifest.key or self.manifest.name
            raise PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool "
                f"{name!r}. Set "
                f"plugins.entries.{plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )

        with self._registration_lock:
            binding = self._registration_binding_locked()
            target_registry = binding.registry
            if target_registry is None:
                from tools.registry import registry as target_registry

            target = binding.manager
            resolved_description = description or schema.get("description", "")
            release_live = self._acquire_live_lease(binding)
            try:
                target_registry.register(
                    name=name,
                    toolset=toolset,
                    schema=schema,
                    handler=handler,
                    check_fn=check_fn,
                    requires_env=requires_env,
                    is_async=is_async,
                    description=resolved_description,
                    emoji=emoji,
                    override=override,
                )
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    target._plugin_tool_names.add(name)
                    registered = getattr(
                        target, "_transaction_tools_registered", None
                    )
                    if registered is not None and name not in registered:
                        registered.append(name)
                    target._generation += 1
            finally:
                release_live()
        logger.debug(
            "Plugin %s registered tool: %s%s",
            self.manifest.name, name, " (override)" if override else "",
        )

    # -- override trust gate ------------------------------------------------

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """Return True if this plugin is configured to override built-in tools.

        Bundled plugins (shipped with Hermes core) are trusted by default —
        an override there is a deliberate maintainer choice, not a third-party
        plugin trying to elevate privilege. For every other source, require
        ``allow_tool_override: true`` under
        ``plugins.entries.<plugin_id>`` in config.yaml.
        """
        source = getattr(self.manifest, "source", "") or ""
        if source == "bundled":
            return True
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            # If we can't load config, fail closed — better to break the
            # override than silently grant it.
            return False
        plugin_id = self.manifest.key or self.manifest.name
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get(plugin_id) or {}
        return bool(entry.get("allow_tool_override", False))

    # -- governed MCP registration ----------------------------------------

    def _mcp_governance_allowed(self) -> bool:
        """Require an exact, explicit operator opt-in for this plugin."""

        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            if not isinstance(cfg, dict):
                return False
            plugins = cfg.get("plugins")
            if not isinstance(plugins, dict):
                return False
            entries = plugins.get("entries")
            if not isinstance(entries, dict):
                return False
            plugin_id = self.manifest.key or self.manifest.name
            entry = entries.get(plugin_id)
            return (
                isinstance(entry, dict)
                and entry.get("allow_mcp_governance") is True
            )
        except Exception:
            return False

    def register_mcp_governor(self, exact_server_name: str, policy: Any) -> None:
        """Claim one exact raw MCP server for a neutral two-phase policy.

        The claim is manager-owned and therefore participates in the same
        staging, conflict check, atomic commit, force-reload rollback, and read
        locking as the other plugin registration surfaces.
        """

        if (
            not isinstance(exact_server_name, str)
            or not exact_server_name
            or exact_server_name != exact_server_name.strip()
            or any(char in exact_server_name for char in "*?[]")
            or any(ord(char) < 32 or ord(char) == 127 for char in exact_server_name)
        ):
            raise MCPGovernanceRegistrationError(
                "MCP governor requires one exact non-empty raw server name"
            )
        if not self._mcp_governance_allowed():
            plugin_id = self.manifest.key or self.manifest.name
            raise MCPGovernanceRegistrationError(
                f"Plugin {plugin_id!r} requires explicit "
                "allow_mcp_governance: true operator configuration"
            )
        if not callable(getattr(policy, "prepare", None)) or not callable(
            getattr(policy, "finalize", None)
        ):
            raise MCPGovernanceRegistrationError(
                "MCP governor policy must implement prepare and finalize"
            )

        plugin_id = self.manifest.key or self.manifest.name
        registration = MCPGovernorRegistration(
            server_name=exact_server_name,
            plugin_id=plugin_id,
            policy=policy,
            secret_env_names=_manifest_private_secret_names(self.manifest),
        )
        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    existing = target._mcp_governors.get(exact_server_name)
                    if existing is not None:
                        raise MCPGovernanceRegistrationError(
                            "MCP server already has a governor"
                        )
                    target._mcp_governors[exact_server_name] = registration
                    registered = getattr(
                        target, "_transaction_mcp_governors_registered", None
                    )
                    if registered is not None:
                        registered.append(exact_server_name)
                    target._generation += 1
            finally:
                release_live()
        logger.debug(
            "Plugin %s registered MCP governor for exact server %s",
            self.manifest.name,
            exact_server_name,
        )

    # -- message injection --------------------------------------------------

    def inject_message(self, content: str, role: str = "user") -> bool:
        """Inject a message into the active conversation.

        If the agent is idle (waiting for user input), this starts a new turn.
        If the agent is running, this interrupts and injects the message.

        This enables plugins (e.g. remote control viewers, messaging bridges)
        to send messages into the conversation from external sources.

        Returns True if the message was queued successfully.
        """
        cli = self._manager._cli_ref
        if cli is None:
            logger.warning("inject_message: no CLI reference (not available in gateway mode)")
            return False

        msg = content if role == "user" else f"[{role}] {content}"

        if getattr(cli, "_agent_running", False):
            # Agent is mid-turn — interrupt with the message
            cli._interrupt_queue.put(msg)
        else:
            # Agent is idle — queue as next input
            cli._pending_input.put(msg)
        return True

    # -- CLI command registration --------------------------------------------

    def register_cli_command(
        self,
        name: str,
        help: str,
        setup_fn: Callable,
        handler_fn: Callable | None = None,
        description: str = "",
    ) -> None:
        """Register a CLI subcommand (e.g. ``hermes honcho ...``).

        The *setup_fn* receives an argparse subparser and should add any
        arguments/sub-subparsers.  If *handler_fn* is provided it is set
        as the default dispatch function via ``set_defaults(func=...)``."""
        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    target._cli_commands[name] = {
                        "name": name,
                        "help": help,
                        "description": description,
                        "setup_fn": setup_fn,
                        "handler_fn": handler_fn,
                        "plugin": self.manifest.name,
                    }
                    target._generation += 1
            finally:
                release_live()
        logger.debug("Plugin %s registered CLI command: %s", self.manifest.name, name)

    # -- slash command registration -------------------------------------------

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        args_hint: str = "",
    ) -> None:
        """Register a slash command (e.g. ``/lcm``) available in CLI and gateway sessions.

        The handler signature is ``fn(raw_args: str) -> str | None``.
        It may also be an async callable — the gateway dispatch handles both.

        Unlike ``register_cli_command()`` (which creates ``hermes <subcommand>``
        terminal commands), this registers in-session slash commands that users
        invoke during a conversation.

        ``args_hint`` is an optional short string (e.g. ``"<file>"`` or
        ``"dias:7 formato:json"``) used by gateway adapters to surface the
        command with an argument field — for example Discord's native slash
        command picker. Plugin commands without ``args_hint`` register as
        parameterless in Discord and still accept trailing text when invoked
        as free-form chat.

        Names conflicting with built-in commands are rejected with a warning.
        """
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Plugin '%s' tried to register a command with an empty name.",
                self.manifest.name,
            )
            return

        # Reject if it conflicts with a built-in command
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(
                    "Plugin '%s' tried to register command '/%s' which conflicts "
                    "with a built-in command. Skipping.",
                    self.manifest.name, clean,
                )
                return
        except Exception:
            pass  # If commands module isn't available, skip the check

        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    target._plugin_commands[clean] = {
                        "handler": handler,
                        "description": description or "Plugin command",
                        "plugin": self.manifest.name,
                        "args_hint": (args_hint or "").strip(),
                    }
                    registered = getattr(
                        target, "_transaction_commands_registered", None
                    )
                    if registered is not None and clean not in registered:
                        registered.append(clean)
                    target._generation += 1
            finally:
                release_live()
        logger.debug("Plugin %s registered command: /%s", self.manifest.name, clean)

    # -- tool dispatch -------------------------------------------------------

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call through the registry, with parent agent context.

        This is the public interface for plugin slash commands that need to call
        tools like ``delegate_task`` without reaching into framework internals.
        The parent agent (if available) is resolved automatically — plugins never
        need to access the agent directly.

        Args:
            tool_name: Registry name of the tool (e.g. ``"delegate_task"``).
            args: Tool arguments dict (same as what the model would pass).
            **kwargs: Extra keyword args forwarded to the registry dispatch.

        Returns:
            JSON string from the tool handler (same format as model tool calls).
        """
        from tools.registry import registry

        # Wire up parent agent context when available (CLI mode).
        # In gateway mode _cli_ref is None — tools degrade gracefully
        # (workspace hints fall back to TERMINAL_CWD, no spinner).
        if "parent_agent" not in kwargs:
            cli = self._manager._cli_ref
            agent = getattr(cli, "agent", None) if cli else None
            if agent is not None:
                kwargs["parent_agent"] = agent

        return registry.dispatch(tool_name, args, **kwargs)

    # -- context engine registration -----------------------------------------

    def register_context_engine(self, engine) -> None:
        """Register a context engine to replace the built-in ContextCompressor.

        Only one context engine plugin is allowed. If a second plugin tries
        to register one, it is rejected with a warning.

        The engine must be an instance of ``agent.context_engine.ContextEngine``.
        """
        # Defer the import to avoid circular deps at module level
        from agent.context_engine import ContextEngine
        if not isinstance(engine, ContextEngine):
            logger.warning(
                "Plugin '%s' tried to register a context engine that does not "
                "inherit from ContextEngine. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    if target._context_engine is not None:
                        logger.warning(
                            "Plugin '%s' tried to register a context engine, but one is "
                            "already registered. Only one context engine plugin is allowed.",
                            self.manifest.name,
                        )
                        return
                    target._context_engine = engine
                    target._generation += 1
            finally:
                release_live()
        logger.info(
            "Plugin '%s' registered context engine: %s",
            self.manifest.name, engine.name,
        )

    # -- image gen provider registration ------------------------------------

    def register_image_gen_provider(self, provider) -> None:
        """Register an image generation backend.

        ``provider`` must be an instance of
        :class:`agent.image_gen_provider.ImageGenProvider`. The
        ``provider.name`` attribute is what ``image_gen.provider`` in
        ``config.yaml`` matches against when routing ``image_generate``
        tool calls.
        """
        from agent.image_gen_provider import ImageGenProvider
        from agent.image_gen_registry import register_provider

        if not isinstance(provider, ImageGenProvider):
            logger.warning(
                "Plugin '%s' tried to register an image_gen provider that does "
                "not inherit from ImageGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            self._register_external_value(
                binding, "image_gen", provider, register_provider
            )
        logger.info(
            "Plugin '%s' registered image_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- dashboard auth provider registration --------------------------------

    def register_dashboard_auth_provider(self, provider) -> None:
        """Register a dashboard authentication provider.

        ``provider`` must be an instance of
        :class:`hermes_cli.dashboard_auth.DashboardAuthProvider`. Used by
        the dashboard OAuth auth gate, which engages when the dashboard
        binds to a non-loopback host without ``--insecure``.

        Misbehaving providers (wrong type, duplicate name) are logged at
        WARNING and silently ignored — never raised — so a broken plugin
        cannot crash the host. Same convention as
        ``register_image_gen_provider``.
        """
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider, register_provider,
        )

        if not isinstance(provider, DashboardAuthProvider):
            logger.warning(
                "Plugin '%s' tried to register a dashboard-auth provider "
                "that does not inherit from DashboardAuthProvider. Ignoring.",
                self.manifest.name,
            )
            return
        try:
            with self._registration_lock:
                binding = self._registration_binding_locked()
                name = self._register_external_value(
                    binding,
                    "dashboard",
                    provider,
                    register_provider,
                )
        except (TypeError, ValueError) as e:
            logger.warning(
                "Plugin '%s' failed to register dashboard-auth provider "
                "%r: %s",
                self.manifest.name, getattr(provider, "name", "?"), e,
            )
            return
        if name is None:
            return
        logger.info(
            "Plugin '%s' registered dashboard-auth provider: %s (%s)",
            self.manifest.name, provider.name, provider.display_name,
        )

    # -- video gen provider registration -------------------------------------

    def register_video_gen_provider(self, provider) -> None:
        """Register a video generation backend.

        ``provider`` must be an instance of
        :class:`agent.video_gen_provider.VideoGenProvider`. The
        ``provider.name`` attribute is what ``video_gen.provider`` in
        ``config.yaml`` matches against when routing ``video_generate``
        tool calls.
        """
        from agent.video_gen_provider import VideoGenProvider
        from agent.video_gen_registry import register_provider as _register_video_provider

        if not isinstance(provider, VideoGenProvider):
            logger.warning(
                "Plugin '%s' tried to register a video_gen provider that does "
                "not inherit from VideoGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            self._register_external_value(
                binding,
                "video_gen",
                provider,
                _register_video_provider,
            )
        logger.info(
            "Plugin '%s' registered video_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- web search/extract provider registration ----------------------------

    def register_web_search_provider(self, provider) -> None:
        """Register a web search/extract backend.

        ``provider`` must be an instance of
        :class:`agent.web_search_provider.WebSearchProvider`. The
        ``provider.name`` attribute is what ``web.search_backend`` /
        ``web.extract_backend`` / ``web.backend`` in ``config.yaml``
        matches against when routing ``web_search`` / ``web_extract``
        tool calls.
        """
        from agent.web_search_provider import WebSearchProvider
        from agent.web_search_registry import register_provider as _register_web_provider

        if not isinstance(provider, WebSearchProvider):
            logger.warning(
                "Plugin '%s' tried to register a web provider that does "
                "not inherit from WebSearchProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            self._register_external_value(
                binding, "web", provider, _register_web_provider
            )
        logger.info(
            "Plugin '%s' registered web provider: %s",
            self.manifest.name, provider.name,
        )

    # -- browser provider registration ---------------------------------------

    def register_browser_provider(self, provider) -> None:
        """Register a cloud browser backend.

        ``provider`` must be an instance of
        :class:`agent.browser_provider.BrowserProvider`. The
        ``provider.name`` attribute is what ``browser.cloud_provider`` in
        ``config.yaml`` matches against when routing cloud-mode
        ``browser_*`` tool calls.

        Mirrors :meth:`register_web_search_provider` exactly — same
        registration shape, same gating, same logging. The browser
        subsystem's dispatcher (:func:`tools.browser_tool._get_cloud_provider`)
        consults the registry built up by these calls.
        """
        from agent.browser_provider import BrowserProvider
        from agent.browser_registry import register_provider as _register_browser_provider

        if not isinstance(provider, BrowserProvider):
            logger.warning(
                "Plugin '%s' tried to register a browser provider that does "
                "not inherit from BrowserProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            self._register_external_value(
                binding,
                "browser",
                provider,
                _register_browser_provider,
            )
        logger.info(
            "Plugin '%s' registered browser provider: %s",
            self.manifest.name, provider.name,
        )

    # -- secret source registration -------------------------------------------

    def register_secret_source(self, source) -> None:
        """Register an external secret-manager backend.

        ``source`` must be an instance of
        :class:`agent.secret_sources.base.SecretSource`.  Registered
        sources run during ``load_hermes_dotenv()`` startup — after
        ``~/.hermes/.env`` loads, before Hermes reads credentials — when
        their ``secrets.<source.name>`` config section is enabled.  The
        orchestrator (``agent.secret_sources.registry.apply_all``) owns
        ordering, mapped-vs-bulk precedence, conflict warnings, and
        provenance; the source only fetches.

        NOTE ON TIMING: plugin discovery happens later in startup than
        the first ``load_hermes_dotenv()`` call, so a plugin-registered
        source is not consulted by the initial env load of the process
        that discovers it.  It IS consulted by every subsequently
        spawned Hermes process (gateway children, cron sessions,
        subagents), and immediately after a
        ``reset_secret_source_cache()`` re-pull.  Plugin sources are
        therefore best for supplying credentials to the running fleet;
        the bundled sources cover first-process bootstrap.

        Contract requirements (rejected with a warning otherwise):
        inherit from ``SecretSource``, ``api_version`` matching
        ``SECRET_SOURCE_API_VERSION``, lowercase unique ``name``,
        ``shape`` of ``"mapped"`` or ``"bulk"``, unique ``scheme`` (when
        set), and a ``fetch()`` that never raises and never prompts.
        See the base-module docstring for the full contract.
        """
        from agent.secret_sources.base import SecretSource
        from agent.secret_sources.registry import register_source

        if not isinstance(source, SecretSource):
            logger.warning(
                "Plugin '%s' tried to register a secret source that does "
                "not inherit from SecretSource. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            registered = self._register_external_value(
                binding, "secret", source, register_source
            )
        if registered:
            logger.info(
                "Plugin '%s' registered secret source: %s",
                self.manifest.name, source.name,
            )

    # -- TTS provider registration -------------------------------------------

    def register_tts_provider(self, provider) -> None:
        """Register a text-to-speech backend.

        ``provider`` must be an instance of
        :class:`agent.tts_provider.TTSProvider`. The ``provider.name``
        attribute is what ``tts.provider`` in ``config.yaml`` matches
        against when routing ``text_to_speech`` tool calls — **but
        only when**:

        1. ``provider.name`` is NOT a built-in TTS provider name
           (``edge``, ``openai``, ``elevenlabs``, …). Built-ins always
           win — the registry rejects shadowing names with a warning.
        2. There is NO ``tts.providers.<name>: type: command`` entry
           with the same name. Command-providers (PR #17843) win on
           name collision because config is more local than plugin
           install.

        Coexists with the command-provider registry rather than
        replacing it — see issue #30398 for the full design rationale.
        """
        from agent.tts_provider import TTSProvider
        from agent.tts_registry import register_provider as _register_tts_provider

        if not isinstance(provider, TTSProvider):
            logger.warning(
                "Plugin '%s' tried to register a TTS provider that does "
                "not inherit from TTSProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            name = self._register_external_value(
                binding,
                "tts",
                provider,
                _register_tts_provider,
            )
        if name is None:
            return
        logger.info(
            "Plugin '%s' registered TTS provider: %s",
            self.manifest.name, provider.name,
        )

    # -- transcription (STT) provider registration ---------------------------

    def register_transcription_provider(self, provider) -> None:
        """Register a speech-to-text backend.

        ``provider`` must be an instance of
        :class:`agent.transcription_provider.TranscriptionProvider`.
        The ``provider.name`` attribute is what ``stt.provider`` in
        ``config.yaml`` matches against when routing
        :func:`tools.transcription_tools.transcribe_audio` calls —
        **but only when**:

        1. ``provider.name`` is NOT a built-in STT provider name
           (``local``, ``local_command``, ``groq``, ``openai``,
           ``mistral``, ``xai``). Built-ins always win — the registry
           rejects shadowing names with a warning.
        2. There is NO ``stt.providers.<name>: type: command`` entry
           with the same name. Command-providers win on name
           collision because config is more local than plugin install
           — same precedence rule as TTS.

        Coexists with the in-tree dispatcher and the STT
        command-provider registry rather than replacing them. The 6
        built-in STT backends keep their native implementations in
        ``tools/transcription_tools.py``; this hook is for *new* Python
        engines (OpenRouter, SenseAudio, Gemini-STT, custom proprietary
        backends).
        """
        from agent.transcription_provider import TranscriptionProvider
        from agent.transcription_registry import register_provider as _register_stt_provider

        if not isinstance(provider, TranscriptionProvider):
            logger.warning(
                "Plugin '%s' tried to register a transcription provider that "
                "does not inherit from TranscriptionProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_lock:
            binding = self._registration_binding_locked()
            name = self._register_external_value(
                binding,
                "stt",
                provider,
                _register_stt_provider,
            )
        if name is None:
            return
        logger.info(
            "Plugin '%s' registered transcription provider: %s",
            self.manifest.name, provider.name,
        )

    # -- platform adapter registration ---------------------------------------

    def register_platform(
        self,
        name: str,
        label: str,
        adapter_factory: Callable,
        check_fn: Callable,
        validate_config: Callable | None = None,
        required_env: list | None = None,
        install_hint: str = "",
        **entry_kwargs: Any,
    ) -> None:
        """Register a gateway platform adapter.

        The adapter_factory receives a ``PlatformConfig`` and returns a
        ``BasePlatformAdapter`` subclass instance.  The gateway calls
        ``check_fn()`` before instantiation to verify dependencies.

        Extra keyword arguments are forwarded to ``PlatformEntry`` (e.g.
        ``setup_fn``, ``emoji``, ``allowed_users_env``, ``platform_hint``).
        Unknown keys raise TypeError from the dataclass constructor.

        Example::

            ctx.register_platform(
                name="irc",
                label="IRC",
                adapter_factory=lambda cfg: IRCAdapter(cfg),
                check_fn=lambda: True,
                emoji="💬",
                setup_fn=irc_interactive_setup,
            )
        """
        from gateway.platform_registry import platform_registry, PlatformEntry

        entry_kwargs.setdefault("plugin_name", self.manifest.name)
        entry = PlatformEntry(
            name=name,
            label=label,
            adapter_factory=adapter_factory,
            check_fn=check_fn,
            validate_config=validate_config,
            required_env=required_env or [],
            install_hint=install_hint,
            source="plugin",
            **entry_kwargs,
        )
        with self._registration_lock:
            binding = self._registration_binding_locked()
            targets = binding.external_registries
            if targets is None:
                release_live = self._acquire_live_lease(binding)
                try:
                    platform_registry.register(entry)
                    self._record_external_registration(
                        binding, "platform", name
                    )
                finally:
                    release_live()
            else:
                _transaction, staged = targets["platform"]
                staged.register(entry)
                self._record_external_registration(binding, "platform", name)

        logger.debug(
            "Plugin %s registered platform: %s",
            self.manifest.name,
            name,
        )

    # -- slack action handler registration ----------------------------------

    def register_slack_action_handler(
        self,
        action_id: Any,
        callback: Callable,
    ) -> None:
        """Register a Slack Block Kit action handler from a plugin.

        Hermes' Slack adapter wires registered handlers into its
        ``slack_bolt.AsyncApp`` at connect time. The callback is invoked
        when a user clicks a button (or interacts with another Block Kit
        action element) whose ``action_id`` matches.

        Callback signature follows the slack_bolt convention::

            async def handler(ack, body, action) -> None:
                await ack()  # required, within 3 seconds
                ...

        Args:
            action_id: Whatever ``slack_bolt.App.action()`` accepts —
                a literal ``action_id`` string, a compiled ``re.Pattern``
                for matching multiple ids, or a constraint dict
                (e.g. ``{"action_id": "...", "block_id": "..."}``).
            callback: Async callable receiving ``(ack, body, action)``.

        Raises:
            ValueError: if ``callback`` is not callable, or ``action_id``
                is empty/None.

        Example::

            async def _on_approve(ack, body, action):
                await ack()
                # apply some workflow keyed on action["value"]

            ctx.register_slack_action_handler("inbox_sweep_approve", _on_approve)
        """
        if not callable(callback):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register a Slack "
                f"action handler with a non-callable callback."
            )
        if action_id is None or (isinstance(action_id, str) and not action_id.strip()):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register a Slack "
                f"action handler with an empty action_id."
            )
        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    target._slack_action_handlers.append(
                        (action_id, callback, self.manifest.name)
                    )
                    target._generation += 1
            finally:
                release_live()
        logger.debug(
            "Plugin %s registered Slack action handler: %s",
            self.manifest.name,
            action_id,
        )

    # -- hook registration --------------------------------------------------

    # -- auxiliary task registration ---------------------------------------

    def register_auxiliary_task(
        self,
        key: str,
        *,
        display_name: str,
        description: str,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a plugin-defined auxiliary LLM task.

        Auxiliary tasks are LLM-backed side jobs (vision analysis, web extraction,
        compression, smart-approval, etc.) that route through ``auxiliary_client.py``.
        Each task has its own ``auxiliary.<key>`` config block where users can
        pin a provider/model independent of the main chat model.

        Plugins use this to declare their own auxiliary tasks without touching
        core files. After registration, the task:

          - Appears in the ``hermes model → Configure auxiliary models`` picker
          - Has its provider/model/base_url/api_key bridged from config.yaml to
            ``AUXILIARY_<KEY_UPPER>_*`` env vars at gateway startup
          - Gets default routing fields (provider="auto", model="", etc.) merged
            into loaded configs so ``cfg.get("auxiliary", {}).get(key)`` works

        Args:
            key: stable task key (snake_case). Used in config ``auxiliary.<key>``
                and env vars ``AUXILIARY_<KEY_UPPER>_*``. Must not shadow a
                built-in task key (vision, compression, web_extract, approval,
                mcp, title_generation, skills_hub, curator).
            display_name: human-readable name shown in the picker.
            description: short one-line description shown next to the name.
            defaults: optional dict of default routing fields. Recognized keys:
                ``provider`` (default "auto"), ``model`` (default ""),
                ``base_url`` (default ""), ``api_key`` (default ""),
                ``timeout`` (default 60), ``extra_body`` (default {}),
                plus any task-specific extras (e.g. ``download_timeout``).
                Unknown keys are preserved verbatim — the plugin owns the
                schema for its own task.

        Raises:
            ValueError: if *key* is empty, contains invalid characters, or
                shadows a built-in auxiliary task key.

        Example:
            ctx.register_auxiliary_task(
                key="memory_retain_filter",
                display_name="Memory retain filter",
                description="hindsight pre-retain dedup/extract",
                defaults={"provider": "auto", "timeout": 30},
            )
        """
        # Validate key shape
        if not key or not isinstance(key, str):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register auxiliary task "
                f"with invalid key {key!r}"
            )
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(
                f"Plugin '{self.manifest.name}' auxiliary task key {key!r} "
                f"must contain only alphanumeric characters and underscores"
            )

        # Lazy import to avoid circular: hermes_cli.main imports plugins indirectly
        from hermes_cli.main import _AUX_TASKS as _BUILTIN_AUX_TASKS

        builtin_keys = {k for k, _name, _desc in _BUILTIN_AUX_TASKS}
        if key in builtin_keys:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — that key is reserved for a built-in task. "
                f"Pick a plugin-namespaced key (e.g. '{self.manifest.name}_{key}')."
            )

        # Normalize defaults — plugin owns the schema, but we ensure routing
        # fields exist with sensible types so consumers don't crash.
        merged_defaults: Dict[str, Any] = {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
        }
        if defaults:
            for k, v in defaults.items():
                merged_defaults[k] = v

        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    existing = target._aux_tasks.get(key)
                    if (
                        existing is not None
                        and existing.get("plugin") != self.manifest.name
                    ):
                        raise ValueError(
                            f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                            f"{key!r} — already registered by plugin "
                            f"'{existing.get('plugin')}'"
                        )
                    target._aux_tasks[key] = {
                        "key": key,
                        "display_name": display_name,
                        "description": description,
                        "defaults": merged_defaults,
                        "plugin": self.manifest.name,
                    }
                    target._generation += 1
            finally:
                release_live()
        logger.debug(
            "Plugin %s registered auxiliary task: %s (%s)",
            self.manifest.name,
            key,
            display_name,
        )

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a lifecycle hook callback.

        Unknown hook names produce a warning but are still stored so
        forward-compatible plugins don't break.
        """
        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Plugin '%s' registered unknown hook '%s' "
                "(valid: %s)",
                self.manifest.name,
                hook_name,
                ", ".join(sorted(VALID_HOOKS)),
            )
        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    target._hooks.setdefault(hook_name, []).append(callback)
                    target._generation += 1
            finally:
                release_live()
        logger.debug("Plugin %s registered hook: %s", self.manifest.name, hook_name)

    # -- middleware registration -------------------------------------------

    def register_middleware(self, kind: str, callback: Callable) -> None:
        """Register a behavior-changing middleware callback.

        Middleware is separate from observer hooks: request middleware may
        rewrite the effective payload, and execution middleware may wrap the
        real callback. Unknown kinds are stored for forward compatibility but
        warned so plugin authors can catch typos.
        """
        if kind not in VALID_MIDDLEWARE:
            logger.warning(
                "Plugin '%s' registered unknown middleware '%s' "
                "(valid: %s)",
                self.manifest.name,
                kind,
                ", ".join(sorted(VALID_MIDDLEWARE)),
            )
        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    target._middleware.setdefault(kind, []).append(callback)
                    target._generation += 1
            finally:
                release_live()
        logger.debug("Plugin %s registered middleware: %s", self.manifest.name, kind)

    # -- skill registration -------------------------------------------------

    def register_skill(
        self,
        name: str,
        path: Path,
        description: str = "",
    ) -> None:
        """Register a read-only skill provided by this plugin.

        The skill becomes resolvable as ``'<plugin_name>:<name>'`` via
        ``skill_view()``.  It does **not** enter the flat
        ``~/.hermes/skills/`` tree and is **not** listed in the system
        prompt's ``<available_skills>`` index — plugin skills are
        opt-in explicit loads only.

        Raises:
            ValueError: if *name* contains ``':'`` or invalid characters.
            FileNotFoundError: if *path* does not exist.
        """
        from agent.skill_utils import _NAMESPACE_RE

        if ":" in name:
            raise ValueError(
                f"Skill name '{name}' must not contain ':' "
                f"(the namespace is derived from the plugin name "
                f"'{self.manifest.name}' automatically)."
            )
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(
                f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+."
            )
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")

        qualified = f"{self.manifest.name}:{name}"
        with self._registration_lock:
            binding = self._registration_binding_locked()
            target = binding.manager
            release_live = self._acquire_live_lease(binding)
            try:
                with target._lock:
                    self._ensure_live_locked(binding, target)
                    target._plugin_skills[qualified] = {
                        "path": path,
                        "plugin": self.manifest.name,
                        "bare_name": name,
                        "description": description,
                    }
                    target._generation += 1
            finally:
                release_live()
        logger.debug(
            "Plugin %s registered skill: %s",
            self.manifest.name, qualified,
        )


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self) -> None:
        # Discovery writers serialize here while readers keep using the live
        # generation. Plugin callbacks are never invoked under registry locks.
        self._discovery_lock = threading.RLock()
        self._context_registration_lock = threading.RLock()
        self._lock = threading.RLock()
        self._generation = 0
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._middleware: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._plugin_platform_names: Set[str] = set()
        self._plugin_external_names: Dict[str, Set[str]] = {
            surface: set() for surface in _EXTERNAL_COMMIT_SURFACES
        }
        self._cli_commands: Dict[str, dict] = {}
        self._context_engine = None  # Set by a plugin via register_context_engine()
        self._plugin_commands: Dict[str, dict] = {}  # Slash commands registered by plugins
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        # Plugin skill registry: qualified name → metadata dict.
        self._plugin_skills: Dict[str, Dict[str, Any]] = {}
        # Plugin-registered auxiliary tasks: key → {key, display_name,
        # description, defaults, plugin}. See PluginContext.register_auxiliary_task.
        self._aux_tasks: Dict[str, Dict[str, Any]] = {}
        # Slack Block Kit action handlers registered by plugins. Each entry
        # is (matcher, callback, plugin_name); the Slack adapter wires them
        # into its slack_bolt App at connect() time. ``matcher`` is whatever
        # ``app.action()`` accepts (a literal action_id string, a compiled
        # ``re.Pattern``, or a constraint dict); ``callback`` is an async
        # function with the slack_bolt signature ``(ack, body, action)``.
        self._slack_action_handlers: List[tuple] = []
        self._live_context_generation = 0
        self._live_context_condition = threading.Condition(threading.RLock())
        self._live_context_active: Dict[int, int] = {}
        self._live_context_active_by_thread: Dict[tuple[int, int], int] = {}
        self._live_context_revoking: set[int] = set()
        # Exact raw MCP server name -> immutable policy attribution. Governors
        # are absent from the model schema and are read only at invocation.
        self._mcp_governors: Dict[str, MCPGovernorRegistration] = {}

    def _record_external_registration(self, surface: str, name: str) -> None:
        self._plugin_external_names.setdefault(surface, set()).add(name)
        if surface == "platform":
            self._plugin_platform_names.add(name)
        self._generation += 1

    def _acquire_live_context_lease(
        self,
        plugin_name: str,
        generation: int,
    ) -> Callable[[], None]:
        owner_thread_id = threading.get_ident()
        owner_key = (generation, owner_thread_id)
        with self._live_context_condition:
            if (
                self._live_context_generation != generation
                or generation in self._live_context_revoking
            ):
                raise RuntimeError(
                    f"Plugin context for {plugin_name!r} is no longer live"
                )
            self._live_context_active[generation] = (
                self._live_context_active.get(generation, 0) + 1
            )
            self._live_context_active_by_thread[owner_key] = (
                self._live_context_active_by_thread.get(owner_key, 0) + 1
            )

        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            with self._live_context_condition:
                active = self._live_context_active.get(generation, 0)
                if active <= 1:
                    self._live_context_active.pop(generation, None)
                    self._live_context_condition.notify_all()
                else:
                    self._live_context_active[generation] = active - 1
                thread_active = self._live_context_active_by_thread.get(owner_key, 0)
                if thread_active <= 1:
                    self._live_context_active_by_thread.pop(owner_key, None)
                else:
                    self._live_context_active_by_thread[owner_key] = thread_active - 1

        return release

    def _reject_reentrant_force_reload(self, generation: int) -> None:
        with self._live_context_condition:
            if self._live_context_active_by_thread.get(
                (generation, threading.get_ident()),
                0,
            ):
                raise RuntimeError(
                    "force reload cannot run from a live plugin registration"
                )

    def _begin_live_context_revocation(self, generation: int) -> None:
        with self._live_context_condition:
            self._reject_reentrant_force_reload(generation)
            if self._live_context_active.get(generation, 0):
                raise _ForceSweepAbort(
                    "plugin force reload conflicted with an active live registration"
                )
            self._live_context_revoking.add(generation)

    def _cancel_live_context_revocation(self, generation: int) -> None:
        with self._live_context_condition:
            self._live_context_revoking.discard(generation)
            self._live_context_condition.notify_all()

    def _snapshot_owned_state(self) -> _PluginManagerState:
        """Take a non-aliasing checkpoint under the manager state lock."""
        with self._lock:
            return _PluginManagerState(self)

    def _install_owned_state_locked(self, state: Any) -> None:
        """Install already-independent manager state while ``_lock`` is held."""
        self._plugins = state._plugins
        self._hooks = state._hooks
        self._middleware = state._middleware
        self._plugin_tool_names = state._plugin_tool_names
        self._plugin_platform_names = state._plugin_platform_names
        self._plugin_external_names = state._plugin_external_names
        self._cli_commands = state._cli_commands
        self._context_engine = state._context_engine
        self._plugin_commands = state._plugin_commands
        self._plugin_skills = state._plugin_skills
        self._aux_tasks = state._aux_tasks
        self._slack_action_handlers = state._slack_action_handlers
        self._mcp_governors = state._mcp_governors
        self._discovered = state._discovered
        self._live_context_generation = state._live_context_generation
        self._generation = max(self._generation, state._generation) + 1

    def _restore_owned_state_locked(self, state: Any) -> None:
        """Restore a transaction checkpoint exactly while ``_lock`` is held."""
        self._plugins = state._plugins
        self._hooks = state._hooks
        self._middleware = state._middleware
        self._plugin_tool_names = state._plugin_tool_names
        self._plugin_platform_names = state._plugin_platform_names
        self._plugin_external_names = state._plugin_external_names
        self._cli_commands = state._cli_commands
        self._context_engine = state._context_engine
        self._plugin_commands = state._plugin_commands
        self._plugin_skills = state._plugin_skills
        self._aux_tasks = state._aux_tasks
        self._slack_action_handlers = state._slack_action_handlers
        self._mcp_governors = state._mcp_governors
        self._discovered = state._discovered
        self._live_context_generation = state._live_context_generation
        self._generation = state._generation

    def discover_and_load(self, force: bool = False) -> None:
        """Load plugins, publishing a force reload as one complete generation."""
        if force:
            self._reject_reentrant_force_reload(self._live_context_generation)
            self._force_discover_and_load()
            return
        with self._discovery_lock:
            with self._lock:
                if self._discovered:
                    return
            if env_var_enabled("HERMES_SAFE_MODE"):
                logger.info("HERMES_SAFE_MODE=1 - plugin discovery skipped")
                with self._lock:
                    self._discovered = True
                    self._generation += 1
                return

            with self._lock:
                self._discovered = True
                self._generation += 1
            try:
                self._discover_and_load_inner()
            except BaseException:
                # Initial discovery publishes the complete active-winner deny
                # set before importing any plugin. Keep that safe overblock if
                # a later plugin raises BaseException; partial registrations
                # remain committed and the next call retries the sweep.
                with self._lock:
                    self._discovered = False
                    self._generation += 1
                raise

    def _force_discover_and_load(self) -> None:
        """Stage and publish a force generation under discovery serialization."""
        with self._discovery_lock:
            if env_var_enabled("HERMES_SAFE_MODE"):
                logger.info("HERMES_SAFE_MODE=1 - plugin discovery skipped")
                with self._lock:
                    self._discovered = True
                    self._generation += 1
                return

            # Private names are a deny-only security policy. Publish the
            # complete candidate set inside _discover_and_load_inner() before
            # arbitrary plugin code runs, then restore this exact checkpoint
            # if the candidate generation cannot commit.
            private_policy_snapshot = snapshot_private_secret_policy()
            transaction = _RegistrationTransaction(
                self,
                replace_owned=True,
            )
            module_snapshot = _snapshot_module_namespace(_NS_PARENT)
            try:
                self._discover_and_load_inner(transaction)
                transaction.commit()
            except (_ForceSweepAbort, RegistryTransactionConflict) as exc:
                restore_private_secret_policy(private_policy_snapshot)
                _restore_module_namespace(_NS_PARENT, module_snapshot)
                if isinstance(exc, RegistryTransactionConflict):
                    logger.warning(
                        "Plugin force-reload commit conflict: "
                        "surface=%s expected_generation=%d "
                        "actual_generation=%d; retaining prior generation",
                        exc.surface,
                        exc.expected_generation,
                        exc.actual_generation,
                    )
                return
            except BaseException:
                restore_private_secret_policy(private_policy_snapshot)
                _restore_module_namespace(_NS_PARENT, module_snapshot)
                raise

    def _discover_and_load_inner(
        self,
        transaction: Optional[_RegistrationTransaction] = None,
    ) -> None:
        """The actual discovery sweep — see :meth:`discover_and_load`."""
        manifests: List[PluginManifest] = []
        target = transaction.manager_view if transaction is not None else self

        # 1. Bundled plugins (<repo>/plugins/<name>/)
        #
        # Repo-shipped plugins live next to hermes_cli/. Two layouts are
        # supported (see ``_scan_directory`` for details):
        #
        #   - flat: ``plugins/disk-cleanup/plugin.yaml`` (standalone)
        #   - category: ``plugins/image_gen/openai/plugin.yaml`` (backend)
        #
        # ``memory/``, ``context_engine/``, and ``model-providers/`` are
        # skipped at the top level — they have their own discovery systems
        # (plugins/memory/__init__.py, providers/__init__.py). ``platforms/``
        # is a category holding platform adapters (scanned one level deeper
        # below).
        repo_plugins = get_bundled_plugins_dir()
        logger.debug("Scanning bundled plugins: %s", repo_plugins)
        bundled = self._scan_directory(
            repo_plugins,
            source="bundled",
            skip_names={"memory", "context_engine", "platforms", "model-providers"},
        )
        logger.debug("  bundled (top-level): %d manifest(s)", len(bundled))
        manifests.extend(bundled)
        bundled_platforms = self._scan_directory(
            repo_plugins / "platforms", source="bundled"
        )
        logger.debug("  bundled/platforms: %d manifest(s)", len(bundled_platforms))
        manifests.extend(bundled_platforms)

        # 2. User plugins (~/.hermes/plugins/)
        user_dir = get_hermes_home() / "plugins"
        logger.debug("Scanning user plugins: %s", user_dir)
        user_manifests = self._scan_directory(user_dir, source="user")
        logger.debug("  user: %d manifest(s)", len(user_manifests))
        manifests.extend(user_manifests)

        # 3. Project plugins (./.hermes/plugins/)
        if _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            project_dir = Path.cwd() / ".hermes" / "plugins"
            logger.debug("Scanning project plugins: %s", project_dir)
            project_manifests = self._scan_directory(project_dir, source="project")
            logger.debug("  project: %d manifest(s)", len(project_manifests))
            manifests.extend(project_manifests)
        else:
            logger.debug(
                "Project plugins disabled (set HERMES_ENABLE_PROJECT_PLUGINS=1 to enable)"
            )

        # 4. Pip / entry-point plugins
        ep_manifests = self._scan_entry_points()
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)

        # Load each manifest (skip user-disabled plugins).
        # Later sources override earlier ones on key collision — user
        # plugins take precedence over bundled, project plugins take
        # precedence over user. Dedup here so we only load the final
        # winner. Keys are path-derived (``image_gen/openai``,
        # ``disk-cleanup``) so ``tts/openai`` and ``image_gen/openai``
        # don't collide even when both manifests say ``name: openai``.
        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = opt-in default (nothing enabled)
        winners: Dict[str, PluginManifest] = {}
        for manifest in manifests:
            winners[manifest.key or manifest.name] = manifest

        # Build the complete replacement while the old immutable deny set
        # remains live. Publish once, before any winning plugin module is
        # imported or any deferred arbitrary-code loader is registered.
        private_names: set[str] = set()
        invalid_private_declarations: set[str] = set()
        for lookup_key, manifest in winners.items():
            if not _manifest_is_active_for_general_discovery(
                manifest, disabled, enabled
            ):
                continue
            try:
                private_names.update(_manifest_private_secret_names(manifest))
            except PrivateSecretPolicyError:
                invalid_private_declarations.add(lookup_key)
        replace_private_secret_names(private_names)

        for manifest in winners.values():
            lookup_key = manifest.key or manifest.name

            # Explicit disable always wins (matches on key or on legacy
            # bare name for back-compat with existing user configs).
            if lookup_key in disabled or manifest.name in disabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "disabled via config"
                with target._lock:
                    target._plugins[lookup_key] = loaded
                    target._generation += 1
                logger.debug("Skipping disabled plugin '%s'", lookup_key)
                continue

            # A malformed private declaration disables an otherwise active
            # plugin before import. Manifest metadata is attacker-controlled,
            # so neither the declaration nor the underlying exception is
            # reflected into logs.
            if lookup_key in invalid_private_declarations:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "PrivateSecretPolicyError: plugin private-secret "
                    "declaration invalid"
                )
                with target._lock:
                    target._plugins[lookup_key] = loaded
                    target._generation += 1
                logger.warning(
                    "Failed to load plugin '%s': invalid private-secret "
                    "environment declaration",
                    manifest.name,
                )
                continue

            # Exclusive plugins (memory providers) have their own
            # discovery/activation path. The general loader records the
            # manifest for introspection but does not load the module.
            if manifest.kind == "exclusive":
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "exclusive plugin — activate via <category>.provider config"
                )
                with target._lock:
                    target._plugins[lookup_key] = loaded
                    target._generation += 1
                logger.debug(
                    "Skipping '%s' (exclusive, handled by category discovery)",
                    lookup_key,
                )
                continue

            # Model provider plugins are loaded by providers/__init__.py
            # (its own lazy discovery keyed off first get_provider_profile()
            # call). We record the manifest here for introspection but do
            # not import the module — a second import would create two
            # ProviderProfile instances and break the "last writer wins"
            # override semantics between bundled and user plugins.
            if manifest.kind == "model-provider":
                loaded = LoadedPlugin(manifest=manifest, enabled=True)
                with target._lock:
                    target._plugins[lookup_key] = loaded
                    target._generation += 1
                logger.debug(
                    "Skipping '%s' (model-provider, handled by providers/ discovery)",
                    lookup_key,
                )
                continue

            # Built-in backends auto-load — they ship with hermes and must
            # just work. Selection among them (e.g. which image_gen backend
            # services calls) is driven by ``<category>.provider`` config,
            # enforced by the tool wrapper.
            if manifest.source == "bundled" and manifest.kind == "backend":
                self._load_plugin(manifest, transaction=transaction)
                continue

            # Bundled platform plugins (gateway adapters: telegram, discord,
            # feishu, teams, ...) are registered LAZILY. Their modules import
            # heavy, platform-specific SDKs at module level (lark_oapi,
            # microsoft_teams, discord.py, slack_bolt, ...), so eagerly loading
            # all ~20 of them added several seconds to every `hermes`
            # invocation — including plain `hermes chat`, which never touches a
            # gateway platform. Instead we register a cheap deferred loader in
            # the platform_registry keyed on the platform name; the real module
            # is imported only when the gateway / cron / setup / send_message
            # path actually asks for that platform. Every platform Hermes ships
            # remains available out of the box — it just loads on first use.
            if manifest.source == "bundled" and manifest.kind == "platform":
                self._register_deferred_platform(
                    manifest,
                    transaction=transaction,
                )
                continue

            # Everything else (standalone, user-installed backends,
            # entry-point plugins) is opt-in via plugins.enabled.
            # Accept both the path-derived key and the legacy bare name
            # so existing configs keep working.
            is_enabled = (
                enabled is not None
                and (lookup_key in enabled or manifest.name in enabled)
            )
            if not is_enabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "not enabled in config (run `hermes plugins enable {}` to activate)"
                    .format(lookup_key)
                )
                with target._lock:
                    target._plugins[lookup_key] = loaded
                    target._generation += 1
                logger.debug(
                    "Skipping '%s' (not in plugins.enabled)", lookup_key
                )
                continue
            self._load_plugin(manifest, transaction=transaction)

        if manifests:
            with target._lock:
                found_count = len(target._plugins)
                enabled_count = sum(
                    1 for plugin in target._plugins.values() if plugin.enabled
                )
            logger.info(
                "Plugin discovery complete: %d found, %d enabled",
                found_count,
                enabled_count,
            )

    # -----------------------------------------------------------------------
    # Directory scanning
    # -----------------------------------------------------------------------

    def _scan_directory(
        self,
        path: Path,
        source: str,
        skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """Read ``plugin.yaml`` manifests from subdirectories of *path*.

        Supports two layouts, mixed freely:

        * **Flat** — ``<root>/<plugin-name>/plugin.yaml``. Key is
          ``<plugin-name>`` (e.g. ``disk-cleanup``).
        * **Category** — ``<root>/<category>/<plugin-name>/plugin.yaml``,
          where the ``<category>`` directory itself has no ``plugin.yaml``.
          Key is ``<category>/<plugin-name>`` (e.g. ``image_gen/openai``).
          Depth is capped at two segments.

        *skip_names* is an optional allow-list of names to ignore at the
        top level (kept for back-compat; the current call sites no longer
        pass it now that categories are first-class).
        """
        return self._scan_directory_level(
            path, source, skip_names=skip_names, prefix="", depth=0
        )

    def _scan_directory_level(
        self,
        path: Path,
        source: str,
        *,
        skip_names: Optional[Set[str]],
        prefix: str,
        depth: int,
    ) -> List[PluginManifest]:
        """Recursive implementation of :meth:`_scan_directory`.

        ``prefix`` is the category path already accumulated ("" at root,
        "image_gen" one level in). ``depth`` is the recursion depth; we
        cap at 2 so ``<root>/a/b/c/`` is ignored.
        """
        manifests: List[PluginManifest] = []
        if not path.is_dir():
            return manifests

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if depth == 0 and skip_names and child.name in skip_names:
                continue
            manifest_file = child / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = child / "plugin.yml"

            if manifest_file.exists():
                manifest = self._parse_manifest(
                    manifest_file, child, source, prefix
                )
                if manifest is not None:
                    manifests.append(manifest)
                continue

            # No manifest at this level. If we're still within the depth
            # cap, treat this directory as a category namespace and recurse
            # one level in looking for children with manifests.
            if depth >= 1:
                logger.debug("Skipping %s (no plugin.yaml, depth cap reached)", child)
                continue

            sub_prefix = f"{prefix}/{child.name}" if prefix else child.name
            manifests.extend(
                self._scan_directory_level(
                    child,
                    source,
                    skip_names=None,
                    prefix=sub_prefix,
                    depth=depth + 1,
                )
            )

        return manifests

    def _parse_manifest(
        self,
        manifest_file: Path,
        plugin_dir: Path,
        source: str,
        prefix: str,
    ) -> Optional[PluginManifest]:
        """Parse a single ``plugin.yaml`` into a :class:`PluginManifest`.

        Returns ``None`` on parse failure (logs a warning).
        """
        try:
            if yaml is None:
                logger.warning("PyYAML not installed – cannot load %s", manifest_file)
                return None
            data = fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}

            name = data.get("name", plugin_dir.name)
            key = f"{prefix}/{plugin_dir.name}" if prefix else name

            raw_kind = data.get("kind", "standalone")
            if not isinstance(raw_kind, str):
                raw_kind = "standalone"
            kind = raw_kind.strip().lower()
            if kind not in _VALID_PLUGIN_KINDS:
                logger.warning(
                    "Plugin %s: unknown kind '%s' (valid: %s); treating as 'standalone'",
                    key, raw_kind, ", ".join(sorted(_VALID_PLUGIN_KINDS)),
                )
                kind = "standalone"

            # Auto-coerce user-installed memory providers to kind="exclusive"
            # so they're routed to plugins/memory discovery instead of being
            # loaded by the general PluginManager (which has no
            # register_memory_provider on PluginContext). Mirrors the
            # heuristic in plugins/memory/__init__.py:_is_memory_provider_dir.
            # Bundled memory providers are already skipped via skip_names.
            if kind == "standalone" and "kind" not in data:
                init_file = plugin_dir / "__init__.py"
                if init_file.exists():
                    try:
                        source_text = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
                        if (
                            "register_memory_provider" in source_text
                            or "MemoryProvider" in source_text
                        ):
                            kind = "exclusive"
                            logger.debug(
                                "Plugin %s: detected memory provider, "
                                "treating as kind='exclusive'",
                                key,
                            )
                        elif (
                            "register_provider" in source_text
                            and "ProviderProfile" in source_text
                        ):
                            # Model provider plugin (calls register_provider()
                            # from ``providers`` with a ProviderProfile). Route
                            # to providers/__init__.py discovery.
                            kind = "model-provider"
                            logger.debug(
                                "Plugin %s: detected model provider, "
                                "treating as kind='model-provider'",
                                key,
                            )
                    except Exception:
                        pass

            logger.debug(
                "Parsed manifest: key=%s name=%s kind=%s source=%s path=%s",
                key, name, kind, source, plugin_dir,
            )
            return PluginManifest(
                name=name,
                version=str(data.get("version", "")),
                description=data.get("description", ""),
                author=data.get("author", ""),
                requires_env=data.get("requires_env", []),
                provides_tools=data.get("provides_tools", []),
                provides_hooks=data.get("provides_hooks", []),
                source=source,
                path=str(plugin_dir),
                kind=kind,
                key=key,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse %s: %s", manifest_file, exc, exc_info=_PLUGINS_DEBUG,
            )
            return None

    # -----------------------------------------------------------------------
    # Entry-point scanning
    # -----------------------------------------------------------------------

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Check ``importlib.metadata`` for pip-installed plugins."""
        manifests: List[PluginManifest] = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINTS_GROUP)
            elif isinstance(eps, dict):
                group_eps = eps.get(ENTRY_POINTS_GROUP, [])
            else:
                group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

            for ep in group_eps:
                manifest = PluginManifest(
                    name=ep.name,
                    source="entrypoint",
                    path=ep.value,
                    key=ep.name,
                )
                manifests.append(manifest)
        except Exception as exc:
            logger.debug("Entry-point scan failed: %s", exc)

        return manifests

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _platform_name_from_manifest(self, manifest: PluginManifest) -> str:
        """Derive the gateway platform name (e.g. ``feishu``) for a platform plugin.

        The platform name registered via ``register_platform(name=...)`` lives
        inside the adapter module (which we are explicitly trying NOT to import
        early). It is not carried in ``plugin.yaml``. Across every bundled
        platform plugin the manifest name is ``<platform>-platform`` and the
        plugin directory basename is ``<platform>``, so we derive the name
        without importing: strip a trailing ``-platform`` from the manifest
        name, falling back to the directory basename. This is also a sensible
        convention for third-party platform plugins.
        """
        name = manifest.name or ""
        if name.endswith("-platform"):
            return name[: -len("-platform")]
        if manifest.path:
            return Path(manifest.path).name
        return name

    def _register_deferred_platform(
        self,
        manifest: PluginManifest,
        transaction: Optional[_RegistrationTransaction] = None,
    ) -> None:
        """Stage a bundled platform loader with its manager attribution."""
        own_transaction = transaction is None
        transaction = transaction or _RegistrationTransaction(self)
        target = transaction.manager_view
        lookup_key = manifest.key or manifest.name
        platform_name = self._platform_name_from_manifest(manifest)

        loaded = LoadedPlugin(manifest=manifest, enabled=True)
        loaded.deferred = True
        with target._lock:
            target._plugins[lookup_key] = loaded
            target._record_external_registration("platform", platform_name)

        def _loader(_manifest: PluginManifest = manifest) -> None:
            self._load_plugin(_manifest)

        transaction.external_views["platform"].register_deferred(
            platform_name,
            _loader,
        )
        if own_transaction:
            transaction.commit()
        logger.debug(
            "Registered deferred platform loader: %s (plugin=%s)",
            platform_name,
            lookup_key,
        )

    def _load_plugin(
        self,
        manifest: PluginManifest,
        transaction: Optional[_RegistrationTransaction] = None,
    ) -> None:
        """Stage one plugin and publish only through the ordered transaction."""
        with self._discovery_lock:
            self._load_plugin_serialized(manifest, transaction)

    def _load_plugin_serialized(
        self,
        manifest: PluginManifest,
        transaction: Optional[_RegistrationTransaction],
    ) -> None:
        own_transaction = transaction is None
        transaction = transaction or _RegistrationTransaction(self)
        loaded = LoadedPlugin(manifest=manifest)
        plugin_id = manifest.key or manifest.name
        module_root = (
            _directory_module_name(manifest)
            if manifest.source in {"user", "project", "bundled"}
            else None
        )
        manager_before = _PluginManagerState(transaction.manager_view)
        tools_start = len(transaction.manager_view._transaction_tools_registered)
        commands_start = len(
            transaction.manager_view._transaction_commands_registered
        )
        governors_start = len(
            transaction.manager_view._transaction_mcp_governors_registered
        )
        modules_before = (
            _snapshot_module_namespace(module_root) if module_root else {}
        )
        phase = "import"
        context = PluginContext(
            manifest,
            self,
            registration_manager=transaction.manager_view,
            registration_registry=transaction.tool_view,
            registration_external_registries=transaction.context_targets(),
            registration_lock=transaction.registration_lock,
        )
        try:
            transaction.tool_view.register_plugin_override_policy(
                module_root or plugin_id,
                context._tool_override_allowed(""),
            )
            if manifest.source in {"user", "project", "bundled"}:
                module = self._load_directory_module(manifest)
            else:
                module = self._load_entrypoint_module(manifest)

            loaded.module = module
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                raise RuntimeError("plugin has no register function")

            phase = "registration"
            register_fn(context)
            staged = transaction.manager_view
            loaded.tools_registered = list(
                staged._transaction_tools_registered[tools_start:]
            )
            loaded.hooks_registered = [
                name
                for name, callbacks in staged._hooks.items()
                if len(callbacks) > len(manager_before._hooks.get(name, []))
            ]
            loaded.middleware_registered = [
                name
                for name, callbacks in staged._middleware.items()
                if len(callbacks) > len(manager_before._middleware.get(name, []))
            ]
            loaded.commands_registered = list(
                staged._transaction_commands_registered[commands_start:]
            )
            loaded.mcp_governors_registered = list(
                staged._transaction_mcp_governors_registered[governors_start:]
            )
            loaded.enabled = True
            with staged._lock:
                staged._plugins[plugin_id] = loaded
                staged._generation += 1
            transaction.contexts.append(context)
            phase = "commit"
            if own_transaction:
                transaction.commit()
            logger.debug(
                "Plugin '%s' staged %d tool(s), %d hook(s), %d middleware, "
                "%d slash command(s)",
                plugin_id,
                len(loaded.tools_registered),
                len(loaded.hooks_registered),
                len(loaded.middleware_registered),
                len(loaded.commands_registered),
            )
            return
        except BaseException as exc:
            if module_root:
                _restore_module_namespace(module_root, modules_before)
            loaded.enabled = False
            loaded.module = None
            if not isinstance(exc, Exception):
                raise
            loaded.error = (
                "RuntimeError: plugin commit failed"
                if phase == "commit"
                else f"{type(exc).__name__}: plugin {phase} failed"
            )
            if isinstance(exc, RegistryTransactionConflict):
                logger.warning(
                    "Plugin commit conflict: plugin=%s surface=%s "
                    "expected_generation=%d actual_generation=%d",
                    plugin_id,
                    exc.surface,
                    exc.expected_generation,
                    exc.actual_generation,
                )
            else:
                logger.warning(
                    "Failed to load plugin '%s': %s",
                    manifest.name,
                    loaded.error,
                )
            if not own_transaction:
                raise _ForceSweepAbort() from None
            with self._lock:
                self._plugins[plugin_id] = loaded
                self._generation += 1

    def _load_directory_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Import a directory-based plugin as ``hermes_plugins.<slug>``.

        The module slug is derived from ``manifest.key`` so category-namespaced
        plugins (``image_gen/openai``) import as
        ``hermes_plugins.image_gen__openai`` without colliding with any
        future ``tts/openai``.
        """
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")

        # Ensure the namespace parent package exists
        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg

        module_name = _directory_module_name(manifest)
        # A force reload must execute against a clean package tree. The caller
        # already captured prior identities and will restore them on failure.
        _restore_module_namespace(module_name, {})
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")

        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_entrypoint_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference."""
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=ENTRY_POINTS_GROUP)
        elif isinstance(eps, dict):
            group_eps = eps.get(ENTRY_POINTS_GROUP, [])
        else:
            group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

        for ep in group_eps:
            if ep.name == manifest.name:
                return ep.load()

        raise ImportError(
            f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'"
        )

    # -----------------------------------------------------------------------
    # Hook invocation
    # -----------------------------------------------------------------------

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Call all registered callbacks for *hook_name*.

        Each callback is wrapped in its own try/except so a misbehaving
        plugin cannot break the core agent loop.

        Returns a list of non-``None`` return values from callbacks.

        For ``pre_llm_call``, callbacks may return a dict describing
        context to inject into the current turn's user message::

            {"context": "recalled text..."}
            "recalled text..."          # plain string, equivalent

        Context is ALWAYS injected into the user message, never the
        system prompt.  This preserves the prompt cache prefix — the
        system prompt stays identical across turns so cached tokens
        are reused.  All injected context is ephemeral — never
        persisted to session DB.
        """
        kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        with self._lock:
            callbacks = list(self._hooks.get(hook_name, []))
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    hook_name,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        with self._lock:
            return bool(self._hooks.get(hook_name))

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        with self._lock:
            return bool(self._middleware.get(kind))

    def invoke_middleware(self, kind: str, **kwargs: Any) -> List[Any]:
        """Call registered middleware callbacks for *kind*.

        Each callback is isolated so one plugin cannot break the base runtime
        path. Middleware that wants to change behavior must return the shape
        documented by the caller-specific contract.
        """
        with self._lock:
            callbacks = list(self._middleware.get(kind, []))
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Middleware '%s' callback %s raised: %s",
                    kind,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    # -----------------------------------------------------------------------
    # Slack action handler accessor
    # -----------------------------------------------------------------------

    def get_slack_action_handlers(self) -> List[tuple]:
        """Return the list of plugin-registered Slack action handlers.

        Each entry is a ``(action_id, callback, plugin_name)`` tuple.
        Consumed by the Slack adapter at connect time to wire callbacks
        into its ``slack_bolt.AsyncApp``.

        Plugins register handlers via
        :meth:`PluginContext.register_slack_action_handler`.
        """
        with self._lock:
            return list(self._slack_action_handlers)

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        with self._lock:
            plugins = sorted(self._plugins.items())
        result: List[Dict[str, Any]] = []
        for key, loaded in plugins:
            result.append(
                {
                    "name": loaded.manifest.name,
                    "key": loaded.manifest.key or loaded.manifest.name,
                    "kind": loaded.manifest.kind,
                    "version": loaded.manifest.version,
                    "description": loaded.manifest.description,
                    "source": loaded.manifest.source,
                    "enabled": loaded.enabled,
                    "tools": len(loaded.tools_registered),
                    "hooks": len(loaded.hooks_registered),
                    "middleware": len(loaded.middleware_registered),
                    "commands": len(loaded.commands_registered),
                    "error": loaded.error,
                }
            )
        return result

    # -----------------------------------------------------------------------
    # Plugin skill lookups
    # -----------------------------------------------------------------------

    def find_plugin_skill(self, qualified_name: str) -> Optional[Path]:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        with self._lock:
            entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> List[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        with self._lock:
            return sorted(
                e["bare_name"]
                for qn, e in self._plugin_skills.items()
                if qn.startswith(prefix)
            )

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        with self._lock:
            self._plugin_skills.pop(qualified_name, None)

    def get_context_engine(self) -> Any:
        with self._lock:
            return self._context_engine

    def get_plugin_command(self, name: str) -> Optional[dict]:
        with self._lock:
            entry = self._plugin_commands.get(name)
            return dict(entry) if entry else None

    def get_plugin_commands(self) -> Dict[str, dict]:
        with self._lock:
            return {
                name: dict(entry)
                for name, entry in self._plugin_commands.items()
            }

    def get_cli_commands(self) -> Dict[str, dict]:
        with self._lock:
            return {
                name: dict(entry) for name, entry in self._cli_commands.items()
            }

    def get_auxiliary_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    **self._aux_tasks[key],
                    "defaults": dict(
                        self._aux_tasks[key].get("defaults", {})
                    ),
                }
                for key in sorted(self._aux_tasks)
            ]

    def get_middleware_callbacks(self, kind: str) -> List[Callable]:
        with self._lock:
            return list(self._middleware.get(kind, []))

    def get_tool_state_snapshot(
        self,
    ) -> tuple[Set[str], Dict[str, LoadedPlugin]]:
        with self._lock:
            return set(self._plugin_tool_names), dict(self._plugins)

    def get_mcp_governor(self, exact_server_name: str) -> MCPGovernorRegistration | None:
        """Return one immutable exact-server registration under the state lock."""

        with self._lock:
            return self._mcp_governors.get(exact_server_name)

    def get_mcp_governor_attribution(
        self, exact_server_name: str
    ) -> str | None:
        """Return the owning plugin id without exposing mutable manager state."""

        with self._lock:
            registration = self._mcp_governors.get(exact_server_name)
            return registration.plugin_id if registration is not None else None


# ---------------------------------------------------------------------------
# Module-level singleton & convenience functions
# ---------------------------------------------------------------------------

_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Return (and lazily create) the global PluginManager singleton."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def discover_plugins(force: bool = False) -> None:
    """Discover and load all plugins.

    Default behavior is idempotent. Pass ``force=True`` to rescan plugin
    manifests and reload state in the current process.
    """
    get_plugin_manager().discover_and_load(force=force)


def get_mcp_governor(
    exact_server_name: str,
) -> MCPGovernorRegistration | None:
    """Read the active exact-server governor from the process manager."""

    manager = _plugin_manager
    if manager is None:
        return None
    return manager.get_mcp_governor(exact_server_name)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook on loaded plugins.

    Returns a list of non-``None`` return values from plugin callbacks.
    """
    return get_plugin_manager().invoke_hook(hook_name, **kwargs)


def invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    """Invoke registered middleware callbacks.

    Returns a list of non-``None`` return values from middleware callbacks.
    """
    return get_plugin_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """Return True when middleware callbacks are registered for ``kind``."""
    manager = get_plugin_manager()
    method = getattr(manager, "has_middleware", None)
    if callable(method):
        return bool(method(kind))
    return bool(getattr(manager, "_middleware", {}).get(kind))


def has_hook(hook_name: str) -> bool:
    """Return True when a loaded plugin handles a hook."""
    return get_plugin_manager().has_hook(hook_name)


_thread_tool_whitelist = threading.local()


@dataclass(frozen=True)
class _PreToolCallDirective:
    action: Optional[str] = None
    message: Optional[str] = None
    rule_key: Optional[str] = None


def set_thread_tool_whitelist(
    allowed: Optional[Set[str]],
    deny_msg_fmt: str = "Tool '{tool_name}' denied: not in this thread's tool whitelist",
) -> None:
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt


def clear_thread_tool_whitelist() -> None:
    _thread_tool_whitelist.allowed = None


def _get_pre_tool_call_directive_details(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> _PreToolCallDirective:
    """Check ``pre_tool_call`` hooks for a blocking or approval directive.

    Plugins that need to enforce policy (rate limiting, security
    restrictions, approval workflows) can return one of::

        {"action": "block",   "message": "Reason the tool was blocked"}
        {"action": "approve", "message": "Why this needs human confirmation"}
        {"action": "approve", "message": "...", "rule_key": "write_file:ssh"}

    from their ``pre_tool_call`` callback.

    - ``block`` vetoes the tool call outright (the message becomes the tool
      result the model sees).
    - ``approve`` ESCALATES to the existing human-approval gate
      (``prompt_dangerous_approval`` on CLI, the approval callback on the
      gateway) — the same mechanism Tier-2 dangerous shell patterns use.
      This lets a plugin require a human ``[o]nce/[s]ession/[a]lways/[d]eny``
      decision on ANY tool, not just terminal command strings. The caller is
      responsible for invoking the gate (see
      :func:`tools.approval.request_tool_approval`).
    - ``rule_key`` is optional and only honored for ``approve`` directives. It
      lets plugins choose the allowlist grain for `[a]lways` approvals.

    The first valid directive wins. Invalid or irrelevant hook return values
    are silently ignored so existing observer-only hooks are unaffected.
    """
    allowed = getattr(_thread_tool_whitelist, "allowed", None)
    if allowed is not None and tool_name not in allowed:
        fmt = getattr(_thread_tool_whitelist, "fmt", "Tool '{tool_name}' denied")
        return _PreToolCallDirective(
            action="block",
            message=fmt.format(tool_name=tool_name),
        )

    hook_results = invoke_hook(
        "pre_tool_call",
        tool_name=tool_name,
        args=args if isinstance(args, dict) else {},
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        middleware_trace=list(middleware_trace or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        if action not in ("block", "approve"):
            continue
        message = result.get("message")
        message = message if isinstance(message, str) and message else None
        # A block directive requires a message (it becomes the tool result);
        # an approve directive can carry an optional reason.
        if action == "block" and not message:
            continue
        rule_key = result.get("rule_key") if action == "approve" else None
        rule_key = rule_key.strip() if isinstance(rule_key, str) else None
        if not rule_key:
            rule_key = None
        return _PreToolCallDirective(action=action, message=message, rule_key=rule_key)

    return _PreToolCallDirective()


def get_pre_tool_call_directive(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Check ``pre_tool_call`` hooks for a blocking or approval directive.

    Backward-compatible public helper: returns ``(directive, message)`` where
    ``directive`` is ``"block"``, ``"approve"``, or ``None``. Internal callers
    that need approve-specific metadata use
    :func:`_get_pre_tool_call_directive_details`.
    """
    details = _get_pre_tool_call_directive_details(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    return (details.action, details.message)


def get_pre_tool_call_block_message(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Back-compat shim: return only a ``block`` message (or ``None``).

    Deprecated in favor of :func:`get_pre_tool_call_directive`, which also
    surfaces the ``approve`` escalation directive. Kept so any external caller
    importing the old name keeps working; ``approve`` directives are invisible
    to this shim (it only reports blocks).
    """
    directive, message = get_pre_tool_call_directive(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    return message if directive == "block" else None


def resolve_pre_tool_block(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Resolve the pre_tool_call directive to a final block message (or None).

    Single entry point for every tool-dispatch site: fetches the plugin
    directive and, for an ``approve`` escalation, invokes the human-approval
    gate (:func:`tools.approval.request_tool_approval`). Returns the message
    the tool result should carry when the call is blocked, or ``None`` when
    the call may proceed.

    Centralizing this keeps the security-critical fail-closed logic in ONE
    place instead of copy-pasted across the concurrent/sequential/helper
    dispatch paths: an ``approve`` directive whose gate errors, denies, or
    times out is fail-closed to a block; ``block`` blocks with its message;
    anything else proceeds.
    """
    details = _get_pre_tool_call_directive_details(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    if details.action == "block":
        return details.message
    if details.action == "approve":
        try:
            from tools.approval import request_tool_approval
            result = request_tool_approval(
                tool_name,
                details.message or "",
                rule_key=details.rule_key or tool_name,
            )
        except Exception:
            # Fail-closed: if the gate itself errors, block rather than
            # silently execute an action a plugin flagged for approval.
            return f"BLOCKED: plugin approval gate failed for {tool_name}"
        if not result.get("approved"):
            return str(
                result.get("message")
                or f"BLOCKED: plugin approval required for {tool_name}"
            )
    return None


def get_pre_verify_continue_message(
    *,
    session_id: str = "",
    platform: str = "",
    model: str = "",
    coding: bool = False,
    attempt: int = 0,
    final_response: str = "",
    changed_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Check user ``pre_verify`` hooks for a directive to keep the agent going.

    Fired once per turn when the agent edited code and is about to verify/finish.
    A hook keeps the turn going (run a check, defer it, tidy the diff) by
    returning::

        {"action": "continue", "message": "<follow-up for the model>"}

    The Claude-Code Stop shape ``{"decision": "block", "reason": "..."}`` (block
    the stop == keep going) is accepted too. The first directive carrying a
    non-empty message wins; any other return lets the turn finish. Mirrors
    :func:`get_pre_tool_call_block_message` — the call site stays a one-liner.

    ``coding`` / ``attempt`` let a hook scope itself (``if not coding`` …) and
    self-throttle (``if attempt`` …), the same way a ``pre_tool_call`` hook
    scopes on ``tool_name``.
    """
    hook_results = invoke_hook(
        "pre_verify",
        session_id=session_id,
        platform=platform,
        model=model,
        coding=coding,
        attempt=attempt,
        final_response=final_response,
        changed_paths=list(changed_paths or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        if action not in ("continue", "block"):
            continue
        message = result.get("message") or result.get("reason")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return None


def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the global manager after ensuring plugin discovery has run.

    Pass ``force=True`` to rescan in the current process.
    """
    manager = get_plugin_manager()
    manager.discover_and_load(force=force)
    return manager


def get_plugin_context_engine():
    """Return the plugin-registered context engine, or None."""
    return _ensure_plugins_discovered().get_context_engine()


def get_plugin_command_handler(name: str) -> Optional[Callable]:
    """Return the handler for a plugin-registered slash command, or ``None``."""
    entry = _ensure_plugins_discovered().get_plugin_command(name)
    return entry["handler"] if entry else None


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command return value, awaiting async handlers when needed.

    Sync CLI/TUI dispatch sites call plugin handlers from plain functions.
    If a handler is async, await it directly when no loop is running; if
    we're already inside an active loop, run it in a helper thread with its
    own loop so the caller still gets a concrete result synchronously. The
    threaded path is bounded by a 30s timeout so a hung async handler cannot
    wedge the terminal indefinitely.
    """
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(
        target=_runner,
        name="hermes-plugin-command-await",
        daemon=True,
    )
    thread.start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError(
            "Plugin command async handler did not complete within "
            f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s"
        )
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_plugin_commands() -> Dict[str, dict]:
    """Return the full plugin commands dict (name → {handler, description, plugin}).

    Triggers idempotent plugin discovery so callers can use plugin commands
    before any explicit discover_plugins() call.
    """
    return _ensure_plugins_discovered().get_plugin_commands()


def get_plugin_auxiliary_tasks() -> List[Dict[str, Any]]:
    """Return all plugin-registered auxiliary tasks as a stable-ordered list.

    Each entry is the registration dict from
    :meth:`PluginContext.register_auxiliary_task`:
    ``{key, display_name, description, defaults, plugin}``.

    Triggers idempotent plugin discovery so callers can read the registry
    before any explicit ``discover_plugins()`` call. Sorted by ``key`` for
    deterministic ordering in pickers and tests.
    """
    return _ensure_plugins_discovered().get_auxiliary_tasks()


def get_plugin_toolsets() -> List[tuple]:
    """Return plugin toolsets as ``(key, label, description)`` tuples.

    Used by the ``hermes tools`` TUI so plugin-provided toolsets appear
    alongside the built-in ones and can be toggled on/off per platform.
    """
    manager = get_plugin_manager()
    plugin_tool_names, loaded_plugins = manager.get_tool_state_snapshot()
    if not plugin_tool_names:
        return []

    try:
        from tools.registry import registry
    except Exception:
        return []

    # Group plugin tool names by their toolset
    toolset_tools: Dict[str, List[str]] = {}
    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for tool_name in plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if not entry:
            continue
        ts = entry.toolset
        toolset_tools.setdefault(ts, []).append(entry.name)

    # Map toolsets back to the plugin that registered them
    for _name, loaded in loaded_plugins.items():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        label = f"🔌 {ts_key.replace('_', ' ').title()}"
        if plugin and plugin.manifest.description:
            desc = plugin.manifest.description
        else:
            desc = ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, label, desc))

    return result
