"""Read-only provenance for the exact host-accepted user interaction.

The public surface in this module is intentionally consumer-only.  Plugins and
tools may inspect :class:`TrustedInteraction` through
:func:`get_current_trusted_interaction`, but cannot construct, bind, or publish
an authenticated value.  Host ingress code owns those operations through the
underscored capability module ``agent._trusted_interaction_issuer``.

The latest accepted value is stored per exact conversation scope.  A bound
interaction is only an anchor for that scope; every read re-resolves the
bounded, expiring ledger.  This matters for busy/OOB input: the already-running
agent task retains its original ContextVar snapshot, while a newly accepted
message can safely advance only that session's ledger entry.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
import threading
import time
from typing import Iterator


_MAX_IDENTIFIER_CHARS = 2048
_DEFAULT_LEDGER_TTL_SECONDS = 15 * 60.0
_DEFAULT_LEDGER_MAX_ENTRIES = 4096
_SECRET_MARKERS = (
    "api_key=",
    "apikey=",
    "authorization:",
    "bearer ",
    "password=",
    "secret=",
    "token=",
)


@dataclass(frozen=True, slots=True, init=False)
class TrustedInteraction:
    """Immutable evidence for one interaction authenticated by the host.

    The contract deliberately contains identifiers and a timestamp only.  It
    has no text, tool arguments, credentials, headers, or arbitrary metadata
    field where a secret could be smuggled into provenance.

    ``init=False`` plus the rejecting initializer makes the API boundary
    explicit: importing this type does not grant an authentication capability.
    Host ingress creates instances through the private issuer module.
    """

    interaction_id: str
    surface: str
    platform: str
    actor_id: str
    channel_id: str
    chat_id: str
    thread_id: str | None
    session_key: str
    session_id: str
    message_id: str
    received_at: float
    profile_id: str
    evidence_hash: str

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise TypeError(
            "TrustedInteraction is host-issued; use the read-only getter API"
        )


_CONTRACT_FIELDS = (
    "interaction_id",
    "surface",
    "platform",
    "actor_id",
    "channel_id",
    "chat_id",
    "thread_id",
    "session_key",
    "session_id",
    "message_id",
    "received_at",
    "profile_id",
)


@dataclass(frozen=True, slots=True)
class _LedgerEntry:
    interaction: TrustedInteraction
    expires_at: float


_ScopeKey = tuple[
    str,
    str,
    str,
    str,
    str,
    str | None,
    str,
    str,
    str,
]


_CURRENT_SCOPE_ANCHOR: ContextVar[TrustedInteraction | None] = ContextVar(
    "hermes_trusted_interaction_scope_anchor",
    default=None,
)
_LATEST_BY_SCOPE: OrderedDict[_ScopeKey, _LedgerEntry] = OrderedDict()
_LEDGER_LOCK = threading.RLock()


def _validate_identifier(
    name: str,
    value: object,
    *,
    optional: bool = False,
) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > _MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{name} is too long")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} contains control characters")
    lowered = value.casefold()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        raise ValueError(f"{name} must not contain secret material")
    return value


def _canonical_payload(values: dict[str, object]) -> bytes:
    return json.dumps(
        {name: values[name] for name in _CONTRACT_FIELDS},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _host_issue_trusted_interaction(**values: object) -> TrustedInteraction:
    """Construct a validated instance for the host-private issuer only."""
    normalized: dict[str, object] = {}
    for name in _CONTRACT_FIELDS:
        if name == "received_at":
            raw_timestamp = values.get(name)
            if isinstance(raw_timestamp, bool) or not isinstance(
                raw_timestamp, (int, float)
            ):
                raise ValueError("received_at must be a finite timestamp")
            timestamp = float(raw_timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("received_at must be a finite timestamp")
            normalized[name] = timestamp
        else:
            normalized[name] = _validate_identifier(
                name,
                values.get(name),
                optional=name == "thread_id",
            )

    interaction = object.__new__(TrustedInteraction)
    for name, value in normalized.items():
        object.__setattr__(interaction, name, value)
    object.__setattr__(
        interaction,
        "evidence_hash",
        hashlib.sha256(_canonical_payload(normalized)).hexdigest(),
    )
    return interaction


def _scope_key(interaction: TrustedInteraction) -> _ScopeKey:
    return (
        interaction.surface,
        interaction.platform,
        interaction.profile_id,
        interaction.channel_id,
        interaction.chat_id,
        interaction.thread_id,
        interaction.session_key,
        interaction.session_id,
        interaction.actor_id,
    )


def _purge_expired_locked(now: float) -> None:
    expired = [
        key for key, entry in _LATEST_BY_SCOPE.items() if entry.expires_at <= now
    ]
    for key in expired:
        _LATEST_BY_SCOPE.pop(key, None)


def _host_publish_trusted_interaction(
    interaction: TrustedInteraction,
    *,
    ttl_seconds: float = _DEFAULT_LEDGER_TTL_SECONDS,
    max_entries: int = _DEFAULT_LEDGER_MAX_ENTRIES,
) -> None:
    if not isinstance(interaction, TrustedInteraction):
        raise TypeError("interaction must be host-issued TrustedInteraction")
    try:
        ttl = float(ttl_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("ttl_seconds must be positive and finite") from exc
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("ttl_seconds must be positive and finite")
    if (
        isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or max_entries <= 0
    ):
        raise ValueError("max_entries must be a positive integer")

    now = time.monotonic()
    key = _scope_key(interaction)
    with _LEDGER_LOCK:
        _purge_expired_locked(now)
        _LATEST_BY_SCOPE[key] = _LedgerEntry(
            interaction=interaction,
            expires_at=now + ttl,
        )
        _LATEST_BY_SCOPE.move_to_end(key)
        while len(_LATEST_BY_SCOPE) > max_entries:
            _LATEST_BY_SCOPE.popitem(last=False)


def _host_bind_trusted_interaction(
    interaction: TrustedInteraction,
) -> Token[TrustedInteraction | None]:
    if not isinstance(interaction, TrustedInteraction):
        raise TypeError("interaction must be host-issued TrustedInteraction")
    return _CURRENT_SCOPE_ANCHOR.set(interaction)


def _host_reset_trusted_interaction(
    token: Token[TrustedInteraction | None],
) -> None:
    _CURRENT_SCOPE_ANCHOR.reset(token)


def _host_drop_trusted_interaction_context() -> None:
    """Fail closed at a new ingress boundary inherited via ``copy_context``."""
    _CURRENT_SCOPE_ANCHOR.set(None)


def _host_clear_trusted_interactions(
    *,
    session_key: str | None = None,
    interaction: TrustedInteraction | None = None,
) -> None:
    """Clear all evidence, one session, or one exact current entry."""
    with _LEDGER_LOCK:
        if interaction is not None:
            key = _scope_key(interaction)
            entry = _LATEST_BY_SCOPE.get(key)
            if entry is not None and entry.interaction is interaction:
                _LATEST_BY_SCOPE.pop(key, None)
            return
        if session_key is None:
            _LATEST_BY_SCOPE.clear()
            return
        for key in [key for key in _LATEST_BY_SCOPE if key[6] == session_key]:
            _LATEST_BY_SCOPE.pop(key, None)


def _anchor_matches_bound_session(anchor: TrustedInteraction) -> bool:
    """Fail closed when a copied ContextVar is replayed under another session."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return False

    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    source = get_session_env("HERMES_SESSION_SOURCE", "")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "")
    actor_id = get_session_env("HERMES_SESSION_USER_ID", "")
    session_key = get_session_env("HERMES_SESSION_KEY", "")
    session_id = get_session_env("HERMES_SESSION_ID", "")
    ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "")
    profile_id = get_session_env("HERMES_SESSION_PROFILE", "")

    normalized_thread_id = anchor.thread_id or ""

    # Local surfaces use the server-assigned runtime/UI id for session_id while
    # HERMES_SESSION_ID remains the durable conversation id.
    if anchor.platform == "local":
        return (
            platform == "local"
            and source == anchor.surface
            and chat_id == anchor.chat_id
            and thread_id == normalized_thread_id
            and actor_id == anchor.actor_id
            and session_key == anchor.session_key
            and bool(session_id)
            and ui_session_id == anchor.chat_id
            and profile_id == anchor.profile_id
        )

    return (
        platform == anchor.platform
        and source == anchor.surface
        and chat_id == anchor.chat_id
        and thread_id == normalized_thread_id
        and actor_id == anchor.actor_id
        and session_key == anchor.session_key
        and session_id == anchor.session_id
        and profile_id == anchor.profile_id
        and (not ui_session_id or ui_session_id == anchor.chat_id)
    )


def get_current_trusted_interaction() -> TrustedInteraction | None:
    """Return the latest host-authenticated interaction for this exact session.

    There is deliberately no global fallback.  A caller without a host-bound
    scope, with a mismatched copied context, or after TTL/reset receives
    ``None`` so a future governed consumer can fail closed.
    """
    anchor = _CURRENT_SCOPE_ANCHOR.get()
    if anchor is None or not _anchor_matches_bound_session(anchor):
        return None

    now = time.monotonic()
    key = _scope_key(anchor)
    with _LEDGER_LOCK:
        _purge_expired_locked(now)
        entry = _LATEST_BY_SCOPE.get(key)
        if entry is None:
            return None
        _LATEST_BY_SCOPE.move_to_end(key)
        return entry.interaction


@contextmanager
def trusted_interaction_context() -> Iterator[TrustedInteraction | None]:
    """Yield a read-only snapshot of the current trusted interaction."""
    yield get_current_trusted_interaction()


__all__ = [
    "TrustedInteraction",
    "get_current_trusted_interaction",
    "trusted_interaction_context",
]
