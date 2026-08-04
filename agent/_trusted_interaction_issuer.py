"""Host-private capability for authenticating accepted interactions.

This module is intentionally underscored and excluded from the public package
API.  Python plugins run as trusted code, but the boundary is explicit: public
consumers can only read; host ingress code alone imports these issuer helpers.
"""

from __future__ import annotations

import base64
from contextvars import Token
from dataclasses import fields
import hmac
import json
import time
import uuid

from agent.trusted_interaction import (
    TrustedInteraction,
    _host_bind_trusted_interaction,
    _host_clear_trusted_interactions,
    _host_drop_trusted_interaction_context,
    _host_issue_trusted_interaction,
    _host_publish_trusted_interaction,
    _host_reset_trusted_interaction,
)


_SERIALIZED_FIELDS = tuple(field.name for field in fields(TrustedInteraction))
_UNSIGNED_FIELD_SET = frozenset(_SERIALIZED_FIELDS)
_SIGNED_FIELD_SET = _UNSIGNED_FIELD_SET | {"ipc_mac"}
_IPC_MAC_KEY_BYTES = 32


def _valid_ipc_mac_key(mac_key: object) -> bytes | None:
    if not isinstance(mac_key, bytes) or len(mac_key) != _IPC_MAC_KEY_BYTES:
        return None
    return mac_key


def _canonical_ipc_payload(payload: dict[str, object]) -> bytes | None:
    try:
        return json.dumps(
            {name: payload[name] for name in _SERIALIZED_FIELDS},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (KeyError, TypeError, ValueError):
        return None


def _restore_unsigned_payload(
    payload: dict[str, object],
) -> TrustedInteraction | None:
    if frozenset(payload) != _UNSIGNED_FIELD_SET:
        return None
    expected_hash = payload["evidence_hash"]
    try:
        restored = _host_issue_trusted_interaction(
            **{
                name: payload[name]
                for name in _SERIALIZED_FIELDS
                if name != "evidence_hash"
            }
        )
    except (TypeError, ValueError):
        return None
    if not isinstance(expected_hash, str) or restored.evidence_hash != expected_hash:
        return None
    return restored


def _ipc_mac(payload: dict[str, object], mac_key: bytes) -> str | None:
    canonical = _canonical_ipc_payload(payload)
    if canonical is None:
        return None
    return base64.urlsafe_b64encode(
        hmac.digest(mac_key, canonical, "sha256")
    ).decode("ascii")


def _issue_trusted_interaction(
    *,
    surface: str,
    platform: str,
    actor_id: str,
    channel_id: str,
    chat_id: str,
    thread_id: str | None,
    session_key: str,
    session_id: str,
    message_id: str,
    received_at: float | None = None,
    profile_id: str,
    interaction_id: str | None = None,
) -> TrustedInteraction:
    return _host_issue_trusted_interaction(
        interaction_id=interaction_id or f"ti_{uuid.uuid4().hex}",
        surface=surface,
        platform=platform,
        actor_id=actor_id,
        channel_id=channel_id,
        chat_id=chat_id,
        thread_id=thread_id,
        session_key=session_key,
        session_id=session_id,
        message_id=message_id,
        received_at=time.time() if received_at is None else received_at,
        profile_id=profile_id,
    )


def _publish_trusted_interaction(
    interaction: TrustedInteraction,
    *,
    ttl_seconds: float = 15 * 60.0,
    max_entries: int = 4096,
) -> None:
    _host_publish_trusted_interaction(
        interaction,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
    )


def _bind_trusted_interaction(
    interaction: TrustedInteraction,
) -> Token[TrustedInteraction | None]:
    return _host_bind_trusted_interaction(interaction)


def _reset_trusted_interaction(
    token: Token[TrustedInteraction | None],
) -> None:
    _host_reset_trusted_interaction(token)


def _drop_trusted_interaction_context() -> None:
    _host_drop_trusted_interaction_context()


def _clear_trusted_interactions(
    *,
    session_key: str | None = None,
    interaction: TrustedInteraction | None = None,
) -> None:
    _host_clear_trusted_interactions(
        session_key=session_key,
        interaction=interaction,
    )


def _serialize_trusted_interaction(
    interaction: TrustedInteraction | None,
) -> dict[str, object] | None:
    if interaction is None:
        return None
    return {
        name: getattr(interaction, name)
        for name in _SERIALIZED_FIELDS
    }


def _attach_trusted_interaction_ipc_mac(
    payload: object,
    *,
    mac_key: bytes,
) -> dict[str, object] | None:
    """Copy and authenticate one exact serialized interaction for private IPC."""
    key = _valid_ipc_mac_key(mac_key)
    if key is None or not isinstance(payload, dict):
        return None
    copied = dict(payload)
    if frozenset(copied) == _SIGNED_FIELD_SET:
        copied.pop("ipc_mac")
    if frozenset(copied) != _UNSIGNED_FIELD_SET:
        return None
    if _restore_unsigned_payload(copied) is None:
        return None
    ipc_mac = _ipc_mac(copied, key)
    if ipc_mac is None:
        return None
    copied["ipc_mac"] = ipc_mac
    return copied


def _restore_trusted_interaction(
    payload: object,
    *,
    mac_key: bytes | None,
) -> TrustedInteraction | None:
    """Restore only evidence authenticated for this exact compute-host spawn."""
    key = _valid_ipc_mac_key(mac_key)
    if key is None or not isinstance(payload, dict):
        return None
    copied = dict(payload)
    if frozenset(copied) != _SIGNED_FIELD_SET:
        return None
    received_mac = copied.pop("ipc_mac")
    if not isinstance(received_mac, str):
        return None
    expected_mac = _ipc_mac(copied, key)
    try:
        mac_matches = expected_mac is not None and hmac.compare_digest(
            received_mac.encode("ascii"),
            expected_mac.encode("ascii"),
        )
    except UnicodeEncodeError:
        return None
    if not mac_matches:
        return None
    return _restore_unsigned_payload(copied)
