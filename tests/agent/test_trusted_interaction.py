from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time

import pytest

from agent.trusted_interaction import (
    TrustedInteraction,
    get_current_trusted_interaction,
    trusted_interaction_context,
)
from agent._trusted_interaction_issuer import (
    _attach_trusted_interaction_ipc_mac,
    _bind_trusted_interaction,
    _clear_trusted_interactions,
    _issue_trusted_interaction,
    _publish_trusted_interaction,
    _reset_trusted_interaction,
    _restore_trusted_interaction,
    _serialize_trusted_interaction,
)
from gateway.session_context import clear_session_vars, set_session_vars


def _issue(**overrides) -> TrustedInteraction:
    values = {
        "surface": "gateway",
        "platform": "telegram",
        "actor_id": "user-7",
        "channel_id": "chat-12",
        "chat_id": "chat-12",
        "thread_id": "topic-4",
        "session_key": "agent:main:telegram:group:chat-12:topic-4",
        "session_id": "session-db-9",
        "message_id": "message-31",
        "received_at": 1_754_000_000.25,
        "profile_id": "default",
    }
    values.update(overrides)
    return _issue_trusted_interaction(**values)


@pytest.fixture(autouse=True)
def _clear_ledger():
    _clear_trusted_interactions()
    yield
    _clear_trusted_interactions()


def test_public_constructor_cannot_authenticate_plugin_values():
    with pytest.raises(TypeError, match="host-issued"):
        TrustedInteraction(
            interaction_id="forged",
            surface="gateway",
            platform="telegram",
            actor_id="attacker",
            channel_id="victim-chat",
            chat_id="victim-chat",
            thread_id=None,
            session_key="victim-session-key",
            session_id="victim-session-id",
            message_id="forged-message",
            received_at=time.time(),
            profile_id="default",
        )


def test_host_issued_value_is_frozen_and_hashes_every_contract_field():
    interaction = _issue()

    with pytest.raises(dataclasses.FrozenInstanceError):
        interaction.message_id = "replacement"  # type: ignore[misc]

    canonical = {
        "actor_id": interaction.actor_id,
        "channel_id": interaction.channel_id,
        "chat_id": interaction.chat_id,
        "interaction_id": interaction.interaction_id,
        "message_id": interaction.message_id,
        "platform": interaction.platform,
        "profile_id": interaction.profile_id,
        "received_at": interaction.received_at,
        "session_id": interaction.session_id,
        "session_key": interaction.session_key,
        "surface": interaction.surface,
        "thread_id": interaction.thread_id,
    }
    expected = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert interaction.evidence_hash == expected
    assert len(interaction.evidence_hash) == 64

    changed = _issue(message_id="message-32")
    assert changed.evidence_hash != interaction.evidence_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("surface", ""),
        ("platform", " "),
        ("actor_id", ""),
        ("channel_id", ""),
        ("chat_id", ""),
        ("session_key", ""),
        ("session_id", ""),
        ("message_id", ""),
        ("profile_id", ""),
        ("received_at", math.nan),
        ("received_at", math.inf),
    ],
)
def test_issuer_rejects_empty_required_fields_and_non_finite_time(field, value):
    with pytest.raises(ValueError):
        _issue(**{field: value})


@pytest.mark.parametrize(
    "secret_value",
    [
        "token=should-not-be-provenance",
        "Authorization: Bearer should-not-be-provenance",
        "password=should-not-be-provenance",
    ],
)
def test_issuer_rejects_secret_shaped_identifier_values(secret_value):
    with pytest.raises(ValueError, match="secret material"):
        _issue(actor_id=secret_value)


def test_public_getter_uses_exact_bound_session_and_never_global_last_writer():
    first = _issue()
    second = _issue(
        actor_id="user-8",
        channel_id="chat-99",
        chat_id="chat-99",
        thread_id="topic-2",
        session_key="agent:main:telegram:group:chat-99:topic-2",
        session_id="session-db-10",
        message_id="message-88",
    )
    _publish_trusted_interaction(first)
    _publish_trusted_interaction(second)

    first_token = _bind_trusted_interaction(first)
    first_session_tokens = set_session_vars(
        platform="telegram",
        source=first.surface,
        chat_id=first.chat_id,
        thread_id=first.thread_id or "",
        user_id=first.actor_id,
        session_key=first.session_key,
        session_id=first.session_id,
        message_id=first.message_id,
        profile=first.profile_id,
    )
    try:
        assert get_current_trusted_interaction() is first
        with trusted_interaction_context() as current:
            assert current is first
    finally:
        clear_session_vars(first_session_tokens)
        _reset_trusted_interaction(first_token)

    second_token = _bind_trusted_interaction(second)
    second_session_tokens = set_session_vars(
        platform="telegram",
        source=second.surface,
        chat_id=second.chat_id,
        thread_id=second.thread_id or "",
        user_id=second.actor_id,
        session_key=second.session_key,
        session_id=second.session_id,
        message_id=second.message_id,
        profile=second.profile_id,
    )
    try:
        assert get_current_trusted_interaction() is second
    finally:
        clear_session_vars(second_session_tokens)
        _reset_trusted_interaction(second_token)


def test_busy_publication_replaces_message_without_crossing_thread_scope():
    original = _issue(message_id="message-original")
    busy = _issue(message_id="message-busy")
    wrong_thread = _issue(
        thread_id="topic-other",
        session_key="agent:main:telegram:group:chat-12:topic-other",
        session_id="session-db-other",
        message_id="message-other",
    )
    _publish_trusted_interaction(original)

    token = _bind_trusted_interaction(original)
    session_tokens = set_session_vars(
        platform="telegram",
        source=original.surface,
        chat_id=original.chat_id,
        thread_id=original.thread_id or "",
        user_id=original.actor_id,
        session_key=original.session_key,
        session_id=original.session_id,
        message_id=original.message_id,
        profile=original.profile_id,
    )
    try:
        _publish_trusted_interaction(wrong_thread)
        assert get_current_trusted_interaction() is original

        _publish_trusted_interaction(busy)
        assert get_current_trusted_interaction() is busy
        assert get_current_trusted_interaction().message_id == "message-busy"
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)


def test_same_telegram_group_topic_isolated_by_actor_and_replaces_per_actor():
    actor_a = _issue(actor_id="actor-a", message_id="message-a-1")
    actor_b = _issue(actor_id="actor-b", message_id="message-b-1")
    actor_a_later = _issue(actor_id="actor-a", message_id="message-a-2")
    _publish_trusted_interaction(actor_a)
    _publish_trusted_interaction(actor_b)

    def current_for(interaction: TrustedInteraction):
        token = _bind_trusted_interaction(interaction)
        session_tokens = set_session_vars(
            platform=interaction.platform,
            source=interaction.surface,
            chat_id=interaction.chat_id,
            thread_id=interaction.thread_id or "",
            user_id=interaction.actor_id,
            session_key=interaction.session_key,
            session_id=interaction.session_id,
            profile=interaction.profile_id,
        )
        try:
            return get_current_trusted_interaction()
        finally:
            clear_session_vars(session_tokens)
            _reset_trusted_interaction(token)

    assert current_for(actor_a) is actor_a
    assert current_for(actor_b) is actor_b

    _publish_trusted_interaction(actor_a_later)

    assert current_for(actor_a) is actor_a_later
    assert current_for(actor_b) is actor_b

    _clear_trusted_interactions(session_key=actor_a.session_key)
    assert current_for(actor_a) is None
    assert current_for(actor_b) is None


@pytest.mark.parametrize(
    ("coordinate", "replacement"),
    [
        ("platform", ""),
        ("platform", "discord"),
        ("source", ""),
        ("source", "gateway-other"),
        ("chat_id", ""),
        ("chat_id", "chat-other"),
        ("thread_id", ""),
        ("thread_id", "topic-other"),
        ("user_id", ""),
        ("user_id", "actor-other"),
        ("session_key", ""),
        ("session_key", "session-key-other"),
        ("session_id", ""),
        ("session_id", "session-id-other"),
        ("profile", ""),
        ("profile", "profile-other"),
    ],
)
def test_nonlocal_anchor_rejects_cleared_or_mismatched_core_coordinate(
    coordinate, replacement
):
    interaction = _issue()
    _publish_trusted_interaction(interaction)
    token = _bind_trusted_interaction(interaction)
    values = {
        "platform": interaction.platform,
        "source": interaction.surface,
        "chat_id": interaction.chat_id,
        "thread_id": interaction.thread_id or "",
        "user_id": interaction.actor_id,
        "session_key": interaction.session_key,
        "session_id": interaction.session_id,
        "profile": interaction.profile_id,
    }
    values[coordinate] = replacement
    session_tokens = set_session_vars(**values)
    try:
        assert get_current_trusted_interaction() is None
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)


def test_nonlocal_anchor_accepts_exact_core_coordinates():
    interaction = _issue()
    _publish_trusted_interaction(interaction)
    token = _bind_trusted_interaction(interaction)
    session_tokens = set_session_vars(
        platform=interaction.platform,
        source=interaction.surface,
        chat_id=interaction.chat_id,
        thread_id=interaction.thread_id or "",
        user_id=interaction.actor_id,
        session_key=interaction.session_key,
        session_id=interaction.session_id,
        profile=interaction.profile_id,
    )
    try:
        assert get_current_trusted_interaction() is interaction
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)


@pytest.mark.parametrize(
    ("coordinate", "replacement"),
    [
        ("platform", ""),
        ("platform", "telegram"),
        ("source", ""),
        ("source", "tui-other"),
        ("chat_id", ""),
        ("chat_id", "runtime-other"),
        ("user_id", ""),
        ("user_id", "transport-other"),
        ("session_key", ""),
        ("session_key", "session-key-other"),
        ("profile", ""),
        ("profile", "profile-other"),
        ("ui_session_id", ""),
        ("ui_session_id", "runtime-other"),
    ],
)
def test_local_anchor_rejects_cleared_or_mismatched_core_coordinate(
    coordinate, replacement
):
    interaction = _issue(
        surface="desktop",
        platform="local",
        actor_id="transport-7",
        channel_id="transport-7",
        chat_id="runtime-9",
        thread_id=None,
        session_key="local-session-key-9",
        session_id="runtime-9",
        profile_id="work",
    )
    _publish_trusted_interaction(interaction)
    token = _bind_trusted_interaction(interaction)
    values = {
        "platform": "local",
        "source": interaction.surface,
        "chat_id": interaction.chat_id,
        "thread_id": "",
        "user_id": interaction.actor_id,
        "session_key": interaction.session_key,
        "session_id": "durable-session-9",
        "ui_session_id": interaction.chat_id,
        "profile": interaction.profile_id,
    }
    values[coordinate] = replacement
    session_tokens = set_session_vars(**values)
    try:
        assert get_current_trusted_interaction() is None
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)


def test_local_anchor_requires_durable_session_id_and_accepts_exact_coordinates():
    interaction = _issue(
        surface="desktop",
        platform="local",
        actor_id="transport-7",
        channel_id="transport-7",
        chat_id="runtime-9",
        thread_id=None,
        session_key="local-session-key-9",
        session_id="runtime-9",
        profile_id="work",
    )
    _publish_trusted_interaction(interaction)
    token = _bind_trusted_interaction(interaction)
    values = {
        "platform": "local",
        "source": interaction.surface,
        "chat_id": interaction.chat_id,
        "thread_id": "",
        "user_id": interaction.actor_id,
        "session_key": interaction.session_key,
        "session_id": "durable-session-9",
        "ui_session_id": interaction.chat_id,
        "profile": interaction.profile_id,
    }
    session_tokens = set_session_vars(**values)
    try:
        assert get_current_trusted_interaction() is interaction
    finally:
        clear_session_vars(session_tokens)

    missing_session_tokens = set_session_vars(**{**values, "session_id": ""})
    try:
        assert get_current_trusted_interaction() is None
    finally:
        clear_session_vars(missing_session_tokens)
        _reset_trusted_interaction(token)


def test_context_and_ledger_reset_return_none():
    interaction = _issue()
    _publish_trusted_interaction(interaction)
    token = _bind_trusted_interaction(interaction)
    session_tokens = set_session_vars(
        platform="telegram",
        source=interaction.surface,
        chat_id=interaction.chat_id,
        thread_id=interaction.thread_id or "",
        user_id=interaction.actor_id,
        session_key=interaction.session_key,
        session_id=interaction.session_id,
        message_id=interaction.message_id,
        profile=interaction.profile_id,
    )
    clear_session_vars(session_tokens)
    _reset_trusted_interaction(token)
    _clear_trusted_interactions(session_key=interaction.session_key)

    assert get_current_trusted_interaction() is None


def test_expired_ledger_entry_fails_closed(monkeypatch):
    import agent.trusted_interaction as module

    now = 10_000.0
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    interaction = _issue()
    _publish_trusted_interaction(interaction, ttl_seconds=2)
    token = _bind_trusted_interaction(interaction)
    session_tokens = set_session_vars(
        platform="telegram",
        source=interaction.surface,
        chat_id=interaction.chat_id,
        thread_id=interaction.thread_id or "",
        user_id=interaction.actor_id,
        session_key=interaction.session_key,
        session_id=interaction.session_id,
        message_id=interaction.message_id,
        profile=interaction.profile_id,
    )
    try:
        now = 10_003.0
        assert get_current_trusted_interaction() is None
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)


def test_ledger_size_bound_evicts_oldest_exact_scope():
    oldest = _issue()
    newest = _issue(
        chat_id="chat-new",
        channel_id="chat-new",
        thread_id=None,
        session_key="agent:main:telegram:dm:chat-new",
        session_id="session-new",
        message_id="message-new",
    )
    _publish_trusted_interaction(oldest, max_entries=1)
    _publish_trusted_interaction(newest, max_entries=1)
    token = _bind_trusted_interaction(oldest)
    session_tokens = set_session_vars(
        platform="telegram",
        source=oldest.surface,
        chat_id=oldest.chat_id,
        thread_id=oldest.thread_id or "",
        user_id=oldest.actor_id,
        session_key=oldest.session_key,
        session_id=oldest.session_id,
        message_id=oldest.message_id,
        profile=oldest.profile_id,
    )
    try:
        assert get_current_trusted_interaction() is None
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)


_IPC_MAC_KEY = bytes(range(32))
_WRONG_IPC_MAC_KEY = bytes(reversed(range(32)))


def _signed_payload(interaction: TrustedInteraction | None = None) -> dict:
    payload = _serialize_trusted_interaction(interaction or _issue())
    assert payload is not None
    signed = _attach_trusted_interaction_ipc_mac(
        payload,
        mac_key=_IPC_MAC_KEY,
    )
    assert signed is not None
    return signed


def test_host_private_ipc_round_trip_requires_valid_explicit_mac_key():
    interaction = _issue()
    payload = _serialize_trusted_interaction(interaction)
    assert payload is not None
    signed = _attach_trusted_interaction_ipc_mac(
        payload,
        mac_key=_IPC_MAC_KEY,
    )

    restored = _restore_trusted_interaction(signed, mac_key=_IPC_MAC_KEY)
    assert restored == interaction
    assert restored is not interaction
    assert _restore_trusted_interaction(signed, mac_key=None) is None
    assert (
        _restore_trusted_interaction(signed, mac_key=_WRONG_IPC_MAC_KEY)
        is None
    )


@pytest.mark.parametrize(
    "bad_key",
    [b"x" * 31, b"x" * 33, bytearray(b"x" * 32)],
    ids=["short", "long", "non-bytes"],
)
def test_host_private_ipc_rejects_malformed_mac_keys(bad_key):
    assert _restore_trusted_interaction(_signed_payload(), mac_key=bad_key) is None


def test_host_private_ipc_rejects_missing_mac_and_legacy_hash_only_payload():
    payload = _serialize_trusted_interaction(_issue())
    assert payload is not None

    assert "evidence_hash" in payload
    assert "ipc_mac" not in payload
    assert _restore_trusted_interaction(payload, mac_key=_IPC_MAC_KEY) is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("message_id", "tampered-message"),
        lambda payload: payload.__setitem__("evidence_hash", "0" * 64),
        lambda payload: payload.__setitem__("ipc_mac", "A" * 43),
        lambda payload: payload.__setitem__("ipc_mac", "not-base64%%%"),
        lambda payload: payload.__setitem__("ipc_mac", "non-ascii-é"),
        lambda payload: payload.__setitem__("ipc_mac", 17),
        lambda payload: payload.__setitem__("unexpected", "field"),
        lambda payload: payload.pop("actor_id"),
        lambda payload: payload.pop("ipc_mac"),
    ],
    ids=[
        "payload-tamper",
        "evidence-hash-tamper",
        "mac-tamper",
        "malformed-mac",
        "non-ascii-mac",
        "non-string-mac",
        "extra-field",
        "missing-field",
        "missing-mac",
    ],
)
def test_host_private_ipc_rejects_tamper_and_non_exact_field_sets(mutation):
    signed = _signed_payload()

    mutation(signed)

    assert _restore_trusted_interaction(signed, mac_key=_IPC_MAC_KEY) is None
