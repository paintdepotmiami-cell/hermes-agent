from dataclasses import FrozenInstanceError, asdict, replace
from concurrent.futures import ThreadPoolExecutor

import asyncio
import hashlib
import json

import pytest


def _api():
    from tools.fresh_approval import (
        ConversationBinding,
        FreshApprovalCoordinator,
        InvocationSubject,
    )

    binding = ConversationBinding(
        profile_id="profile-main",
        platform="telegram",
        channel_id="telegram-account-primary",
        chat_id="chat-11",
        session_key="agent:main:telegram:group:chat-11:thread-3",
        session_id="session-19",
        thread_id="thread-3",
        actor_id="operator-7",
    )
    subject = InvocationSubject(
        invocation_id="invocation-1",
        tool_call_id="tool-call-2",
        turn_id="turn-3",
        origin_message_id="message-4",
        args_hash="args-sha256",
        proposal_hash="proposal-sha256",
        conversation=binding,
    )
    return FreshApprovalCoordinator(), binding, subject


def test_fresh_approval_success_returns_exact_one_use_grant():
    coordinator, binding, subject = _api()

    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id="operator-7",
        binding=binding,
        inbound_id="callback-5",
    )
    resolution = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    )

    assert request.mode == "fresh"
    assert request.subject is subject
    assert request.approval_id
    assert resolution.outcome == "approved"
    assert resolution.subject is subject
    assert resolution.evidence is not None
    assert resolution.evidence.responder.binding == responder.binding
    assert resolution.evidence.responder.host_nonce == responder.host_nonce
    assert not hasattr(resolution.evidence.responder, "_issuer")
    assert resolution.grant is not None
    assert "opaque" in repr(resolution.grant).lower()
    assert coordinator.consume(resolution.grant, subject) is True


def test_local_desktop_session_needs_no_fabricated_principal():
    coordinator, binding, subject = _api()
    local_binding = type(binding)(
        profile_id="default",
        platform="local_desktop",
        channel_id="desktop",
        chat_id="runtime-session-19",
        session_key="agent:main:local:session-19",
        session_id="runtime-session-19",
        thread_id=None,
        actor_id=None,
    )
    local_subject = type(subject)(
        invocation_id=subject.invocation_id,
        tool_call_id=subject.tool_call_id,
        turn_id=subject.turn_id,
        origin_message_id=subject.origin_message_id,
        args_hash=subject.args_hash,
        proposal_hash=subject.proposal_hash,
        conversation=local_binding,
    )
    request = coordinator.request(local_subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="local_desktop_session",
        principal_id=None,
        binding=local_binding,
        inbound_id="rpc-22",
    )

    resolution = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    )

    assert resolution.outcome == "approved"
    assert resolution.evidence.responder.principal_id is None
    assert coordinator.consume(resolution.grant, local_subject) is True


def test_approval_value_objects_are_immutable():
    coordinator, _binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)

    with pytest.raises(FrozenInstanceError):
        request.mode = "legacy"


@pytest.mark.parametrize("decision", ["yes", "session", "always", "approve"])
def test_fresh_mode_rejects_every_decision_except_approve_once_or_deny(decision):
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-5",
    )

    with pytest.raises(ApprovalRejected, match="decision"):
        coordinator.resolve(request.approval_id, decision=decision, responder=responder)


def test_origin_message_cannot_supply_fresh_approval():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id=subject.origin_message_id,
    )

    with pytest.raises(ApprovalRejected, match="origin"):
        coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=responder,
        )


def test_missing_authenticated_responder_is_rejected_in_fresh_mode():
    from tools.fresh_approval import ApprovalRejected

    coordinator, _binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)

    with pytest.raises(ApprovalRejected, match="responder"):
        coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=None,
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("actor_id", "operator-other"),
        ("profile_id", "profile-other"),
        ("platform", "telegram-other"),
        ("channel_id", "telegram-other"),
        ("chat_id", "chat-other"),
        ("thread_id", "thread-other"),
        ("session_key", "session-key-other"),
        ("session_id", "session-other"),
    ],
)
def test_responder_must_match_every_conversation_coordinate(field, wrong_value):
    from tools.fresh_approval import ApprovalRejected, ConversationBinding

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    values = {
        "profile_id": binding.profile_id,
        "platform": binding.platform,
        "channel_id": binding.channel_id,
        "chat_id": binding.chat_id,
        "session_key": binding.session_key,
        "session_id": binding.session_id,
        "thread_id": binding.thread_id,
        "actor_id": binding.actor_id,
    }
    values[field] = wrong_value
    wrong_binding = ConversationBinding(**values)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=wrong_binding.actor_id,
        binding=wrong_binding,
        inbound_id=f"callback-wrong-{field}",
    )

    with pytest.raises(ApprovalRejected, match="binding"):
        coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=responder,
        )


def test_wrong_approval_id_does_not_fall_back_to_fifo():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    first = coordinator.request(subject, ttl_seconds=30)
    second = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-unknown-id",
    )

    with pytest.raises(ApprovalRejected, match="unknown"):
        coordinator.resolve(
            "not-either-request-id",
            decision="approve_once",
            responder=responder,
        )

    assert first.approval_id != second.approval_id


def test_expired_request_blocks_resolution_and_awaits_as_expired():
    from tools.fresh_approval import ApprovalRejected, FreshApprovalCoordinator

    now = [100.0]
    _unused, binding, subject = _api()
    coordinator = FreshApprovalCoordinator(clock=lambda: now[0])
    request = coordinator.request(subject, ttl_seconds=5)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-after-expiry",
    )
    now[0] = 106.0

    with pytest.raises(ApprovalRejected, match="expired"):
        coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=responder,
        )

    resolution = coordinator.await_resolution(request.approval_id, timeout=0)
    assert resolution.outcome == "expired"
    assert resolution.grant is None


def test_cancel_blocks_later_evidence():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    cancelled = coordinator.cancel(request.approval_id)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-after-cancel",
    )

    assert cancelled.outcome == "cancelled"
    assert coordinator.await_resolution(request.approval_id, timeout=0) is cancelled
    with pytest.raises(ApprovalRejected, match="resolved"):
        coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=responder,
        )


def test_duplicate_resolution_and_responder_replay_are_rejected():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    first = coordinator.request(subject, ttl_seconds=30)
    second = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-once-only",
    )
    coordinator.resolve(first.approval_id, decision="approve_once", responder=responder)

    with pytest.raises(ApprovalRejected, match="resolved"):
        coordinator.resolve(first.approval_id, decision="deny", responder=responder)
    with pytest.raises(ApprovalRejected, match="replay"):
        coordinator.resolve(
            second.approval_id, decision="approve_once", responder=responder
        )


def test_deny_is_terminal_and_never_mints_a_grant():
    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="deny-message",
    )

    resolution = coordinator.resolve(
        request.approval_id,
        decision="deny",
        responder=responder,
    )

    assert resolution.outcome == "denied"
    assert resolution.grant is None
    assert coordinator.await_resolution(request.approval_id, timeout=0) is resolution


@pytest.mark.parametrize(
    "change",
    [
        {"invocation_id": "wrong-invocation"},
        {"tool_call_id": "wrong-tool-call"},
        {"turn_id": "wrong-turn"},
        {"args_hash": "wrong-args-hash"},
        {"proposal_hash": "wrong-proposal-hash"},
    ],
)
def test_grant_is_bound_to_every_invocation_and_hash_coordinate(change):
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id=f"callback-{next(iter(change))}",
    )
    resolution = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    )

    with pytest.raises(ApprovalRejected, match="subject mismatch"):
        coordinator.consume(resolution.grant, replace(subject, **change))
    assert coordinator.consume(resolution.grant, subject) is True


def test_consumed_grant_cannot_be_replayed():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-consume",
    )
    grant = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    ).grant

    assert coordinator.consume(grant, subject) is True
    with pytest.raises(ApprovalRejected, match="consumed"):
        coordinator.consume(grant, subject)


def test_responder_authenticated_before_request_is_not_fresh():
    from tools.fresh_approval import ApprovalRejected, FreshApprovalCoordinator

    now = [100.0]
    _unused, binding, subject = _api()
    coordinator = FreshApprovalCoordinator(clock=lambda: now[0])
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="old-message",
    )
    now[0] = 101.0
    request = coordinator.request(subject, ttl_seconds=30)

    with pytest.raises(ApprovalRejected, match="fresh"):
        coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=responder,
        )


def test_thread_waiter_receives_exact_resolution():
    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-thread-waiter",
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(
            coordinator.await_resolution,
            request.approval_id,
            timeout=2,
        )
        expected = coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=responder,
        )

    assert waiting.result(timeout=2) is expected


@pytest.mark.asyncio
async def test_async_waiter_does_not_block_event_loop():
    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="rpc-async-waiter",
    )
    waiting = asyncio.create_task(
        coordinator.await_resolution_async(request.approval_id, timeout=2)
    )
    await asyncio.sleep(0)

    expected = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    )

    assert await waiting is expected


def test_concurrent_responses_resolve_exactly_once():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)

    def attempt(index):
        responder = coordinator._mint_authenticated_responder_for_host(
            assurance="telegram_authenticated",
            principal_id=binding.actor_id,
            binding=binding,
            inbound_id=f"callback-race-{index}",
        )
        try:
            return coordinator.resolve(
                request.approval_id,
                decision="approve_once",
                responder=responder,
            ).outcome
        except ApprovalRejected as exc:
            return exc.reason

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(16)))

    assert outcomes.count("approved") == 1
    assert outcomes.count("approval_already_resolved") == 15


def test_concurrent_grant_consumption_succeeds_exactly_once():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-consume-race",
    )
    grant = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    ).grant

    def consume(_index):
        try:
            return coordinator.consume(grant, subject)
        except ApprovalRejected as exc:
            return exc.reason

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(consume, range(16)))

    assert outcomes.count(True) == 1
    assert outcomes.count("unknown_or_consumed_approval_grant") == 15


@pytest.mark.parametrize(
    "field",
    [
        "invocation_id",
        "tool_call_id",
        "turn_id",
        "origin_message_id",
        "args_hash",
        "proposal_hash",
    ],
)
def test_request_rejects_missing_subject_coordinates(field):
    coordinator, _binding, subject = _api()

    with pytest.raises(ValueError, match=field):
        coordinator.request(replace(subject, **{field: ""}), ttl_seconds=30)


def test_request_rejects_missing_conversation_coordinates():
    coordinator, binding, subject = _api()

    for field in (
        "profile_id",
        "platform",
        "channel_id",
        "chat_id",
        "session_key",
        "session_id",
    ):
        with pytest.raises(ValueError, match=field):
            coordinator.request(
                replace(subject, conversation=replace(binding, **{field: ""})),
                ttl_seconds=30,
            )


def test_authenticated_responder_requires_server_inbound_id():
    from tools.fresh_approval import ApprovalRejected

    coordinator, binding, _subject = _api()

    with pytest.raises(ApprovalRejected, match="inbound"):
        coordinator._mint_authenticated_responder_for_host(
            assurance="telegram_authenticated",
            principal_id=binding.actor_id,
            binding=binding,
            inbound_id="",
        )


def test_responder_issuer_is_an_explicitly_private_host_only_seam():
    coordinator, _binding, _subject = _api()

    assert not hasattr(coordinator, "authenticate_responder")
    assert callable(coordinator._mint_authenticated_responder_for_host)
    assert (
        "api separation"
        in (coordinator._mint_authenticated_responder_for_host.__doc__ or "").lower()
    )


def test_complete_binding_keeps_host_routing_and_session_ids_separate():
    from tools.fresh_approval import ConversationBinding

    binding = ConversationBinding(
        profile_id="profile-main",
        platform="telegram",
        channel_id="telegram-account-primary",
        chat_id="chat-11",
        thread_id="thread-3",
        session_key="agent:main:telegram:group:chat-11:thread-3",
        session_id="stored-session-19",
        actor_id="operator-7",
    )

    assert binding.profile_id == "profile-main"
    assert binding.platform == "telegram"
    assert binding.channel_id == "telegram-account-primary"
    assert binding.chat_id == "chat-11"
    assert binding.thread_id == "thread-3"
    assert binding.session_key == "agent:main:telegram:group:chat-11:thread-3"
    assert binding.session_id == "stored-session-19"
    assert binding.actor_id == "operator-7"


def test_evidence_hash_is_canonical_deterministic_and_excludes_issuer_secret():
    from tools.fresh_approval import FreshApprovalCoordinator

    _unused, binding, subject = _api()
    now = [100.0]
    coordinator = FreshApprovalCoordinator(clock=lambda: now[0])
    request = coordinator.request(subject, ttl_seconds=30)
    now[0] = 101.0
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-canonical",
    )
    now[0] = 102.0
    evidence = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    ).evidence

    binding_payload = {
        "actor_id": binding.actor_id,
        "channel_id": binding.channel_id,
        "chat_id": binding.chat_id,
        "platform": binding.platform,
        "profile_id": binding.profile_id,
        "session_id": binding.session_id,
        "session_key": binding.session_key,
        "thread_id": binding.thread_id,
    }
    payload = {
        "approval_id": request.approval_id,
        "confirmed_at": 102.0,
        "decision": "approve_once",
        "request": {"created_at": 100.0, "expires_at": 130.0},
        "responder": {
            "assurance": "telegram_authenticated",
            "authenticated_at": 101.0,
            "binding": binding_payload,
            "host_nonce": responder.host_nonce,
            "inbound_id": "callback-canonical",
            "principal_id": binding.actor_id,
        },
        "subject": {
            "args_hash": subject.args_hash,
            "conversation": binding_payload,
            "invocation_id": subject.invocation_id,
            "origin_message_id": subject.origin_message_id,
            "proposal_hash": subject.proposal_hash,
            "tool_call_id": subject.tool_call_id,
            "turn_id": subject.turn_id,
        },
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert evidence.confirmed_at == 102.0
    assert evidence.evidence_hash == hashlib.sha256(canonical).hexdigest()
    assert len(evidence.evidence_hash) == 64
    assert evidence.evidence_hash == evidence.evidence_hash.lower()
    assert "_issuer" not in asdict(evidence)["responder"]
    assert coordinator.verify_evidence(evidence) is True


def test_evidence_verification_rejects_decision_and_binding_tampering():
    coordinator, binding, subject = _api()
    request = coordinator.request(subject, ttl_seconds=30)
    responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-tamper",
    )
    evidence = coordinator.resolve(
        request.approval_id,
        decision="approve_once",
        responder=responder,
    ).evidence

    assert coordinator.verify_evidence(replace(evidence, decision="deny")) is False
    tampered_binding = replace(evidence.responder.binding, chat_id="other-chat")
    tampered_responder = replace(evidence.responder, binding=tampered_binding)
    assert (
        coordinator.verify_evidence(replace(evidence, responder=tampered_responder))
        is False
    )


def test_replay_state_is_retained_then_pruned_without_dropping_live_state():
    from tools.fresh_approval import ApprovalRejected, FreshApprovalCoordinator

    _unused, binding, subject = _api()
    now = [100.0]
    coordinator = FreshApprovalCoordinator(
        clock=lambda: now[0],
        replay_ttl=10,
    )
    short = coordinator.request(subject, ttl_seconds=5)
    short_responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-short-replay",
    )
    coordinator.resolve(
        short.approval_id,
        decision="approve_once",
        responder=short_responder,
    )
    long_grant_request = coordinator.request(subject, ttl_seconds=100)
    long_responder = coordinator._mint_authenticated_responder_for_host(
        assurance="telegram_authenticated",
        principal_id=binding.actor_id,
        binding=binding,
        inbound_id="callback-long-replay",
    )
    long_grant = coordinator.resolve(
        long_grant_request.approval_id,
        decision="approve_once",
        responder=long_responder,
    ).grant
    pending = coordinator.request(subject, ttl_seconds=100)

    now[0] = 106.0
    assert coordinator.get_request(short.approval_id) is short
    assert coordinator.get_request(pending.approval_id) is pending
    assert coordinator.consume(long_grant, subject) is True
    with pytest.raises(ApprovalRejected, match="resolved"):
        coordinator.resolve(
            short.approval_id,
            decision="deny",
            responder=short_responder,
        )

    now[0] = 111.0
    assert coordinator.get_request(pending.approval_id) is pending
    with pytest.raises(ApprovalRejected, match="unknown"):
        coordinator.get_request(short.approval_id)
    assert short_responder.host_nonce not in coordinator._seen_responder_nonces
    assert all(
        inbound_id != short_responder.inbound_id
        for _assurance, _binding, inbound_id in coordinator._seen_inbound
    )

    now[0] = 211.0
    replacement = coordinator.request(subject, ttl_seconds=5)
    assert set(coordinator._records) == {replacement.approval_id}
    assert coordinator._grants == {}
    assert coordinator._seen_responder_nonces == {}
    assert coordinator._seen_inbound == {}


@pytest.mark.parametrize(
    "unsafe_time", [float("nan"), float("inf"), float("-inf"), 1 << 53]
)
def test_request_rejects_non_finite_or_unsafe_numeric_times(unsafe_time):
    from tools.fresh_approval import FreshApprovalCoordinator

    _unused, _binding, subject = _api()
    coordinator = FreshApprovalCoordinator(clock=lambda: unsafe_time)

    with pytest.raises(ValueError, match="finite|safe integer|safe integer range"):
        coordinator.request(subject, ttl_seconds=30)


@pytest.mark.parametrize("field", ["thread_id", "actor_id"])
def test_optional_binding_strings_reject_blank_values(field):
    coordinator, binding, subject = _api()
    blank_binding = replace(binding, **{field: "  "})

    with pytest.raises(ValueError, match=field):
        coordinator.request(
            replace(subject, conversation=blank_binding),
            ttl_seconds=30,
        )
