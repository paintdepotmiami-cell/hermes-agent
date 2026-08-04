from types import SimpleNamespace
import threading

import pytest


class _Transport:
    def write(self, _obj):
        return True

    def close(self):
        return None


def _install_prompt_handlers():
    from tui_gateway import methods_prompt

    server = SimpleNamespace(
        _methods={},
        _profile_scoped=lambda fn: fn,
    )
    methods_prompt.register(server)
    return server


def _local_request(monkeypatch):
    from tools import fresh_approval

    coordinator = fresh_approval.FreshApprovalCoordinator()
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    binding = fresh_approval.ConversationBinding(
        profile_id="default",
        platform="local_desktop",
        channel_id="desktop",
        chat_id="runtime-session-1",
        session_key="agent:main:local:runtime-session-1",
        session_id="runtime-session-1",
        thread_id=None,
        actor_id=None,
    )
    subject = fresh_approval.InvocationSubject(
        invocation_id="invocation-local-1",
        tool_call_id="tool-call-local-1",
        turn_id="turn-local-1",
        origin_message_id="prompt-rpc-1",
        args_hash="args-local-1",
        proposal_hash="proposal-local-1",
        conversation=binding,
    )
    return coordinator, coordinator.request(subject, ttl_seconds=30)


def test_fresh_approval_respond_accepts_only_id_and_decision_and_derives_host_facts(
    monkeypatch,
):
    from tui_gateway import server

    coordinator, request = _local_request(monkeypatch)
    owner = _Transport()
    server._sessions["runtime-session-1"] = {
        "transport": owner,
        "session_key": "agent:main:local:runtime-session-1",
        "source": "desktop",
        "profile_id": "default",
    }
    try:
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": "approval-rpc-22",
                "method": "approval.respond",
                "params": {
                    "approval_id": request.approval_id,
                    "decision": "approve_once",
                },
            },
            transport=owner,
        )
    finally:
        server._sessions.pop("runtime-session-1", None)

    assert response["id"] == "approval-rpc-22"
    assert response["result"] == {"resolved": True, "outcome": "approved"}
    resolution = coordinator.await_resolution(request.approval_id, timeout=0)
    responder = resolution.evidence.responder
    assert responder.assurance == "local_desktop_session"
    assert responder.principal_id is None
    assert responder.binding == request.subject.conversation
    assert responder.inbound_id == "approval-rpc-22"


def test_governed_local_binding_matches_exact_host_transport_provenance(
    monkeypatch,
):
    from tools import fresh_approval
    from tui_gateway.transport import _host_transport_identity

    coordinator = fresh_approval.FreshApprovalCoordinator()
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    owner = _Transport()
    transport_id = _host_transport_identity(owner)
    binding = fresh_approval.ConversationBinding(
        profile_id="default",
        platform="local",
        channel_id=transport_id,
        chat_id="runtime-session-1",
        session_key="agent:main:local:runtime-session-1",
        session_id="runtime-session-1",
        thread_id=None,
        actor_id=transport_id,
    )
    request = coordinator.request(
        fresh_approval.InvocationSubject(
            invocation_id="governed-invocation-local-1",
            tool_call_id="governed-tool-call-local-1",
            turn_id="governed-turn-local-1",
            origin_message_id="governed-prompt-rpc-1",
            args_hash="governed-operation-hash-local-1",
            proposal_hash="governed-proposal-hash-local-1",
            conversation=binding,
        ),
        ttl_seconds=30,
    )
    server = _install_prompt_handlers()
    server._ok = lambda rid, result: {"id": rid, "result": result}
    server._err = lambda rid, code, message: {
        "id": rid,
        "error": {"code": code, "message": message},
    }
    server.current_transport = lambda: owner
    server._sessions_lock = threading.RLock()
    server._sessions = {"runtime-session-1": {
        "transport": owner,
        "session_key": binding.session_key,
        "source": "desktop",
        "profile_id": "default",
    }}
    response = server._methods["approval.respond"](
        "governed-approval-rpc-22",
        {
            "approval_id": request.approval_id,
            "decision": "approve_once",
        },
    )

    assert response["result"] == {"resolved": True, "outcome": "approved"}
    resolution = coordinator.await_resolution(request.approval_id, timeout=0)
    assert resolution.evidence.responder.binding == binding
    assert resolution.evidence.responder.inbound_id == (
        "governed-approval-rpc-22"
    )


@pytest.mark.parametrize("field", ["principal_id", "session_id", "user_id"])
def test_fresh_approval_respond_rejects_client_claimed_identity(monkeypatch, field):
    _coordinator, request = _local_request(monkeypatch)
    server = _install_prompt_handlers()
    server._ok = lambda rid, result: {"id": rid, "result": result}
    server._err = lambda rid, code, message: {
        "id": rid,
        "error": {"code": code, "message": message},
    }

    response = server._methods["approval.respond"](
        "approval-rpc-23",
        {
            "approval_id": request.approval_id,
            "decision": "approve_once",
            field: "fabricated-client-claim",
        },
    )

    assert response["error"]["code"] == -32602
    assert "only approval_id and decision" in response["error"]["message"]


def test_fresh_approval_respond_requires_backend_rpc_request_id(monkeypatch):
    _coordinator, request = _local_request(monkeypatch)
    server = _install_prompt_handlers()
    server._ok = lambda rid, result: {"id": rid, "result": result}
    server._err = lambda rid, code, message: {
        "id": rid,
        "error": {"code": code, "message": message},
    }

    response = server._methods["approval.respond"](
        None,
        {"approval_id": request.approval_id, "decision": "approve_once"},
    )

    assert response["error"]["code"] == -32600
    assert "rpc request id" in response["error"]["message"].lower()


def test_public_fresh_approval_respond_requires_exact_transport_owned_session(
    monkeypatch,
):
    from tui_gateway import server

    owner = _Transport()
    other = _Transport()
    owner_sid = "runtime-session-1"
    other_sid = "runtime-session-other"

    def install_sessions():
        server._sessions[owner_sid] = {
            "transport": owner,
            "session_key": "agent:main:local:runtime-session-1",
            "source": "desktop",
            "profile_id": "default",
        }
        server._sessions[other_sid] = {
            "transport": other,
            "session_key": "agent:main:local:runtime-session-other",
            "source": "desktop",
            "profile_id": "default",
        }

    def rpc(request, rid):
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "approval.respond",
            "params": {
                "approval_id": request.approval_id,
                "decision": "approve_once",
            },
        }

    try:
        install_sessions()
        coordinator, request = _local_request(monkeypatch)
        response = server.dispatch(rpc(request, "approval-owner"), transport=owner)
        assert response["result"] == {"resolved": True, "outcome": "approved"}

        install_sessions()
        coordinator, request = _local_request(monkeypatch)
        response = server.dispatch(rpc(request, "approval-other"), transport=other)
        assert response["error"]["code"] == 4009
        assert coordinator.await_resolution(request.approval_id, timeout=0) is None

        install_sessions()
        coordinator, request = _local_request(monkeypatch)
        response = server.handle_request(rpc(request, "approval-no-transport"))
        assert response["error"]["code"] == 4009
        assert coordinator.await_resolution(request.approval_id, timeout=0) is None
    finally:
        server._sessions.pop(owner_sid, None)
        server._sessions.pop(other_sid, None)


def test_legacy_approval_respond_still_uses_session_fifo_without_responder(
    monkeypatch,
):
    from tools import approval

    server = _install_prompt_handlers()
    server._ok = lambda rid, result: {"id": rid, "result": result}
    server._err = lambda rid, code, message: {
        "id": rid,
        "error": {"code": code, "message": message},
    }
    server._sess = lambda params, rid: (
        {"session_key": "legacy-session"},
        None,
    )
    calls = []
    monkeypatch.setattr(
        approval,
        "resolve_gateway_approval",
        lambda session_key, choice, resolve_all=False: (
            calls.append((session_key, choice, resolve_all)) or 1
        ),
    )

    response = server._methods["approval.respond"](
        "legacy-rpc",
        {"session_id": "runtime-legacy", "choice": "once"},
    )

    assert response["result"] == {"resolved": 1}
    assert calls == [("legacy-session", "once", False)]
