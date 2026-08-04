from __future__ import annotations

import threading
import types

import pytest

from agent.trusted_interaction import get_current_trusted_interaction
from tui_gateway import server


_REAL_THREAD = threading.Thread


class _Transport:
    def write(self, _obj):
        return True

    def close(self):
        return None


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        del daemon
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        del timeout


def _session(transport, agent, sid="runtime-1"):
    return {
        "agent": agent,
        "session_key": f"agent:main:local:{sid}",
        "profile_id": "default",
        "source": "desktop",
        "transport": transport,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        "last_active": 0.0,
    }


def _rpc(rid, sid, text, **extra):
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "method": "prompt.submit",
        "params": {"session_id": sid, "text": text, **extra},
    }


@pytest.fixture(autouse=True)
def _clean_sessions():
    yield
    for session in list(server._sessions.values()):
        key = str(session.get("session_key") or "")
        if key:
            server._clear_local_trusted_interaction(session_key=key)
    server._sessions.clear()


@pytest.fixture()
def _public_turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_resolve_session_platform", lambda: "desktop")
    monkeypatch.setattr(
        server,
        "_load_dashboard_process_isolation_config",
        lambda: {},
    )
    monkeypatch.setattr(
        server,
        "_session_uses_compute_host",
        lambda session, cfg: False,
    )
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    monkeypatch.setattr(
        server,
        "_wait_agent_for_prompt",
        lambda session, rid, sid: None,
    )
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(
        server,
        "_sync_agent_model_with_config",
        lambda sid, session: None,
    )
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(
        server,
        "_sync_session_key_after_compress",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)


def test_public_prompt_submit_binds_server_owned_interaction_and_resets_context(
    _public_turn_env,
):
    seen = []

    def run_conversation(*_args, **_kwargs):
        seen.append(get_current_trusted_interaction())
        return {"final_response": "done"}

    transport = _Transport()
    agent = types.SimpleNamespace(
        session_id="agent:main:local:runtime-1",
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    server._sessions["runtime-1"] = _session(transport, agent)

    response = server.dispatch(
        _rpc(
            "client-rpc-id",
            "runtime-1",
            "hello",
            actor_id="forged-actor",
            message_id="forged-message",
        ),
        transport=transport,
    )

    assert response["result"] == {"status": "streaming"}
    assert len(seen) == 1
    interaction = seen[0]
    assert interaction is not None
    assert interaction.surface == "desktop"
    assert interaction.platform == "local"
    assert interaction.chat_id == "runtime-1"
    assert interaction.session_id == "runtime-1"
    assert interaction.session_key == "agent:main:local:runtime-1"
    assert interaction.profile_id == "default"
    assert interaction.actor_id.startswith("local_transport_")
    assert interaction.actor_id == interaction.channel_id
    assert interaction.actor_id != "forged-actor"
    assert interaction.message_id.startswith("local_message_")
    assert interaction.message_id not in {"client-rpc-id", "forged-message"}
    assert get_current_trusted_interaction() is None


def test_public_busy_merged_prompts_keep_latest_and_reject_other_transport(
    monkeypatch,
):
    owner = _Transport()
    intruder = _Transport()
    agent = types.SimpleNamespace(interrupt=lambda *_args: None)
    session = _session(owner, agent)
    session["running"] = True
    server._sessions["runtime-1"] = session
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"display": {"busy_input_mode": "queue"}},
    )

    first = server.dispatch(
        _rpc("busy-1", "runtime-1", "one"),
        transport=owner,
    )
    first_interaction = session["queued_prompt"]["trusted_interaction"]
    second = server.dispatch(
        _rpc("busy-2", "runtime-1", "two"),
        transport=owner,
    )
    latest = session["queued_prompt"]["trusted_interaction"]

    assert first["result"]["status"] == "queued"
    assert second["result"]["status"] == "queued"
    assert session["queued_prompt"]["text"] == "one\n\ntwo"
    assert latest.interaction_id != first_interaction.interaction_id
    assert latest.received_at >= first_interaction.received_at

    rejected = server.dispatch(
        _rpc("busy-3", "runtime-1", "intrusion"),
        transport=intruder,
    )
    assert rejected["error"]["code"] == 4009
    assert session["queued_prompt"]["text"] == "one\n\ntwo"
    assert session["queued_prompt"]["trusted_interaction"] is latest


def test_concurrent_public_sessions_do_not_cross_leak(
    _public_turn_env,
    monkeypatch,
):
    monkeypatch.setattr(server.threading, "Thread", _REAL_THREAD)
    barrier = threading.Barrier(2)
    finished = threading.Event()
    seen = {}
    seen_lock = threading.Lock()

    def agent_for(sid):
        def run_conversation(*_args, **_kwargs):
            barrier.wait(timeout=5)
            with seen_lock:
                seen[sid] = get_current_trusted_interaction()
                if len(seen) == 2:
                    finished.set()
            return {"final_response": "done"}

        return types.SimpleNamespace(
            session_id=f"agent:main:local:{sid}",
            run_conversation=run_conversation,
            clear_interrupt=lambda: None,
        )

    transport_a = _Transport()
    transport_b = _Transport()
    server._sessions["runtime-a"] = _session(
        transport_a,
        agent_for("runtime-a"),
        "runtime-a",
    )
    server._sessions["runtime-b"] = _session(
        transport_b,
        agent_for("runtime-b"),
        "runtime-b",
    )

    response_a = server.dispatch(
        _rpc("rpc-a", "runtime-a", "A"),
        transport=transport_a,
    )
    response_b = server.dispatch(
        _rpc("rpc-b", "runtime-b", "B"),
        transport=transport_b,
    )

    assert response_a["result"] == {"status": "streaming"}
    assert response_b["result"] == {"status": "streaming"}
    assert finished.wait(5)
    assert seen["runtime-a"].chat_id == "runtime-a"
    assert seen["runtime-b"].chat_id == "runtime-b"
    assert seen["runtime-a"].session_key.endswith("runtime-a")
    assert seen["runtime-b"].session_key.endswith("runtime-b")
    assert seen["runtime-a"].actor_id != seen["runtime-b"].actor_id
    for sid in ("runtime-a", "runtime-b"):
        run_thread = server._sessions[sid].get("_run_thread")
        if run_thread is not None:
            run_thread.join(timeout=5)


def test_approval_respond_is_distinct_and_does_not_mint_action_provenance(
    monkeypatch,
):
    from tools import fresh_approval

    owner = _Transport()
    session_key = "agent:main:local:runtime-1"
    server._sessions["runtime-1"] = {
        "transport": owner,
        "session_key": session_key,
        "source": "desktop",
        "profile_id": "default",
    }
    coordinator = fresh_approval.FreshApprovalCoordinator()
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    binding = fresh_approval.ConversationBinding(
        profile_id="default",
        platform="local_desktop",
        channel_id="desktop",
        chat_id="runtime-1",
        session_key=session_key,
        session_id="runtime-1",
        thread_id=None,
        actor_id=None,
    )
    request = coordinator.request(
        fresh_approval.InvocationSubject(
            invocation_id="invocation-1",
            tool_call_id="tool-call-1",
            turn_id="turn-1",
            origin_message_id="original-prompt-message",
            args_hash="args-hash",
            proposal_hash="proposal-hash",
            conversation=binding,
        ),
        ttl_seconds=30,
    )
    observed = []
    original_resolve = coordinator.resolve

    def resolve(*args, **kwargs):
        observed.append(get_current_trusted_interaction())
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(coordinator, "resolve", resolve)
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "approval-response-rpc",
            "method": "approval.respond",
            "params": {
                "approval_id": request.approval_id,
                "decision": "approve_once",
            },
        },
        transport=owner,
    )

    assert response["result"] == {"resolved": True, "outcome": "approved"}
    assert observed == [None]
    resolution = coordinator.await_resolution(request.approval_id, timeout=0)
    assert resolution.subject.origin_message_id == (
        "original-prompt-message"
    )
    assert resolution.evidence.responder.inbound_id == "approval-response-rpc"
