from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent.trusted_interaction import get_current_trusted_interaction
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionEntry, SessionSource, build_session_key


class _TelegramAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        del is_reconnect
        return True

    async def disconnect(self):
        return None

    async def send(self, *args, **kwargs):
        del args, kwargs
        return SendResult(success=True, message_id="bot-message")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(
    *,
    chat_id="chat-1",
    user_id="actor-1",
    thread_id="thread-1",
    message_id="message-1",
):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        thread_id=thread_id,
        message_id=message_id,
    )


def _event(
    text="hello",
    *,
    chat_id="chat-1",
    user_id="actor-1",
    thread_id="thread-1",
    message_id="message-1",
    source_message_id=None,
    raw_thread_id=None,
):
    source = _source(
        chat_id=chat_id,
        user_id=user_id,
        thread_id=thread_id,
        message_id=source_message_id or message_id,
    )
    raw = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=chat_id),
        message_id=message_id,
        message_thread_id=(
            thread_id if raw_thread_id is None else raw_thread_id
        ),
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=raw,
        message_id=message_id,
        timestamp=datetime.now(timezone.utc),
    )


def _runner(monkeypatch, tmp_path):
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="unused")
        }
    )
    runner = gateway_run.GatewayRunner(config)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda source: source.user_id == "actor-1"
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda source: None
    runner._is_telegram_topic_lane = lambda source: False
    runner._cache_session_source = lambda key, source: None
    runner._is_session_run_current = lambda key, generation: True
    runner._begin_session_run_generation = lambda key: 1
    runner._reply_anchor_for_event = lambda event: None
    runner._get_guild_id = lambda event: None
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    adapter = _TelegramAdapter(
        PlatformConfig(enabled=True, token="unused"),
        Platform.TELEGRAM,
    )
    adapter._busy_text_mode = ""
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: adapter}

    source = _source()
    session_key = build_session_key(source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="server-session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.peek_session_id.return_value = "server-session-1"
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "unused"},
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner, adapter, session_entry


def _success_result():
    return {
        "final_response": "done",
        "messages": [],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model": "test/model",
    }


@pytest.mark.asyncio
async def test_telegram_normal_and_public_busy_input_reach_executor_as_latest(
    monkeypatch,
    tmp_path,
):
    runner, adapter, session_entry = _runner(monkeypatch, tmp_path)
    session_key = session_entry.session_key
    started = asyncio.Event()
    busy_accepted = asyncio.Event()
    seen = []
    tool_seen = []
    tool_name = "trusted_interaction_test_probe"

    import model_tools
    from tools.registry import registry

    def tool_handler(_args, **_kwargs):
        tool_seen.append(get_current_trusted_interaction())
        return "ok"

    registry.register(
        name=tool_name,
        toolset="trusted-interaction-test",
        schema={
            "name": tool_name,
            "description": "test-only provenance probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=tool_handler,
    )

    def execute_probe():
        return model_tools.handle_function_call(
            tool_name,
            {},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )

    async def run_agent(**_kwargs):
        await runner._run_in_executor_with_context(execute_probe)
        seen.append(tool_seen[-1])
        started.set()
        await asyncio.wait_for(busy_accepted.wait(), timeout=5)
        await runner._run_in_executor_with_context(execute_probe)
        seen.append(tool_seen[-1])
        return _success_result()

    runner._run_agent = run_agent
    runner._busy_input_mode = "queue"
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()

    normal = _event()
    turn = asyncio.create_task(
        runner._handle_message_with_agent(
            normal,
            normal.source,
            session_key,
            1,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    busy = _event(text="busy follow-up", message_id="message-2")
    assert build_session_key(
        busy.source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
    ) == session_key
    await adapter.handle_message(busy)
    assert getattr(busy, "_trusted_interaction", None) is not None
    busy_accepted.set()
    try:
        await asyncio.wait_for(turn, timeout=5)

        assert [item.message_id for item in seen] == ["message-1", "message-2"]
        assert all(item.actor_id == "actor-1" for item in seen)
        assert all(item.chat_id == "chat-1" for item in seen)
        assert all(item.thread_id == "thread-1" for item in seen)
        assert all(item.session_key == session_key for item in seen)
        assert all(item.session_id == "server-session-1" for item in seen)
        assert adapter._pending_messages[session_key] is busy
        assert get_current_trusted_interaction() is None
    finally:
        registry.deregister(tool_name)
        runner._shutdown_executor()


@pytest.mark.asyncio
async def test_unauthorized_busy_input_does_not_advance_running_turn_provenance(
    monkeypatch,
    tmp_path,
):
    runner, adapter, session_entry = _runner(monkeypatch, tmp_path)
    session_key = session_entry.session_key
    started = asyncio.Event()
    rejected = asyncio.Event()
    seen = []

    async def run_agent(**_kwargs):
        seen.append(get_current_trusted_interaction())
        started.set()
        await asyncio.wait_for(rejected.wait(), timeout=5)
        seen.append(get_current_trusted_interaction())
        return _success_result()

    runner._run_agent = run_agent
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()
    normal = _event()
    turn = asyncio.create_task(
        runner._handle_message_with_agent(
            normal,
            normal.source,
            session_key,
            1,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    intruder = _event(
        text="not authorized",
        user_id="intruder",
        message_id="message-intruder",
    )
    await adapter.handle_message(intruder)
    rejected.set()
    await asyncio.wait_for(turn, timeout=5)

    assert [item.message_id for item in seen] == ["message-1", "message-1"]
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_kwargs"),
    [
        {"source_message_id": "source-says-other"},
        {"raw_thread_id": "raw-thread-other"},
    ],
)
async def test_telegram_message_or_thread_mismatch_fails_closed(
    monkeypatch,
    tmp_path,
    event_kwargs,
):
    runner, _adapter, session_entry = _runner(monkeypatch, tmp_path)
    seen = []

    async def run_agent(**_kwargs):
        seen.append(get_current_trusted_interaction())
        return _success_result()

    runner._run_agent = run_agent
    event = _event(**event_kwargs)
    await runner._handle_message_with_agent(
        event,
        event.source,
        session_entry.session_key,
        1,
    )

    assert seen == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "handler_name", "choice"),
    [
        ("/approve", "_handle_approve_command", "once"),
        ("/deny operator declined", "_handle_deny_command", "deny"),
    ],
)
async def test_telegram_legacy_approval_slash_uses_distinct_response_context(
    monkeypatch,
    tmp_path,
    text,
    handler_name,
    choice,
):
    from tools import approval

    runner, _adapter, session_entry = _runner(monkeypatch, tmp_path)
    observed = []

    def resolve(session_key, selected, **_kwargs):
        observed.append(get_current_trusted_interaction())
        assert session_key == session_entry.session_key
        assert selected == choice
        return 1

    monkeypatch.setattr(approval, "has_blocking_approval", lambda key: True)
    monkeypatch.setattr(approval, "resolve_gateway_approval", resolve)
    event = _event(text=text)

    await getattr(runner, handler_name)(event)

    assert len(observed) == 1
    interaction = observed[0]
    assert interaction is not None
    assert interaction.surface == "gateway_approval"
    assert interaction.platform == "telegram"
    assert interaction.message_id == "message-1"
    assert interaction.session_id == "server-session-1"
    assert get_current_trusted_interaction() is None
