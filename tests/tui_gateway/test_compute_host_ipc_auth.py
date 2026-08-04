from __future__ import annotations

import base64
import hmac
import io
import json
import os
import threading
from types import SimpleNamespace

import pytest

from agent._trusted_interaction_issuer import (
    _issue_trusted_interaction,
    _restore_trusted_interaction,
    _serialize_trusted_interaction,
)
from tui_gateway import host_supervisor as host_module
from tui_gateway.compute_host import ComputeHost


_IPC_MAC_KEY_ENV = "HERMES_COMPUTE_HOST_IPC_MAC_KEY"
_FIRST_KEY = bytes(range(32))
_SECOND_KEY = bytes(reversed(range(32)))
_TRUE_SUBPROCESS_POPEN = host_module.subprocess.Popen


class _CaptureStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        return None


class _FakeProc:
    _next_pid = 4100

    def __init__(self) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.stdin = _CaptureStdin()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        return 0 if self.returncode is None else self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


class _NoopThread:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def start(self) -> None:
        return None


class _ReadyEvent:
    def __init__(self, supervisor, hermes_home: str) -> None:
        self._supervisor = supervisor
        self._hermes_home = hermes_home

    def clear(self) -> None:
        return None

    def wait(self, timeout=None) -> bool:
        del timeout
        self._supervisor._hello = {
            "type": "hello",
            "boot_id": "public-boot-id",
            "build_sha": "test-build",
            "hermes_home": self._hermes_home,
        }
        return True


def _interaction_payload() -> dict[str, object]:
    interaction = _issue_trusted_interaction(
        surface="desktop",
        platform="local",
        actor_id="local-actor",
        channel_id="desktop-window",
        chat_id="runtime-1",
        thread_id=None,
        session_key="agent:main:local:runtime-1",
        session_id="runtime-1",
        message_id="local-message-1",
        received_at=1_754_000_000.25,
        profile_id="default",
    )
    payload = _serialize_trusted_interaction(interaction)
    assert payload is not None
    return payload


def _supervisor_with_fake_spawn(tmp_path, monkeypatch, keys):
    captured: list[dict] = []

    def fake_popen(_argv, **kwargs):
        proc = _FakeProc()
        captured.append({"env": dict(kwargs["env"]), "proc": proc})
        return proc

    key_iter = iter(keys)
    monkeypatch.setattr(
        host_module,
        "secrets",
        SimpleNamespace(token_bytes=lambda size: next(key_iter)),
        raising=False,
    )
    monkeypatch.setattr(host_module, "_Thread", _NoopThread)
    monkeypatch.setattr(host_module.subprocess, "Popen", fake_popen)
    supervisor = host_module.HostSupervisor(
        registry_path=tmp_path / "host.json",
        cwd=tmp_path,
        expected_build_sha="test-build",
        expected_hermes_home=str(tmp_path),
        autostart=False,
    )
    supervisor._hello_event = _ReadyEvent(supervisor, str(tmp_path))
    return supervisor, captured


def _thread_scoped_popen(
    *,
    target_thread_id: int,
    target_popen,
    fallback_popen,
):
    def _guarded_popen(*args, **kwargs):
        if threading.get_ident() == target_thread_id:
            return target_popen(*args, **kwargs)
        return fallback_popen(*args, **kwargs)

    return _guarded_popen


def _last_sent_frame(proc: _FakeProc) -> dict:
    assert proc.stdin.writes
    return json.loads(proc.stdin.writes[-1])


def _json_lines(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines() if line]


def _test_mac_payload(
    payload: dict[str, object],
    key: bytes,
) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        **payload,
        "ipc_mac": base64.urlsafe_b64encode(
            hmac.digest(key, canonical, "sha256")
        ).decode("ascii"),
    }


def test_supervisor_spawn_injects_scrubbed_private_key_without_metadata_leak(
    tmp_path,
    monkeypatch,
    caplog,
):
    caplog.set_level("DEBUG", logger=host_module.__name__)
    supervisor, captured = _supervisor_with_fake_spawn(
        tmp_path,
        monkeypatch,
        [_FIRST_KEY],
    )

    supervisor._spawn_locked(reason="test")

    child_env = captured[0]["env"]
    encoded_key = child_env[_IPC_MAC_KEY_ENV]
    assert base64.urlsafe_b64decode(encoded_key) == _FIRST_KEY
    assert len(base64.urlsafe_b64decode(encoded_key)) == 32
    assert _IPC_MAC_KEY_ENV not in supervisor.hello
    registry = json.loads(supervisor.registry_path.read_text(encoding="utf-8"))
    assert _IPC_MAC_KEY_ENV not in registry
    assert encoded_key not in json.dumps(supervisor.hello, sort_keys=True)
    assert encoded_key not in json.dumps(registry, sort_keys=True)
    assert encoded_key not in repr(supervisor)
    assert encoded_key not in caplog.text


def test_supervisor_submit_replaces_and_authenticates_provenance(
    tmp_path,
    monkeypatch,
):
    supervisor, captured = _supervisor_with_fake_spawn(
        tmp_path,
        monkeypatch,
        [_FIRST_KEY],
    )
    supervisor._spawn_locked(reason="test")
    provenance = _interaction_payload()
    provenance["ipc_mac"] = "writer-controlled-value"

    supervisor.submit_turn(
        {
            "type": "turn.start",
            "sid": "runtime-1",
            "request_id": "turn-1",
            "text": "hello",
            "trusted_interaction": provenance,
        }
    )

    sent = _last_sent_frame(captured[0]["proc"])
    signed = sent["trusted_interaction"]
    assert signed["ipc_mac"] != "writer-controlled-value"
    assert provenance["ipc_mac"] == "writer-controlled-value"
    restored = _restore_trusted_interaction(signed, mac_key=_FIRST_KEY)
    assert restored is not None
    assert restored.message_id == "local-message-1"


def test_server_builds_unsigned_internal_provenance_frame(tmp_path):
    from tui_gateway import server

    session = {
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": "agent:main:local:runtime-1",
        "cols": 80,
        "cwd": str(tmp_path),
        "source": "desktop",
    }

    frame = server._compute_host_turn_frame(
        "turn-1",
        "runtime-1",
        session,
        "hello",
        trusted_interaction=_issue_trusted_interaction(
            surface="desktop",
            platform="local",
            actor_id="local-actor",
            channel_id="desktop-window",
            chat_id="runtime-1",
            thread_id=None,
            session_key="agent:main:local:runtime-1",
            session_id="runtime-1",
            message_id="local-message-1",
            received_at=1_754_000_000.25,
            profile_id="default",
        ),
    )

    assert "trusted_interaction" in frame
    assert "evidence_hash" in frame["trusted_interaction"]
    assert "ipc_mac" not in frame["trusted_interaction"]


def test_supervisor_spawn_fails_before_popen_when_final_policy_strips_key(
    tmp_path,
    monkeypatch,
):
    import private_secret_policy

    supervisor, _captured = _supervisor_with_fake_spawn(
        tmp_path,
        monkeypatch,
        [_FIRST_KEY],
    )
    popen_called = False

    def strip_ipc_key(env):
        return {key: value for key, value in env.items() if key != _IPC_MAC_KEY_ENV}

    def forbidden_popen(*args, **kwargs):
        nonlocal popen_called
        del args, kwargs
        popen_called = True
        raise AssertionError("Popen must not run without the IPC MAC key")

    monkeypatch.setattr(
        private_secret_policy,
        "scrub_private_secret_env",
        strip_ipc_key,
    )
    monkeypatch.setattr(
        host_module.subprocess,
        "Popen",
        _thread_scoped_popen(
            target_thread_id=threading.get_ident(),
            target_popen=forbidden_popen,
            fallback_popen=_TRUE_SUBPROCESS_POPEN,
        ),
    )

    with pytest.raises(RuntimeError, match="IPC MAC"):
        supervisor._spawn_locked(reason="test")

    assert popen_called is False


def test_thread_scoped_popen_blocks_only_target_thread():
    calls: list[tuple[str, int]] = []
    worker_ready = threading.Barrier(2, timeout=5)
    worker_done = threading.Event()
    worker_result: dict[str, object] = {}
    sentinel = object()

    def target_popen(*args, **kwargs):
        del args, kwargs
        calls.append(("target", threading.get_ident()))
        raise AssertionError("target thread must be blocked")

    def fallback_popen(*args, **kwargs):
        del args, kwargs
        calls.append(("fallback", threading.get_ident()))
        return sentinel

    guarded = _thread_scoped_popen(
        target_thread_id=threading.get_ident(),
        target_popen=target_popen,
        fallback_popen=fallback_popen,
    )

    def worker():
        worker_ready.wait()
        worker_result["value"] = guarded("worker")
        worker_done.set()

    thread = threading.Thread(target=worker, name="popen-worker")
    thread.start()
    worker_ready.wait()
    assert worker_done.wait(timeout=5) is True
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert worker_result["value"] is sentinel

    with pytest.raises(AssertionError, match="target thread"):
        guarded("main")

    assert calls == [
        ("fallback", thread.ident),
        ("target", threading.get_ident()),
    ]


def test_supervisor_respawn_rotates_key_and_invalidates_old_frame(
    tmp_path,
    monkeypatch,
):
    supervisor, captured = _supervisor_with_fake_spawn(
        tmp_path,
        monkeypatch,
        [_FIRST_KEY, _SECOND_KEY],
    )
    supervisor._spawn_locked(reason="startup")
    supervisor.submit_turn(
        {
            "sid": "runtime-1",
            "request_id": "turn-old",
            "text": "old",
            "trusted_interaction": _interaction_payload(),
        }
    )
    old_signed = _last_sent_frame(captured[0]["proc"])["trusted_interaction"]

    supervisor._spawn_locked(reason="crash")

    assert base64.urlsafe_b64decode(
        captured[1]["env"][_IPC_MAC_KEY_ENV]
    ) == _SECOND_KEY
    assert _restore_trusted_interaction(old_signed, mac_key=_FIRST_KEY) is not None
    assert _restore_trusted_interaction(old_signed, mac_key=_SECOND_KEY) is None


def _run_compute_turn(monkeypatch, payload, *, env_value: str | None):
    from tui_gateway import server

    if env_value is None:
        monkeypatch.delenv(_IPC_MAC_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(_IPC_MAC_KEY_ENV, env_value)

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    sid = "ipc-auth-runtime"
    seen = []
    session = {
        "agent": SimpleNamespace(),
        "session_key": "agent:main:local:ipc-auth-runtime",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "last_active": 0.0,
        "inflight_turn": None,
        "attached_images": [],
        "cols": 80,
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_start_inflight_turn", lambda *_args: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_args: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})

    def run_prompt_submit(*_args, **kwargs):
        seen.append(kwargs.get("trusted_interaction"))

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt_submit)
    frame = {
        "type": "turn.start",
        "sid": sid,
        "request_id": "ipc-auth-turn",
        "text": "hello",
    }
    if payload is not None:
        frame["trusted_interaction"] = payload
    try:
        host._run_real_turn(frame)
        return host, seen, _json_lines(out)
    finally:
        server._sessions.pop(sid, None)
        host.close()


def test_compute_host_init_pops_and_decodes_private_key(monkeypatch):
    encoded = base64.urlsafe_b64encode(_FIRST_KEY).decode("ascii")
    monkeypatch.setenv(_IPC_MAC_KEY_ENV, encoded)

    host = ComputeHost(stdout=io.StringIO(), max_workers=1, heartbeat_secs=0)
    try:
        assert _IPC_MAC_KEY_ENV not in os.environ
        assert host._ipc_mac_key == _FIRST_KEY
        assert isinstance(host._ipc_mac_key, bytes)
    finally:
        host.close()


def test_compute_host_signed_frame_restores_trusted_interaction(monkeypatch):
    encoded = base64.urlsafe_b64encode(_FIRST_KEY).decode("ascii")
    signed = _test_mac_payload(_interaction_payload(), _FIRST_KEY)

    _host, seen, frames = _run_compute_turn(
        monkeypatch,
        signed,
        env_value=encoded,
    )

    assert len(seen) == 1
    assert seen[0] is not None
    assert seen[0].message_id == "local-message-1"
    assert any(frame["type"] == "turn.end" for frame in frames)


@pytest.mark.parametrize(
    "payload_factory",
    [
        lambda: _interaction_payload(),
        lambda: _test_mac_payload(_interaction_payload(), _SECOND_KEY),
    ],
    ids=["unsigned", "wrong-key"],
)
def test_compute_host_unauthenticated_frame_cannot_enter_governed_turn(
    monkeypatch,
    payload_factory,
):
    encoded = base64.urlsafe_b64encode(_FIRST_KEY).decode("ascii")

    _host, seen, frames = _run_compute_turn(
        monkeypatch,
        payload_factory(),
        env_value=encoded,
    )

    assert seen == [None]
    assert any(frame["type"] == "turn.end" for frame in frames)


@pytest.mark.parametrize(
    "env_value",
    [
        None,
        "malformed-not-base64%%%",
        base64.urlsafe_b64encode(b"x" * 31).decode("ascii"),
        base64.urlsafe_b64encode(b"x" * 33).decode("ascii"),
    ],
    ids=["missing", "malformed", "short", "long"],
)
def test_compute_host_without_capability_keeps_plain_turns_working(
    monkeypatch,
    env_value,
):
    _host, seen, frames = _run_compute_turn(
        monkeypatch,
        None,
        env_value=env_value,
    )

    assert _IPC_MAC_KEY_ENV not in os.environ
    assert seen == [None]
    assert any(frame["type"] == "turn.end" for frame in frames)
    assert not any(frame["type"] == "turn.error" for frame in frames)
