"""Final child-boundary scrubbing for plugin-declared private env names."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import private_secret_policy as policy


_PRIVATE = "PLUGIN_COMMAND_PRIVATE_SECRET"
_PRIVATE_LOWER = _PRIVATE.lower()
_SECRET_VALUE = "SENTINEL_CHILD_SECRET_VALUE"
_PUBLIC_VALUE = "SENTINEL_PUBLIC_VALUE"


@pytest.fixture(autouse=True)
def active_private_policy():
    from tools.env_passthrough import clear_env_passthrough

    snapshot = policy.snapshot_private_secret_policy()
    policy.replace_private_secret_names([_PRIVATE])
    try:
        yield
    finally:
        clear_env_passthrough()
        policy.restore_private_secret_policy(snapshot)


def test_local_helpers_scrub_after_extra_force_passthrough_and_inheritance(
    monkeypatch,
) -> None:
    from tools.env_passthrough import register_env_passthrough
    from tools.environments.local import (
        _make_run_env,
        _sanitize_subprocess_env,
        build_subprocess_env,
        hermes_subprocess_env,
    )

    monkeypatch.setenv(_PRIVATE, _SECRET_VALUE)
    register_env_passthrough([_PRIVATE])

    sanitized = _sanitize_subprocess_env(
        {_PRIVATE: _SECRET_VALUE, "PUBLIC_BASE": _PUBLIC_VALUE},
        {
            f"_HERMES_FORCE_{_PRIVATE}": _SECRET_VALUE,
            _PRIVATE: _SECRET_VALUE,
            "PUBLIC_EXTRA": _PUBLIC_VALUE,
        },
    )
    terminal = _make_run_env(
        {
            _PRIVATE: _SECRET_VALUE,
            f"_HERMES_FORCE_{_PRIVATE}": _SECRET_VALUE,
            "PUBLIC_TERMINAL": _PUBLIC_VALUE,
        }
    )
    inherited = hermes_subprocess_env(inherit_credentials=True)
    trusted = build_subprocess_env(
        {_PRIVATE: _SECRET_VALUE, "PUBLIC_TRUSTED": _PUBLIC_VALUE},
        scrub_secrets=False,
        inherit_profile_home=False,
        extra={_PRIVATE: _SECRET_VALUE, "PUBLIC_OVERLAY": _PUBLIC_VALUE},
    )

    for env in (sanitized, terminal, inherited, trusted):
        assert _PRIVATE not in env
    assert sanitized["PUBLIC_BASE"] == _PUBLIC_VALUE
    assert sanitized["PUBLIC_EXTRA"] == _PUBLIC_VALUE
    assert terminal["PUBLIC_TERMINAL"] == _PUBLIC_VALUE
    assert trusted["PUBLIC_TRUSTED"] == _PUBLIC_VALUE
    assert trusted["PUBLIC_OVERLAY"] == _PUBLIC_VALUE


def test_local_helper_scrubs_case_variants_on_windows(monkeypatch) -> None:
    from tools.environments.local import _sanitize_subprocess_env

    monkeypatch.setattr(policy, "_is_windows_platform", lambda: True)
    scrubbed = _sanitize_subprocess_env(
        {_PRIVATE_LOWER: _SECRET_VALUE},
        {f"_HERMES_FORCE_{_PRIVATE_LOWER}": _SECRET_VALUE},
    )

    assert _PRIVATE_LOWER not in scrubbed


def test_local_terminal_final_popen_scrubs_force_and_passthrough(
    tmp_path, monkeypatch
) -> None:
    import tools.environments.local as local
    from tools.env_passthrough import register_env_passthrough

    captured: dict = {}
    monkeypatch.setenv(_PRIVATE, _SECRET_VALUE)
    register_env_passthrough([_PRIVATE])
    environment = object.__new__(local.LocalEnvironment)
    environment.cwd = str(tmp_path)
    environment.env = {
        _PRIVATE: _SECRET_VALUE,
        f"_HERMES_FORCE_{_PRIVATE}": _SECRET_VALUE,
        "TRUSTED_TERMINAL_SETTING": _PUBLIC_VALUE,
    }

    def fake_popen(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return _CompletedProc()

    monkeypatch.setattr(local, "_find_bash", lambda: "sentinel-bash")
    monkeypatch.setattr(local.subprocess, "Popen", fake_popen)

    environment._run_bash("true")

    assert _PRIVATE not in captured["env"]
    assert captured["env"]["TRUSTED_TERMINAL_SETTING"] == _PUBLIC_VALUE


class _EmptyStream:
    encoding = "utf-8"

    def read(self, _size=-1):
        return ""


class _CompletedProc:
    pid = 41001
    returncode = 0
    stdout = _EmptyStream()
    stderr = _EmptyStream()

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return self.returncode


@pytest.mark.parametrize(
    ("module_name", "runner_name"),
    [
        ("tools.tts_tool", "_run_command_tts"),
        ("tools.transcription_tools", "_run_command_stt"),
    ],
)
def test_tts_stt_final_popen_scrubs_passthrough_and_delegated_overlay(
    monkeypatch, module_name, runner_name
) -> None:
    module = __import__(module_name, fromlist=[runner_name])
    runner = getattr(module, runner_name)
    captured: dict = {}
    monkeypatch.setenv(_PRIVATE, _SECRET_VALUE)
    monkeypatch.setenv("TRUSTED_COMMAND_SETTING", _PUBLIC_VALUE)

    def delegated_overlay(env):
        overlaid = dict(env)
        overlaid[_PRIVATE] = _SECRET_VALUE
        return overlaid

    def fake_popen(command, **kwargs):
        captured["env"] = kwargs["env"]
        return _CompletedProc()

    monkeypatch.setattr(
        "agent.delegation_context.delegated_child_subprocess_env",
        delegated_overlay,
    )
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    result = runner(
        "echo sentinel",
        timeout=1,
        env_passthrough=[_PRIVATE, "TRUSTED_COMMAND_SETTING"],
    )

    assert result.returncode == 0
    assert _PRIVATE not in captured["env"]
    assert captured["env"]["TRUSTED_COMMAND_SETTING"] == _PUBLIC_VALUE


@pytest.mark.parametrize(
    ("module_name", "exercise"),
    [
        (
            "tools.tts_tool",
            lambda module: module._ffmpeg_transcode_to_opus(
                "sentinel-input.wav",
                "sentinel-output.ogg",
            ),
        ),
        (
            "tools.transcription_tools",
            lambda module: module._transcode_audio_for_stt(
                "sentinel-input.ogg",
                "sentinel-work-dir",
            ),
        ),
    ],
)
def test_fixed_argv_ffmpeg_uses_final_private_policy_scrub(
    monkeypatch, module_name, exercise
) -> None:
    module = __import__(module_name, fromlist=["subprocess"])
    captured: dict = {}
    monkeypatch.setenv(_PRIVATE, _SECRET_VALUE)
    monkeypatch.setenv("TRUSTED_FFMPEG_SETTING", _PUBLIC_VALUE)
    if module_name == "tools.tts_tool":
        monkeypatch.setattr(module, "_has_ffmpeg", lambda: True)
    else:
        monkeypatch.setattr(module, "_find_ffmpeg_binary", lambda: "ffmpeg")

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exercise(module)

    assert captured["argv"][0] == "ffmpeg"
    assert _PRIVATE not in captured["env"]
    assert captured["env"]["TRUSTED_FFMPEG_SETTING"] == _PUBLIC_VALUE


@pytest.mark.parametrize(
    ("module_name", "exercise"),
    [
        (
            "tools.tts_tool",
            lambda module: module._ffmpeg_transcode_to_opus(
                "sentinel-input.wav",
                "sentinel-output.ogg",
            ),
        ),
        (
            "tools.transcription_tools",
            lambda module: module._transcode_audio_for_stt(
                "sentinel-input.ogg",
                "sentinel-work-dir",
            ),
        ),
    ],
)
def test_fixed_argv_ffmpeg_policy_failure_prevents_subprocess_run(
    monkeypatch, module_name, exercise
) -> None:
    module = __import__(module_name, fromlist=["subprocess"])
    calls = 0
    monkeypatch.setattr(policy, "_healthy", False)
    if module_name == "tools.tts_tool":
        monkeypatch.setattr(module, "_has_ffmpeg", lambda: True)
    else:
        monkeypatch.setattr(module, "_find_ffmpeg_binary", lambda: "ffmpeg")

    def forbidden_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("subprocess.run must not be reached")

    monkeypatch.setattr(module.subprocess, "run", forbidden_run)

    exercise(module)

    assert calls == 0


def test_compute_host_final_popen_scrubs_self_env_parent_and_preserves_provider(
    tmp_path, monkeypatch
) -> None:
    import tui_gateway.host_supervisor as host_module

    captured: dict = {}
    monkeypatch.setattr(policy, "_is_windows_platform", lambda: True)
    monkeypatch.setenv(_PRIVATE, _SECRET_VALUE)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _SECRET_VALUE)
    monkeypatch.setenv("OPENAI_API_KEY", "SENTINEL_PROVIDER_CREDENTIAL")

    class ReadyEvent:
        def __init__(self, supervisor):
            self.supervisor = supervisor

        def clear(self):
            return None

        def wait(self, timeout=None):
            self.supervisor._hello = {
                "type": "hello",
                "build_sha": "test-build",
                "hermes_home": str(tmp_path),
            }
            return True

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    class Proc(_CompletedProc):
        stdin = SimpleNamespace(write=lambda _value: None, flush=lambda: None)

    def fake_popen(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return Proc()

    monkeypatch.setattr(host_module, "_Thread", NoopThread)
    monkeypatch.setattr(host_module.subprocess, "Popen", fake_popen)
    supervisor = host_module.HostSupervisor(
        registry_path=tmp_path / "host.json",
        cwd=tmp_path,
        env={_PRIVATE_LOWER: _SECRET_VALUE, "PUBLIC_SELF_ENV": _PUBLIC_VALUE},
        expected_build_sha="test-build",
        expected_hermes_home=str(tmp_path),
        autostart=False,
    )
    supervisor._hello_event = ReadyEvent(supervisor)

    supervisor._spawn_locked(reason="test")

    child_env = captured["env"]
    assert _PRIVATE not in child_env
    assert _PRIVATE_LOWER not in child_env
    assert "TELEGRAM_BOT_TOKEN" not in child_env
    assert child_env["OPENAI_API_KEY"] == "SENTINEL_PROVIDER_CREDENTIAL"
    assert child_env["PUBLIC_SELF_ENV"] == _PUBLIC_VALUE


def test_codex_app_server_final_popen_scrubs_explicit_overlay(monkeypatch) -> None:
    import agent.transports.codex_app_server as codex_module

    captured: dict = {}

    class Proc(_CompletedProc):
        stdin = _EmptyStream()

    def fake_popen(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return Proc()

    monkeypatch.setattr(codex_module.subprocess, "Popen", fake_popen)

    codex_module.CodexAppServerClient(
        codex_bin="sentinel-codex",
        env={_PRIVATE: _SECRET_VALUE, "TRUSTED_CODEX_SETTING": _PUBLIC_VALUE},
    )

    assert _PRIVATE not in captured["env"]
    assert captured["env"]["TRUSTED_CODEX_SETTING"] == _PUBLIC_VALUE


def test_browser_passthrough_cannot_restore_private_name(monkeypatch) -> None:
    import tools.browser_tool as browser_tool

    snapshot = policy.snapshot_private_secret_policy()
    try:
        policy.replace_private_secret_names(["BROWSERBASE_API_KEY"])
        monkeypatch.setenv("BROWSERBASE_API_KEY", _SECRET_VALUE)

        child_env = browser_tool._build_browser_env()

        assert "BROWSERBASE_API_KEY" not in child_env
    finally:
        policy.restore_private_secret_policy(snapshot)


def test_mcp_final_sdk_parameters_scrub_secret_source_and_explicit_user_env(
    monkeypatch,
) -> None:
    import hermes_cli.env_loader as env_loader
    import tools.mcp_tool as mcp_tool

    captured: dict = {}
    monkeypatch.setattr(policy, "_is_windows_platform", lambda: True)
    monkeypatch.setenv(_PRIVATE, _SECRET_VALUE)
    monkeypatch.setattr(
        env_loader,
        "get_secret_source",
        lambda name: "private_secret_source" if name == _PRIVATE else None,
    )
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(
        mcp_tool,
        "StdioServerParameters",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(**kwargs),
        raising=False,
    )

    mock_stdio_cm = MagicMock()
    mock_stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
    mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.initialize = AsyncMock(return_value=SimpleNamespace(capabilities=None))
    session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mcp_tool, "stdio_client", lambda *a, **k: mock_stdio_cm, raising=False)
    monkeypatch.setattr(mcp_tool, "ClientSession", lambda *a, **k: mock_session_cm, raising=False)
    monkeypatch.setattr(mcp_tool, "_snapshot_child_pids", lambda: set())
    monkeypatch.setattr(mcp_tool, "_write_stderr_log_header", lambda _name: None)
    monkeypatch.setattr(mcp_tool, "_get_mcp_stderr_log", lambda: None)
    monkeypatch.setattr("tools.osv_check.check_package_for_malware", lambda *a: None)

    async def exercise() -> None:
        server = mcp_tool.MCPServerTask("private-env-test")
        await server.start(
            {
                "command": "sentinel-command",
                "args": [],
                "env": {
                    _PRIVATE_LOWER: _SECRET_VALUE,
                    "TRUSTED_MCP_SETTING": _PUBLIC_VALUE,
                },
            }
        )
        await server.shutdown()

    asyncio.run(exercise())

    assert _PRIVATE not in captured["env"]
    assert _PRIVATE_LOWER not in captured["env"]
    assert captured["env"]["TRUSTED_MCP_SETTING"] == _PUBLIC_VALUE


def test_policy_failure_prevents_popen_and_sdk_builder_without_leaking_value(
    monkeypatch, caplog
) -> None:
    import tools.mcp_tool as mcp_tool
    import tools.tts_tool as tts_tool

    popen_calls = 0
    sdk_calls = 0
    monkeypatch.setenv(_PRIVATE, _SECRET_VALUE)
    monkeypatch.setattr(policy, "_healthy", False)

    def forbidden_popen(*args, **kwargs):
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    def forbidden_sdk(**kwargs):
        nonlocal sdk_calls
        sdk_calls += 1
        raise AssertionError("SDK builder must not be reached")

    monkeypatch.setattr(tts_tool.subprocess, "Popen", forbidden_popen)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "StdioServerParameters", forbidden_sdk, raising=False)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(policy.PrivateSecretPolicyError) as tts_exc:
            tts_tool._run_command_tts(
                "echo sentinel", timeout=1, env_passthrough=[_PRIVATE]
            )
        with pytest.raises(policy.PrivateSecretPolicyError) as mcp_exc:
            asyncio.run(
                mcp_tool.MCPServerTask("policy-failure")._run_stdio(
                    {"command": "sentinel-command", "env": {_PRIVATE: _SECRET_VALUE}}
                )
            )

    assert popen_calls == 0
    assert sdk_calls == 0
    assert _SECRET_VALUE not in str(tts_exc.value)
    assert _SECRET_VALUE not in str(mcp_exc.value)
    assert _SECRET_VALUE not in caplog.text


def test_policy_failure_prevents_local_terminal_popen(tmp_path, monkeypatch) -> None:
    import tools.environments.local as local

    popen_calls = 0
    environment = object.__new__(local.LocalEnvironment)
    environment.cwd = str(tmp_path)
    environment.env = {_PRIVATE: _SECRET_VALUE}
    monkeypatch.setattr(policy, "_healthy", False)
    monkeypatch.setattr(local, "_find_bash", lambda: "sentinel-bash")

    def forbidden_popen(*args, **kwargs):
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(local.subprocess, "Popen", forbidden_popen)

    with pytest.raises(policy.PrivateSecretPolicyError):
        environment._run_bash("true")

    assert popen_calls == 0
