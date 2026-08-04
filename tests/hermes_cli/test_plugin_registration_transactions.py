"""Transactional plugin registration through the public discovery path."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from hermes_cli.plugins import PluginManager
from tools.registry import registry


_OVERRIDE_TOOL = "transaction_override_target"


def _schema(name: str, description: str = "transaction test tool") -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


def _manager_registration_state(manager: PluginManager) -> dict:
    """Identity-preserving view of every PluginManager registration surface."""
    return {
        "hooks": {name: tuple(callbacks) for name, callbacks in manager._hooks.items()},
        "middleware": {
            name: tuple(callbacks) for name, callbacks in manager._middleware.items()
        },
        "tool_names": frozenset(manager._plugin_tool_names),
        "platform_names": frozenset(manager._plugin_platform_names),
        "cli_commands": {
            name: dict(entry) for name, entry in manager._cli_commands.items()
        },
        "slash_commands": {
            name: dict(entry) for name, entry in manager._plugin_commands.items()
        },
        "skills": {
            name: dict(entry) for name, entry in manager._plugin_skills.items()
        },
        "aux_tasks": {
            name: {**entry, "defaults": dict(entry.get("defaults", {}))}
            for name, entry in manager._aux_tasks.items()
        },
        "slack_handlers": tuple(manager._slack_action_handlers),
        "context_engine": manager._context_engine,
    }


@pytest.fixture
def isolated_tool_registry():
    """Restore the process-global registry even while the RED tests fail."""
    with registry._lock:
        before = (
            dict(registry._tools),
            dict(registry._toolset_checks),
            dict(registry._toolset_aliases),
            dict(registry._plugin_override_policy),
            registry._generation,
        )
    try:
        yield
    finally:
        with registry._lock:
            (
                registry._tools,
                registry._toolset_checks,
                registry._toolset_aliases,
                registry._plugin_override_policy,
                old_generation,
            ) = before
            registry._generation = max(registry._generation, old_generation) + 1
        for module_name in list(sys.modules):
            if module_name.startswith("hermes_plugins.transaction_"):
                sys.modules.pop(module_name, None)


@pytest.fixture(autouse=True)
def no_bundled_plugins(tmp_path, monkeypatch):
    """Keep each public discovery sweep scoped to its temporary user plugins."""
    empty_bundled = tmp_path / "empty-bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))


def _write_config(home: Path, enabled: list[str], *, allow_override: str | None = None) -> None:
    plugins: dict = {"enabled": enabled}
    if allow_override is not None:
        plugins["entries"] = {allow_override: {"allow_tool_override": True}}
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": plugins}), encoding="utf-8"
    )


def _write_plugin(home: Path, name: str, source: str, *, submodule: str | None = None) -> Path:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(source, encoding="utf-8")
    (plugin_dir / "SKILL.md").write_text("# Transaction skill\n", encoding="utf-8")
    if submodule is not None:
        (plugin_dir / "child.py").write_text(submodule, encoding="utf-8")
    return plugin_dir


def _all_surfaces_source(*, raise_line: str = "") -> str:
    return f'''from pathlib import Path
from agent.context_engine import ContextEngine
from . import child

register_calls = 0

class TransactionEngine(ContextEngine):
    @property
    def name(self):
        return "transaction-engine"

    def update_from_response(self, usage):
        return None

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        return messages

def _tool(args, **kwargs):
    return "plugin-tool"

def _override(args, **kwargs):
    return "plugin-override"

def register(ctx):
    global register_calls
    register_calls += 1
    ctx.register_tool("transaction_new_tool", "transaction_plugin", {repr(_schema("transaction_new_tool"))}, _tool)
    ctx.register_tool({_OVERRIDE_TOOL!r}, "transaction_plugin", {repr(_schema(_OVERRIDE_TOOL))}, _override, override=True)
    ctx.register_hook("post_tool_call", lambda **kwargs: "plugin-hook")
    ctx.register_middleware("llm_request", lambda **kwargs: {{"plugin": True}})
    ctx.register_command("transaction-slash", lambda raw: "plugin-slash", "transaction slash")
    ctx.register_cli_command("transaction-cli", "transaction cli", lambda parser: None, lambda args: "plugin-cli")
    ctx.register_skill("transaction-skill", Path(__file__).parent / "SKILL.md", "transaction skill")
    ctx.register_auxiliary_task(key="transaction_aux", display_name="Transaction aux", description="transaction task")
    ctx.register_slack_action_handler("transaction_action", lambda **kwargs: "plugin-slack")
    ctx.register_context_engine(TransactionEngine())
    {raise_line or "return None"}
'''


def _register_builtin_override_target() -> object:
    def _builtin(args, **kwargs):
        return "built-in"

    registry.register(
        name=_OVERRIDE_TOOL,
        toolset="transaction_builtin",
        schema=_schema(_OVERRIDE_TOOL, "original built-in"),
        handler=_builtin,
    )
    entry = registry.get_entry(_OVERRIDE_TOOL)
    assert entry is not None
    return entry


def _concurrent_tool_source(*, fail_after_release: bool = False) -> str:
    final_line = (
        "raise RuntimeError('sensitive-concurrent-registration-detail')"
        if fail_after_release
        else "return None"
    )
    return f'''import threading

started = threading.Event()
release = threading.Event()

def _new_tool(args, **kwargs):
    return "committed-new-tool"

def _override(args, **kwargs):
    return "committed-override"

def register(ctx):
    ctx.register_tool("transaction_concurrent_new", "transaction_plugin", {repr(_schema("transaction_concurrent_new"))}, _new_tool)
    ctx.register_tool({_OVERRIDE_TOOL!r}, "transaction_plugin", {repr(_schema(_OVERRIDE_TOOL, "committed override"))}, _override, override=True)
    ctx.register_hook("post_tool_call", lambda **kwargs: "committed-hook")
    started.set()
    if not release.wait(timeout=5):
        raise RuntimeError("test release timeout")
    {final_line}
'''


def _start_discovery(manager: PluginManager) -> tuple[threading.Thread, list[BaseException]]:
    failures: list[BaseException] = []

    def _load() -> None:
        try:
            manager.discover_and_load()
        except BaseException as exc:  # pragma: no cover - asserted by callers
            failures.append(exc)

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    return thread, failures


def _wait_for_plugin_module(plugin_name: str):
    module_name = f"hermes_plugins.{plugin_name}"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "started"):
            assert module.started.wait(timeout=5)
            return module
        time.sleep(0.005)
    raise AssertionError(f"plugin module {module_name!r} did not start registration")


def _assert_live_tools_are_preload_state(original_entry: object) -> None:
    assert registry.get_entry("transaction_concurrent_new") is None
    assert registry.get_entry(_OVERRIDE_TOOL) is original_entry
    assert registry.get_entry(_OVERRIDE_TOOL).handler is original_entry.handler
    definitions = registry.get_definitions(
        {"transaction_concurrent_new", _OVERRIDE_TOOL}
    )
    assert [item["function"]["name"] for item in definitions] == [
        _OVERRIDE_TOOL
    ]
    assert definitions[0]["function"]["description"] == "original built-in"
    assert "Unknown tool" in registry.dispatch("transaction_concurrent_new", {})
    assert registry.dispatch(_OVERRIDE_TOOL, {}) == "built-in"


def test_tool_readers_observe_preload_state_until_atomic_successful_commit(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_concurrent_success"
    original_entry = _register_builtin_override_target()
    _write_plugin(home, plugin_name, _concurrent_tool_source())
    _write_config(home, [plugin_name], allow_override=plugin_name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    thread, failures = _start_discovery(manager)
    module = _wait_for_plugin_module(plugin_name)

    _assert_live_tools_are_preload_state(original_entry)
    tool_names, loaded_plugins = manager.get_tool_state_snapshot()
    assert tool_names == set()
    assert loaded_plugins == {}
    assert manager.has_hook("post_tool_call") is False

    module.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []

    new_entry = registry.get_entry("transaction_concurrent_new")
    override_entry = registry.get_entry(_OVERRIDE_TOOL)
    assert new_entry is not None
    assert override_entry is not None
    assert new_entry.handler is module._new_tool
    assert override_entry.handler is module._override
    assert registry.dispatch("transaction_concurrent_new", {}) == "committed-new-tool"
    assert registry.dispatch(_OVERRIDE_TOOL, {}) == "committed-override"
    assert {
        item["function"]["name"]
        for item in registry.get_definitions(
            {"transaction_concurrent_new", _OVERRIDE_TOOL}
        )
    } == {"transaction_concurrent_new", _OVERRIDE_TOOL}
    tool_names, loaded_plugins = manager.get_tool_state_snapshot()
    assert {"transaction_concurrent_new", _OVERRIDE_TOOL} <= tool_names
    assert loaded_plugins[plugin_name].tools_registered == [
        "transaction_concurrent_new",
        _OVERRIDE_TOOL,
    ]
    assert manager.invoke_hook("post_tool_call") == ["committed-hook"]


def test_failed_concurrent_registration_never_exposes_provisional_tools(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_concurrent_failure"
    original_entry = _register_builtin_override_target()
    _write_plugin(
        home,
        plugin_name,
        _concurrent_tool_source(fail_after_release=True),
    )
    _write_config(home, [plugin_name], allow_override=plugin_name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    thread, failures = _start_discovery(manager)
    module = _wait_for_plugin_module(plugin_name)
    observations: list[tuple[object, object]] = []
    stop_observer = threading.Event()

    def _observe() -> None:
        while not stop_observer.is_set():
            observations.append(
                (
                    registry.get_entry("transaction_concurrent_new"),
                    registry.get_entry(_OVERRIDE_TOOL),
                )
            )

    observer = threading.Thread(target=_observe, daemon=True)
    observer.start()
    try:
        _assert_live_tools_are_preload_state(original_entry)
        module.release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        stop_observer.set()
        observer.join(timeout=5)

    assert failures == []
    assert observations
    assert all(new is None for new, _override in observations)
    assert all(override is original_entry for _new, override in observations)
    _assert_live_tools_are_preload_state(original_entry)
    assert manager._plugin_tool_names == set()
    assert manager.has_hook("post_tool_call") is False
    loaded = manager._plugins[plugin_name]
    assert loaded.enabled is False
    assert loaded.module is None
    assert loaded.error == "RuntimeError: plugin registration failed"
    assert f"hermes_plugins.{plugin_name}" not in sys.modules


def test_commit_fails_closed_if_live_tool_registry_changed_since_snapshot(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_concurrent_conflict"
    original_entry = _register_builtin_override_target()
    _write_plugin(home, plugin_name, _concurrent_tool_source())
    _write_config(home, [plugin_name], allow_override=plugin_name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    thread, failures = _start_discovery(manager)
    module = _wait_for_plugin_module(plugin_name)

    def _concurrent_handler(args, **kwargs):
        return "concurrent-change"

    registry.register(
        name="transaction_concurrent_winner",
        toolset="transaction_concurrent_writer",
        schema=_schema("transaction_concurrent_winner"),
        handler=_concurrent_handler,
    )
    concurrent_entry = registry.get_entry("transaction_concurrent_winner")
    assert concurrent_entry is not None

    module.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []

    assert registry.get_entry("transaction_concurrent_winner") is concurrent_entry
    assert registry.dispatch("transaction_concurrent_winner", {}) == "concurrent-change"
    _assert_live_tools_are_preload_state(original_entry)
    assert manager._plugin_tool_names == set()
    assert manager.has_hook("post_tool_call") is False
    loaded = manager._plugins[plugin_name]
    assert loaded.enabled is False
    assert loaded.module is None
    assert loaded.error == "RuntimeError: plugin commit failed"
    assert f"hermes_plugins.{plugin_name}" not in sys.modules


def test_tool_and_manager_readers_block_inside_the_atomic_commit_section(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_commit_reader_blocking"
    _register_builtin_override_target()
    _write_plugin(home, plugin_name, _concurrent_tool_source())
    _write_config(home, [plugin_name], allow_override=plugin_name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    commit_started = threading.Event()
    release_commit = threading.Event()
    original_install = manager._install_owned_state_locked

    def _pause_manager_install(state) -> None:
        if plugin_name in state._plugins:
            commit_started.set()
            if not release_commit.wait(timeout=5):
                raise RuntimeError("test commit release timeout")
        original_install(state)

    monkeypatch.setattr(manager, "_install_owned_state_locked", _pause_manager_install)
    load_thread, failures = _start_discovery(manager)
    module = _wait_for_plugin_module(plugin_name)
    module.release.set()
    assert commit_started.wait(timeout=5)

    tool_result: list[object] = []
    manager_result: list[object] = []
    tool_done = threading.Event()
    manager_done = threading.Event()

    def _read_tool() -> None:
        tool_result.append(registry.get_entry("transaction_concurrent_new"))
        tool_done.set()

    def _read_manager() -> None:
        manager_result.append(manager.get_tool_state_snapshot())
        manager_done.set()

    tool_reader = threading.Thread(target=_read_tool, daemon=True)
    manager_reader = threading.Thread(target=_read_manager, daemon=True)
    tool_reader.start()
    manager_reader.start()
    assert not tool_done.wait(timeout=0.1)
    assert not manager_done.wait(timeout=0.1)

    release_commit.set()
    load_thread.join(timeout=5)
    tool_reader.join(timeout=5)
    manager_reader.join(timeout=5)
    assert not load_thread.is_alive()
    assert not tool_reader.is_alive()
    assert not manager_reader.is_alive()
    assert failures == []
    assert tool_result[0] is registry.get_entry("transaction_concurrent_new")
    tool_names, plugins = manager_result[0]
    assert "transaction_concurrent_new" in tool_names
    assert plugins[plugin_name].enabled is True


def test_context_retained_callback_targets_live_registries_after_commit(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_runtime_context"
    _write_plugin(
        home,
        plugin_name,
        '''saved_ctx = None

def _initial(args, **kwargs):
    return "initial"

def _late(args, **kwargs):
    return "late"

def register(ctx):
    global saved_ctx
    saved_ctx = ctx
    ctx.register_tool("transaction_runtime_initial", "transaction_runtime", {"name": "transaction_runtime_initial", "description": "initial", "parameters": {"type": "object", "properties": {}}}, _initial)

def register_late():
    saved_ctx.register_tool("transaction_runtime_late", "transaction_runtime", {"name": "transaction_runtime_late", "description": "late", "parameters": {"type": "object", "properties": {}}}, _late)
''',
    )
    _write_config(home, [plugin_name])
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    manager.discover_and_load()
    module = sys.modules[f"hermes_plugins.{plugin_name}"]
    assert registry.get_entry("transaction_runtime_late") is None

    module.register_late()

    assert registry.dispatch("transaction_runtime_late", {}) == "late"
    tool_names, _plugins = manager.get_tool_state_snapshot()
    assert "transaction_runtime_late" in tool_names


def test_failed_registration_rolls_back_every_supported_surface_and_modules(
    tmp_path, monkeypatch, caplog, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_failure"
    old_entry = _register_builtin_override_target()
    previous_same_name = type(sys)("hermes_plugins.transaction_failure")
    sys.modules["hermes_plugins.transaction_failure"] = previous_same_name
    _write_plugin(
        home,
        plugin_name,
        _all_surfaces_source(
            raise_line="raise RuntimeError('sensitive-registration-detail')"
        ),
        submodule="VALUE = 'created during failed import'\n",
    )
    _write_config(home, [plugin_name], allow_override=plugin_name)
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    before_manager = _manager_registration_state(manager)
    with registry._lock:
        before_tools = dict(registry._tools)
        before_checks = dict(registry._toolset_checks)
        before_aliases = dict(registry._toolset_aliases)
        before_policy = dict(registry._plugin_override_policy)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager.discover_and_load()

    assert _manager_registration_state(manager) == before_manager
    with registry._lock:
        assert registry._tools == before_tools
        assert registry._toolset_checks == before_checks
        assert registry._toolset_aliases == before_aliases
        assert registry._plugin_override_policy == before_policy
    assert registry.get_entry(_OVERRIDE_TOOL) is old_entry
    assert registry.get_entry("transaction_new_tool") is None
    assert manager.invoke_hook("post_tool_call") == []
    assert manager.invoke_middleware("llm_request") == []
    assert manager.find_plugin_skill(f"{plugin_name}:transaction-skill") is None
    assert manager.get_slack_action_handlers() == []
    assert "transaction-slash" not in manager._plugin_commands
    assert "transaction-cli" not in manager._cli_commands
    assert "transaction_aux" not in manager._aux_tasks
    assert manager._context_engine is None

    loaded = manager._plugins[plugin_name]
    assert loaded.enabled is False
    assert loaded.module is None
    assert loaded.error
    assert "registration failed" in loaded.error.lower()
    assert "sensitive-registration-detail" not in loaded.error
    assert "sensitive-registration-detail" not in caplog.text
    assert sys.modules["hermes_plugins.transaction_failure"] is previous_same_name
    assert "hermes_plugins.transaction_failure.child" not in sys.modules


def test_import_failure_removes_plugin_module_tree_and_records_safe_error(
    tmp_path, monkeypatch, caplog, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_import_failure"
    _write_plugin(
        home,
        plugin_name,
        '''from . import child
raise RuntimeError("sensitive-import-detail")
''',
        submodule="VALUE = 'temporary import child'\n",
    )
    _write_config(home, [plugin_name])
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager.discover_and_load()

    loaded = manager._plugins[plugin_name]
    assert loaded.enabled is False
    assert loaded.module is None
    assert loaded.error == "RuntimeError: plugin import failed"
    assert "sensitive-import-detail" not in caplog.text
    root = "hermes_plugins.transaction_import_failure"
    assert root not in sys.modules
    assert f"{root}.child" not in sys.modules


def test_success_commits_once_with_attribution_and_normal_discovery_is_idempotent(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_success"
    _register_builtin_override_target()
    _write_plugin(
        home,
        plugin_name,
        _all_surfaces_source(),
        submodule="VALUE = 'success'\n",
    )
    _write_config(home, [plugin_name], allow_override=plugin_name)
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    manager.discover_and_load()
    first_state = _manager_registration_state(manager)
    module = sys.modules["hermes_plugins.transaction_success"]
    manager.discover_and_load()

    assert module.register_calls == 1
    assert _manager_registration_state(manager) == first_state
    loaded = manager._plugins[plugin_name]
    assert loaded.enabled is True
    assert loaded.error is None
    assert loaded.module is module
    assert set(loaded.tools_registered) == {
        "transaction_new_tool",
        _OVERRIDE_TOOL,
    }
    assert loaded.hooks_registered == ["post_tool_call"]
    assert loaded.middleware_registered == ["llm_request"]
    assert loaded.commands_registered == ["transaction-slash"]
    assert len(manager._hooks["post_tool_call"]) == 1
    assert len(manager._middleware["llm_request"]) == 1
    assert len(manager._slack_action_handlers) == 1
    assert manager._cli_commands["transaction-cli"]["plugin"] == plugin_name
    assert manager._plugin_commands["transaction-slash"]["plugin"] == plugin_name
    assert manager._plugin_skills[f"{plugin_name}:transaction-skill"]["plugin"] == plugin_name
    assert manager._aux_tasks["transaction_aux"]["plugin"] == plugin_name
    assert manager._slack_action_handlers[0][2] == plugin_name
    assert manager._context_engine.name == "transaction-engine"
    assert registry.dispatch("transaction_new_tool", {}) == "plugin-tool"
    assert registry.dispatch(_OVERRIDE_TOOL, {}) == "plugin-override"


def test_manager_readers_do_not_observe_a_half_registration(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    plugin_name = "transaction_reader_isolation"
    _write_plugin(
        home,
        plugin_name,
        '''import threading

started = threading.Event()
release = threading.Event()

def register(ctx):
    ctx.register_hook("post_tool_call", lambda **kwargs: "committed-hook")
    started.set()
    if not release.wait(timeout=5):
        raise RuntimeError("test release timeout")
    ctx.register_middleware("llm_request", lambda **kwargs: {"committed": True})
''',
    )
    _write_config(home, [plugin_name])
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    failures: list[BaseException] = []

    def _load() -> None:
        try:
            manager.discover_and_load()
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    module_name = "hermes_plugins.transaction_reader_isolation"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "started"):
            break
        thread.join(timeout=0.01)
    module = sys.modules[module_name]
    assert module.started.wait(timeout=5)

    assert manager.has_hook("post_tool_call") is False
    assert manager.has_middleware("llm_request") is False
    assert manager.list_plugins() == []

    module.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    assert manager.has_hook("post_tool_call") is True
    assert manager.has_middleware("llm_request") is True
    assert manager._plugins[plugin_name].enabled is True


def test_failed_plugin_after_success_does_not_roll_back_success(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    success = "transaction_a_success"
    failure = "transaction_z_failure"
    _write_plugin(
        home,
        success,
        '''def _tool(args, **kwargs):
    return "success-still-present"

def register(ctx):
    ctx.register_tool("transaction_preserved_tool", "transaction_success", {"name": "transaction_preserved_tool", "description": "preserved", "parameters": {"type": "object", "properties": {}}}, _tool)
    ctx.register_hook("post_tool_call", lambda **kwargs: "preserved-hook")
''',
    )
    _write_plugin(
        home,
        failure,
        '''def register(ctx):
    ctx.register_hook("post_tool_call", lambda **kwargs: "leaked-hook")
    raise RuntimeError("failure after success")
''',
    )
    _write_config(home, [success, failure])
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    manager.discover_and_load()

    assert manager._plugins[success].enabled is True
    assert manager._plugins[failure].enabled is False
    assert registry.dispatch("transaction_preserved_tool", {}) == "success-still-present"
    assert manager.invoke_hook("post_tool_call") == ["preserved-hook"]
    assert len(manager._hooks["post_tool_call"]) == 1


def test_failed_registration_restores_platform_and_legacy_provider_registries(
    tmp_path, monkeypatch, isolated_tool_registry
):
    from agent.image_gen_provider import ImageGenProvider
    from agent.image_gen_registry import get_provider, register_provider
    from gateway.platform_registry import PlatformEntry, platform_registry

    class OriginalProvider(ImageGenProvider):
        @property
        def name(self):
            return "transaction-image"

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"owner": "original"}

    original_provider = OriginalProvider()
    original_platform = PlatformEntry(
        name="transaction-platform",
        label="Original platform",
        adapter_factory=lambda config: "original-platform",
        check_fn=lambda: True,
        source="plugin",
    )
    register_provider(original_provider)
    platform_registry.register(original_platform)
    try:
        home = tmp_path / "hermes-home"
        plugin_name = "transaction_external_failure"
        _write_plugin(
            home,
            plugin_name,
            '''from agent.image_gen_provider import ImageGenProvider

class ReplacementProvider(ImageGenProvider):
    @property
    def name(self):
        return "transaction-image"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {"owner": "replacement"}

def register(ctx):
    ctx.register_platform(
        name="transaction-platform",
        label="Replacement platform",
        adapter_factory=lambda config: "replacement-platform",
        check_fn=lambda: True,
    )
    ctx.register_image_gen_provider(ReplacementProvider())
    raise RuntimeError("external rollback")
''',
        )
        _write_config(home, [plugin_name])
        monkeypatch.setenv("HERMES_HOME", str(home))

        manager = PluginManager()
        manager.discover_and_load()

        assert manager._plugins[plugin_name].enabled is False
        assert get_provider("transaction-image") is original_provider
        assert platform_registry.get("transaction-platform") is original_platform
        assert "transaction-platform" not in manager._plugin_platform_names
    finally:
        from agent import image_gen_registry

        with image_gen_registry._lock:
            image_gen_registry._providers.pop("transaction-image", None)
        platform_registry.unregister("transaction-platform")


@pytest.mark.parametrize(
    ("exception_name", "exception_text"),
    [("KeyboardInterrupt", "stop-now"), ("SystemExit", "stop-now")],
)
def test_base_exception_rolls_back_then_propagates(
    tmp_path,
    monkeypatch,
    isolated_tool_registry,
    exception_name,
    exception_text,
):
    home = tmp_path / "hermes-home"
    plugin_name = f"transaction_{exception_name.lower()}"
    _write_plugin(
        home,
        plugin_name,
        f'''from . import child

def _tool(args, **kwargs):
    return "must-not-leak"

def register(ctx):
    ctx.register_tool("{plugin_name}_tool", "transaction_base_exception", {{"name": "{plugin_name}_tool", "description": "must not leak", "parameters": {{"type": "object", "properties": {{}}}}}}, _tool)
    ctx.register_hook("post_tool_call", lambda **kwargs: "must-not-leak")
    raise {exception_name}({exception_text!r})
''',
        submodule="VALUE = 'temporary'\n",
    )
    _write_config(home, [plugin_name])
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    with pytest.raises((KeyboardInterrupt, SystemExit), match=exception_text):
        manager.discover_and_load()

    assert manager.invoke_hook("post_tool_call") == []
    assert registry.get_entry(f"{plugin_name}_tool") is None
    assert f"{plugin_name}_tool" not in manager._plugin_tool_names
    root = f"hermes_plugins.{plugin_name}"
    assert root not in sys.modules
    assert f"{root}.child" not in sys.modules


def test_force_rediscovery_base_exception_restores_previous_coherent_state(
    tmp_path, monkeypatch, isolated_tool_registry
):
    home = tmp_path / "hermes-home"
    success = "transaction_a_coherent"
    interrupting = "transaction_z_interrupt"
    _write_plugin(
        home,
        success,
        '''from . import child

def _tool(args, **kwargs):
    return "coherent"

def register(ctx):
    ctx.register_tool("transaction_coherent_tool", "transaction_coherent", {"name": "transaction_coherent_tool", "description": "coherent", "parameters": {"type": "object", "properties": {}}}, _tool)
    ctx.register_hook("post_tool_call", lambda **kwargs: "coherent-hook")
''',
        submodule="VALUE = 'old coherent module'\n",
    )
    _write_config(home, [success])
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    before_manager = _manager_registration_state(manager)
    before_entry = registry.get_entry("transaction_coherent_tool")
    before_module = sys.modules["hermes_plugins.transaction_a_coherent"]
    before_child = sys.modules["hermes_plugins.transaction_a_coherent.child"]

    _write_plugin(
        home,
        interrupting,
        '''from . import child

def register(ctx):
    ctx.register_hook("post_tool_call", lambda **kwargs: "partial")
    raise KeyboardInterrupt("force sweep interrupted")
''',
        submodule="VALUE = 'temporary interrupt module'\n",
    )
    _write_config(home, [success, interrupting])

    with pytest.raises(KeyboardInterrupt, match="force sweep interrupted"):
        manager.discover_and_load(force=True)

    assert manager._discovered is True
    assert _manager_registration_state(manager) == before_manager
    assert registry.get_entry("transaction_coherent_tool") is before_entry
    assert registry.dispatch("transaction_coherent_tool", {}) == "coherent"
    assert manager.invoke_hook("post_tool_call") == ["coherent-hook"]
    assert sys.modules["hermes_plugins.transaction_a_coherent"] is before_module
    assert sys.modules["hermes_plugins.transaction_a_coherent.child"] is before_child
    assert "hermes_plugins.transaction_z_interrupt" not in sys.modules
    assert "hermes_plugins.transaction_z_interrupt.child" not in sys.modules
