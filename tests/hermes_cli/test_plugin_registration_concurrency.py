"""Concurrency contracts for transactional plugin registration."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent.browser_provider import BrowserProvider
from agent.image_gen_provider import ImageGenProvider
from agent.secret_sources.base import FetchResult, SecretSource
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider
from agent.video_gen_provider import VideoGenProvider
from agent.web_search_provider import WebSearchProvider
from gateway.platform_registry import PlatformEntry, PlatformRegistry
from hermes_cli.dashboard_auth import DashboardAuthProvider
import hermes_cli.plugins as plugins_module
from hermes_cli.plugins import PluginManager
from tools.registry import registry


_SUPPORT_MODULE = "h1_transaction_test_support"
_TOOL_NAME = "h1_transaction_generation_tool"
_PLUGIN_NAME = "h1-transaction-concurrency"


class _InjectedBaseException(BaseException):
    pass


class _ImageProvider(ImageGenProvider):
    def __init__(self, name: str, marker: str) -> None:
        self._name = name
        self.marker = marker

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str, aspect_ratio: str = "landscape", **kwargs: Any) -> dict:
        return {"marker": self.marker}


class _VideoProvider(VideoGenProvider):
    def __init__(self, name: str, marker: str) -> None:
        self._name = name
        self.marker = marker

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        return {"marker": self.marker}


class _WebProvider(WebSearchProvider):
    def __init__(self, name: str, marker: str) -> None:
        self._name = name
        self.marker = marker

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> dict:
        return {"marker": self.marker}


class _BrowserProvider(BrowserProvider):
    def __init__(self, name: str, marker: str) -> None:
        self._name = name
        self.marker = marker

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def create_session(self, task_id: str) -> dict:
        return {"marker": self.marker}

    def close_session(self, session_id: str) -> bool:
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        return None


class _TTSProvider(TTSProvider):
    def __init__(self, name: str, marker: str) -> None:
        self._name = name
        self.marker = marker

    @property
    def name(self) -> str:
        return self._name

    def synthesize(self, text: str, output_path: str, **kwargs: Any) -> str:
        return output_path


class _STTProvider(TranscriptionProvider):
    def __init__(self, name: str, marker: str) -> None:
        self._name = name
        self.marker = marker

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, file_path: str, **kwargs: Any) -> dict:
        return {"success": True, "transcript": self.marker, "provider": self.name}


class _DashboardProvider(DashboardAuthProvider):
    name = "h1-dashboard-test"
    display_name = "H1 dashboard test"

    def __init__(self, name: str, marker: str) -> None:
        self.name = name
        self.display_name = marker
        self.marker = marker

    def start_login(self, *, redirect_uri: str):
        raise NotImplementedError

    def complete_login(self, **kwargs: Any):
        raise NotImplementedError

    def verify_session(self, *, access_token: str):
        return None

    def refresh_session(self, *, refresh_token: str):
        raise NotImplementedError

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


class _SecretProvider(SecretSource):
    shape = "mapped"

    def __init__(self, name: str, marker: str) -> None:
        self.name = name
        self.label = marker
        self.marker = marker

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        return FetchResult()


def _external_values(marker: str, namespace: str) -> dict[str, Any]:
    names = {
        surface: f"h1-{namespace}-{surface.replace('_', '-')}"
        for surface in (
            "image_gen",
            "video_gen",
            "web",
            "browser",
            "secret",
            "tts",
            "stt",
            "dashboard",
            "platform",
        )
    }
    names["secret"] = f"h1_{namespace.replace('-', '_')}_secret"
    return {
        "image_gen": _ImageProvider(names["image_gen"], marker),
        "video_gen": _VideoProvider(names["video_gen"], marker),
        "web": _WebProvider(names["web"], marker),
        "browser": _BrowserProvider(names["browser"], marker),
        "secret": _SecretProvider(names["secret"], marker),
        "tts": _TTSProvider(names["tts"], marker),
        "stt": _STTProvider(names["stt"], marker),
        "dashboard": _DashboardProvider(names["dashboard"], marker),
        "platform": PlatformEntry(
            name=names["platform"],
            label=marker,
            adapter_factory=lambda config, value=marker: value,
            check_fn=lambda: True,
            source="plugin",
            plugin_name=_PLUGIN_NAME,
        ),
    }


def _write_plugin(home: Path) -> None:
    plugin_dir = home / "plugins" / _PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": _PLUGIN_NAME, "version": "1.0"}),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        f'''import {_SUPPORT_MODULE} as support

def register(ctx):
    values = support.values
    ctx.register_image_gen_provider(values["image_gen"])
    ctx.register_video_gen_provider(values["video_gen"])
    ctx.register_web_search_provider(values["web"])
    ctx.register_browser_provider(values["browser"])
    ctx.register_secret_source(values["secret"])
    ctx.register_tts_provider(values["tts"])
    ctx.register_transcription_provider(values["stt"])
    ctx.register_dashboard_auth_provider(values["dashboard"])
    platform = values["platform"]
    ctx.register_platform(
        name=platform.name,
        label=platform.label,
        adapter_factory=platform.adapter_factory,
        check_fn=platform.check_fn,
    )
    if support.register_generation_state:
        ctx.register_tool(
            {_TOOL_NAME!r},
            "h1_transaction",
            {{
                "name": {_TOOL_NAME!r},
                "description": support.marker,
                "parameters": {{"type": "object", "properties": {{}}}},
            }},
            support.tool_handler,
        )
        ctx.register_hook("post_tool_call", support.hook)
    support.started.set()
    if not support.release.wait(timeout=5):
        raise RuntimeError("test release timeout")
    if support.fail:
        raise RuntimeError("planned registration failure")
''',
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [_PLUGIN_NAME]}}),
        encoding="utf-8",
    )


def _support(
    values: dict[str, Any],
    marker: str,
    *,
    released: bool = False,
    fail: bool = False,
    register_generation_state: bool = False,
) -> types.ModuleType:
    module = types.ModuleType(_SUPPORT_MODULE)
    module.values = values
    module.marker = marker
    module.started = threading.Event()
    module.release = threading.Event()
    if released:
        module.release.set()
    module.fail = fail
    module.register_generation_state = register_generation_state
    module.tool_handler = lambda args, **kwargs: marker
    module.hook = lambda **kwargs: marker
    sys.modules[_SUPPORT_MODULE] = module
    return module


def _start_discovery(
    manager: PluginManager,
    *,
    force: bool = False,
) -> tuple[threading.Thread, threading.Event, list[BaseException]]:
    done = threading.Event()
    failures: list[BaseException] = []

    def _run() -> None:
        try:
            manager.discover_and_load(force=force)
        except BaseException as exc:  # pragma: no cover - asserted by callers
            failures.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, done, failures


def _join(thread: threading.Thread, done: threading.Event) -> None:
    assert done.wait(timeout=5), "plugin discovery did not finish"
    thread.join(timeout=1)
    assert not thread.is_alive()


def _external_readers(platform_registry: PlatformRegistry) -> dict[str, Any]:
    from agent import (
        browser_registry,
        image_gen_registry,
        transcription_registry,
        tts_registry,
        video_gen_registry,
        web_search_registry,
    )
    from agent.secret_sources import registry as secret_registry
    from hermes_cli.dashboard_auth import get_provider as get_dashboard_provider

    return {
        "image_gen": image_gen_registry.get_provider,
        "video_gen": video_gen_registry.get_provider,
        "web": web_search_registry.get_provider,
        "browser": browser_registry.get_provider,
        "secret": secret_registry.get_source,
        "tts": tts_registry.get_provider,
        "stt": transcription_registry.get_provider,
        "dashboard": get_dashboard_provider,
        "platform": platform_registry.get,
    }


def _read_external_generation(
    values: dict[str, Any],
    platform_registry: PlatformRegistry,
) -> dict[str, Any]:
    readers = _external_readers(platform_registry)
    return {
        surface: readers[surface](value.name)
        for surface, value in values.items()
    }


def _surface_generations(manager: PluginManager) -> dict[str, int]:
    transactions = plugins_module._external_registry_transactions()
    return {
        "manager": manager._generation,
        "manager_context": manager._live_context_generation,
        "tool": registry._generation,
        **{
            surface: transaction.take_snapshot()._generation
            for surface, transaction in transactions.items()
            if hasattr(transaction.take_snapshot(), "_generation")
        },
        "secret": transactions["secret"].take_snapshot()._mapping._generation,
    }


def _assert_lock_acquirable_from_other_thread(label: str, lock: Any) -> None:
    acquired = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def acquire_lock() -> None:
        try:
            lock.acquire()
            acquired.set()
            assert release.wait(timeout=5), f"{label} release gate timed out"
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            if acquired.is_set():
                lock.release()

    thread = threading.Thread(
        target=acquire_lock,
        name=f"h1-assert-lock-free-{label}",
        daemon=True,
    )
    thread.start()
    assert acquired.wait(timeout=5), f"{label} lock was leaked"
    release.set()
    thread.join(timeout=1)
    assert not thread.is_alive(), f"{label} lock holder did not exit"
    assert failures == []


def _assert_all_registration_locks_acquirable_from_other_threads(
    manager: PluginManager,
    *,
    context_locks: list[Any] | None = None,
) -> None:
    _assert_lock_acquirable_from_other_thread("manager", manager._lock)
    _assert_lock_acquirable_from_other_thread("tool", registry._lock)
    for surface, transaction in plugins_module._external_registry_transactions().items():
        _assert_lock_acquirable_from_other_thread(surface, transaction.lock)
    for index, lock in enumerate(context_locks or []):
        _assert_lock_acquirable_from_other_thread(f"context[{index}]", lock)


def _register_external_direct(
    surface: str,
    value: Any,
    platform_registry: PlatformRegistry,
) -> None:
    from agent import (
        browser_registry,
        image_gen_registry,
        transcription_registry,
        tts_registry,
        video_gen_registry,
        web_search_registry,
    )
    from agent.secret_sources import registry as secret_registry
    from hermes_cli.dashboard_auth import register_provider as register_dashboard

    registrations = {
        "image_gen": image_gen_registry.register_provider,
        "video_gen": video_gen_registry.register_provider,
        "web": web_search_registry.register_provider,
        "browser": browser_registry.register_provider,
        "secret": secret_registry.register_source,
        "tts": tts_registry.register_provider,
        "stt": transcription_registry.register_provider,
        "dashboard": register_dashboard,
        "platform": platform_registry.register,
    }
    registrations[surface](value)


@pytest.fixture(autouse=True)
def isolated_registries(tmp_path, monkeypatch):
    from agent import (
        browser_registry,
        image_gen_registry,
        transcription_registry,
        tts_registry,
        video_gen_registry,
        web_search_registry,
    )
    from agent.secret_sources import registry as secret_registry
    from gateway import platform_registry as platform_module
    from hermes_cli.dashboard_auth import clear_providers

    provider_modules = (
        image_gen_registry,
        video_gen_registry,
        web_search_registry,
        browser_registry,
        tts_registry,
        transcription_registry,
    )
    for module in provider_modules:
        module._reset_for_tests()
    clear_providers()
    secret_registry._reset_registry_for_tests()
    monkeypatch.setattr(secret_registry, "_ensure_builtin_sources", lambda: None)
    isolated_platform_registry = PlatformRegistry()
    monkeypatch.setattr(
        platform_module,
        "platform_registry",
        isolated_platform_registry,
    )

    with registry._lock:
        tool_state = (
            dict(registry._tools),
            dict(registry._toolset_checks),
            dict(registry._toolset_aliases),
            dict(registry._plugin_override_policy),
            registry._generation,
        )

    bundled = tmp_path / "empty-bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    yield isolated_platform_registry

    with registry._lock:
        (
            registry._tools,
            registry._toolset_checks,
            registry._toolset_aliases,
            registry._plugin_override_policy,
            old_generation,
        ) = tool_state
        registry._generation = max(registry._generation, old_generation) + 1
    sys.modules.pop(_SUPPORT_MODULE, None)
    for name in list(sys.modules):
        if name.startswith("hermes_plugins.h1_transaction_concurrency"):
            sys.modules.pop(name, None)


def test_provider_and_platform_readers_never_observe_provisional_registration(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    values = _external_values("candidate", "visibility-success")
    support = _support(values, "candidate")
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    thread, done, failures = _start_discovery(manager)
    assert support.started.wait(timeout=5)

    blocked_observation = _read_external_generation(values, isolated_registries)
    support.release.set()
    _join(thread, done)
    assert failures == []
    assert all(observed is None for observed in blocked_observation.values())
    assert _read_external_generation(values, isolated_registries) == values


def test_failed_provider_and_platform_registration_never_becomes_visible(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    values = _external_values("candidate", "visibility-failure")
    support = _support(values, "candidate", fail=True)
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    thread, done, failures = _start_discovery(manager)
    assert support.started.wait(timeout=5)

    blocked_observation = _read_external_generation(values, isolated_registries)
    support.release.set()
    _join(thread, done)

    assert failures == []
    assert all(observed is None for observed in blocked_observation.values())
    assert all(
        observed is None
        for observed in _read_external_generation(values, isolated_registries).values()
    )
    assert manager._plugins[_PLUGIN_NAME].enabled is False


def test_force_reload_readers_observe_old_generation_until_atomic_swap(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "force")
    support = _support(
        old_values,
        "generation-a",
        released=True,
        register_generation_state=True,
    )
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_tool = registry.get_entry(_TOOL_NAME)
    assert old_tool is not None

    new_values = _external_values("generation-b", "force")
    support = _support(
        new_values,
        "generation-b",
        register_generation_state=True,
    )
    thread, done, failures = _start_discovery(manager, force=True)
    assert support.started.wait(timeout=5)

    blocked_hook = manager.invoke_hook("post_tool_call")
    blocked_tool = registry.get_entry(_TOOL_NAME)
    blocked_external = _read_external_generation(old_values, isolated_registries)

    support.release.set()
    _join(thread, done)
    assert failures == []
    assert blocked_hook == ["generation-a"]
    assert blocked_tool is old_tool
    assert blocked_external == old_values
    assert manager.invoke_hook("post_tool_call") == ["generation-b"]
    assert registry.get_entry(_TOOL_NAME).handler is support.tool_handler
    assert _read_external_generation(new_values, isolated_registries) == new_values


def test_force_reload_failure_preserves_old_generation_without_empty_window(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "force-failure")
    _support(
        old_values,
        "generation-a",
        released=True,
        register_generation_state=True,
    )
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_tool = registry.get_entry(_TOOL_NAME)
    assert old_tool is not None

    new_values = _external_values("generation-b", "force-failure")
    support = _support(
        new_values,
        "generation-b",
        fail=True,
        register_generation_state=True,
    )
    thread, done, failures = _start_discovery(manager, force=True)
    assert support.started.wait(timeout=5)

    blocked_hook = manager.invoke_hook("post_tool_call")
    blocked_tool = registry.get_entry(_TOOL_NAME)
    blocked_external = _read_external_generation(old_values, isolated_registries)

    support.release.set()
    _join(thread, done)
    assert failures == []
    assert blocked_hook == ["generation-a"]
    assert blocked_tool is old_tool
    assert blocked_external == old_values
    assert manager.invoke_hook("post_tool_call") == ["generation-a"]
    assert registry.get_entry(_TOOL_NAME) is old_tool
    observed = _read_external_generation(old_values, isolated_registries)
    assert observed == old_values
    assert {
        getattr(value, "marker", getattr(value, "label", None))
        for value in observed.values()
    } == {"generation-a"}


def test_force_reload_baseexception_after_tool_install_rolls_back_all_surfaces(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "tool-install-interrupt")
    _support(old_values, "generation-a", released=True, register_generation_state=True)
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_tool = registry.get_entry(_TOOL_NAME)
    old_generations = _surface_generations(manager)

    new_values = _external_values("generation-b", "tool-install-interrupt")
    _support(new_values, "generation-b", released=True, register_generation_state=True)
    original_install = registry._install_prepared_transaction_locked

    def install_then_interrupt(snapshot, prepared):
        original_install(snapshot, prepared)
        raise _InjectedBaseException("injected after tool install")

    monkeypatch.setattr(registry, "_install_prepared_transaction_locked", install_then_interrupt)

    with pytest.raises(_InjectedBaseException, match="injected after tool install"):
        manager.discover_and_load(force=True)

    assert _surface_generations(manager) == old_generations
    assert manager._live_context_revoking == set()
    assert manager.invoke_hook("post_tool_call") == ["generation-a"]
    assert registry.get_entry(_TOOL_NAME) is old_tool
    assert _read_external_generation(old_values, isolated_registries) == old_values
    assert _read_external_generation(new_values, isolated_registries) == old_values


def test_force_reload_baseexception_after_external_install_rolls_back_all_surfaces(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "external-install-interrupt")
    _support(old_values, "generation-a", released=True, register_generation_state=True)
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_tool = registry.get_entry(_TOOL_NAME)
    old_generations = _surface_generations(manager)

    new_values = _external_values("generation-b", "external-install-interrupt")
    _support(new_values, "generation-b", released=True, register_generation_state=True)
    original_transactions = plugins_module._external_registry_transactions

    class _InterruptingSurface:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name: str):
            return getattr(self._wrapped, name)

        def install_prepared_locked(self, snapshot, prepared) -> None:
            self._wrapped.install_prepared_locked(snapshot, prepared)
            raise _InjectedBaseException("injected after external install")

    def transactions_with_interrupt():
        transactions = original_transactions()
        transactions["platform"] = _InterruptingSurface(transactions["platform"])
        return transactions

    monkeypatch.setattr(
        plugins_module,
        "_external_registry_transactions",
        transactions_with_interrupt,
    )

    with pytest.raises(_InjectedBaseException, match="injected after external install"):
        manager.discover_and_load(force=True)

    assert _surface_generations(manager) == old_generations
    assert manager._live_context_revoking == set()
    assert manager.invoke_hook("post_tool_call") == ["generation-a"]
    assert registry.get_entry(_TOOL_NAME) is old_tool
    assert _read_external_generation(old_values, isolated_registries) == old_values
    assert _read_external_generation(new_values, isolated_registries) == old_values


def test_force_reload_baseexception_after_activation_keeps_published_generation(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "activation-interrupt")
    _support(old_values, "generation-a", released=True, register_generation_state=True)
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    new_values = _external_values("generation-b", "activation-interrupt")
    _support(new_values, "generation-b", released=True, register_generation_state=True)
    original_activate = plugins_module._PluginPublicationCapability._activate
    rollback_attempted = threading.Event()
    original_restore = registry._restore_transaction_snapshot_exact_locked

    def activate_then_interrupt(self):
        original_activate(self)
        raise _InjectedBaseException("injected after publication activation")

    def observe_rollback(snapshot):
        rollback_attempted.set()
        original_restore(snapshot)

    monkeypatch.setattr(
        plugins_module._PluginPublicationCapability,
        "_activate",
        activate_then_interrupt,
    )
    monkeypatch.setattr(
        registry,
        "_restore_transaction_snapshot_exact_locked",
        observe_rollback,
    )

    with pytest.raises(
        _InjectedBaseException,
        match="injected after publication activation",
    ):
        manager.discover_and_load(force=True)

    assert not rollback_attempted.is_set()
    assert manager._live_context_revoking == set()
    assert manager.invoke_hook("post_tool_call") == ["generation-b"]
    assert registry.get_entry(_TOOL_NAME).handler({}) == "generation-b"
    assert _read_external_generation(new_values, isolated_registries) == new_values


def test_force_reload_mid_cleanup_release_failure_preserves_primary_and_cancels_revocation(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "cleanup-release")
    support = _support(
        old_values,
        "generation-a",
        released=True,
        register_generation_state=True,
    )
    support.saved_contexts = []
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_contexts.append(ctx)\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    new_values = _external_values("generation-b", "cleanup-release")
    support = _support(
        new_values,
        "generation-b",
        released=True,
        register_generation_state=True,
    )
    support.saved_contexts = []
    primary = _InjectedBaseException("injected after all publication locks")
    release_failure = _InjectedBaseException("injected mid cleanup release failure")
    original_tool_validate = registry._validate_prepared_transaction_locked
    original_tool_lock = registry._lock
    fail_release = threading.Event()
    revocation_observed = threading.Event()

    class _FailingReleaseLock:
        def acquire(self, *args, **kwargs):
            return original_tool_lock.acquire(*args, **kwargs)

        def release(self):
            original_tool_lock.release()
            if fail_release.is_set():
                raise release_failure

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()

    def validate_then_interrupt(snapshot, prepared) -> None:
        original_tool_validate(snapshot, prepared)
        assert manager._live_context_revoking == {manager._live_context_generation}
        revocation_observed.set()
        fail_release.set()
        raise primary

    monkeypatch.setattr(registry, "_validate_prepared_transaction_locked", validate_then_interrupt)
    monkeypatch.setattr(registry, "_lock", _FailingReleaseLock())

    with pytest.raises(_InjectedBaseException) as exc_info:
        manager.discover_and_load(force=True)

    assert exc_info.value is primary
    assert revocation_observed.is_set()
    assert any("cleanup" in note.lower() for note in primary.__notes__)
    assert any("mid cleanup release failure" in note for note in primary.__notes__)
    assert manager._live_context_revoking == set()
    monkeypatch.setattr(registry, "_lock", original_tool_lock)
    _assert_all_registration_locks_acquirable_from_other_threads(
        manager,
        context_locks=[
            manager._context_registration_lock,
            *(context._registration_lock for context in support.saved_contexts),
        ],
    )
    assert manager.invoke_hook("post_tool_call") == ["generation-a"]
    assert registry.get_entry(_TOOL_NAME).handler({}) == "generation-a"
    assert _read_external_generation(old_values, isolated_registries) == old_values


def test_registration_transaction_constructor_attempts_all_lock_cleanup(monkeypatch):
    manager = PluginManager()
    original_manager_lock = manager._lock
    original_tool_lock = registry._lock
    external_transactions = plugins_module._external_registry_transactions()
    external_lock_slots: dict[str, tuple[Any, str, Any]] = {}
    external_surfaces = list(external_transactions)
    for surface, transaction in external_transactions.items():
        if surface == "platform":
            owner = transaction
        elif surface == "secret":
            from agent.secret_sources import registry as secret_registry

            owner = secret_registry._registry_state
        else:
            owner = transaction._state
        external_lock_slots[surface] = (owner, "_lock", owner._lock)
    first_failure = _InjectedBaseException("injected constructor release failure 0")
    second_failure = _InjectedBaseException("injected constructor release failure 1")
    first_cleanup_index = 2 + external_surfaces.index("web")
    later_cleanup_index = 2 + external_surfaces.index("stt")
    failures = {first_cleanup_index: first_failure, later_cleanup_index: second_failure}

    class _ReleaseWrapper:
        def __init__(self, lock, index: int) -> None:
            self._lock = lock
            self._index = index
            self._release_count = 0

        def acquire(self, *args, **kwargs):
            return self._lock.acquire(*args, **kwargs)

        def release(self):
            self._lock.release()
            self._release_count += 1
            if self._index in failures and self._release_count > 1:
                raise failures[self._index]

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()

    monkeypatch.setattr(manager, "_lock", _ReleaseWrapper(original_manager_lock, 0))
    monkeypatch.setattr(registry, "_lock", _ReleaseWrapper(original_tool_lock, 1))
    for index, (surface, (owner, attr, original_lock)) in enumerate(
        external_lock_slots.items(),
        start=2,
    ):
        monkeypatch.setattr(owner, attr, _ReleaseWrapper(original_lock, index))

    with pytest.raises(_InjectedBaseException) as exc_info:
        plugins_module._RegistrationTransaction(manager)

    monkeypatch.setattr(manager, "_lock", original_manager_lock)
    monkeypatch.setattr(registry, "_lock", original_tool_lock)
    for owner, attr, original_lock in external_lock_slots.values():
        monkeypatch.setattr(owner, attr, original_lock)

    assert exc_info.value is first_failure
    assert first_failure.__notes__ == [
        "Additional cleanup failure: constructor lock[3]: "
        "_InjectedBaseException: injected constructor release failure 1",
        "Plugin registration cleanup failed; attempted all cleanup actions. "
        "First failure at constructor lock[0].",
    ]
    _assert_all_registration_locks_acquirable_from_other_threads(manager)


def test_force_reload_candidate_callback_worker_registration_does_not_deadlock(
    tmp_path, monkeypatch
):
    home = tmp_path / "hermes-home"
    values = _external_values("generation-a", "worker-callback")
    support = _support(values, "generation-a", released=True)
    support.worker_started = threading.Event()
    support.worker_done = threading.Event()
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        f'''import threading
import {_SUPPORT_MODULE} as support

def register(ctx):
    def worker():
        support.worker_started.set()
        ctx.register_hook("h1_worker_callback_hook", lambda **kwargs: support.marker)
        support.worker_done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    if not support.worker_started.wait(timeout=5):
        raise RuntimeError("worker did not start")
    if not support.worker_done.wait(timeout=5):
        raise RuntimeError("worker registration deadlocked")
    thread.join(timeout=1)
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    manager.discover_and_load(force=True)

    assert support.worker_done.is_set()
    assert manager.invoke_hook("h1_worker_callback_hook") == ["generation-a"]


def test_force_reload_removes_registrations_absent_from_new_generation(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "force-remove")
    _support(
        old_values,
        "generation-a",
        released=True,
        register_generation_state=True,
    )
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    assert registry.get_entry(_TOOL_NAME) is not None
    assert _read_external_generation(old_values, isolated_registries) == old_values
    module_namespace = f"hermes_plugins.{_PLUGIN_NAME.replace('-', '_')}"
    with registry._lock:
        assert module_namespace in registry._plugin_override_policy

    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": []}}),
        encoding="utf-8",
    )
    manager.discover_and_load(force=True)

    assert manager.invoke_hook("post_tool_call") == []
    assert registry.get_entry(_TOOL_NAME) is None
    assert all(
        observed is None
        for observed in _read_external_generation(
            old_values,
            isolated_registries,
        ).values()
    )
    assert manager._plugin_tool_names == set()
    assert all(not names for names in manager._plugin_external_names.values())
    assert manager._plugins[_PLUGIN_NAME].enabled is False
    with registry._lock:
        assert module_namespace not in registry._plugin_override_policy


def test_force_reload_preserves_same_key_external_replacement_after_manager_snapshot(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    namespace = "same-key-replacement"
    old_values = _external_values("generation-a", namespace)
    _support(old_values, "generation-a", released=True)
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    replacement_values = _external_values("direct-replacement", namespace)
    new_values = _external_values("generation-b", namespace)
    _support(new_values, "generation-b", released=True)
    replacement_written = threading.Event()
    snapshot_entered = threading.Event()
    original_transactions = plugins_module._external_registry_transactions

    class _ImageSnapshotBarrier:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self._started = False

        def __getattr__(self, name: str):
            return getattr(self._wrapped, name)

        def take_snapshot(self):
            if not self._started:
                self._started = True
                snapshot_entered.set()

                def write_replacement() -> None:
                    _register_external_direct(
                        "image_gen",
                        replacement_values["image_gen"],
                        isolated_registries,
                    )
                    replacement_written.set()

                writer = threading.Thread(target=write_replacement, daemon=True)
                writer.start()
                replacement_written.wait(timeout=0.2)
            return self._wrapped.take_snapshot()

    def transactions_with_barrier():
        transactions = original_transactions()
        transactions["image_gen"] = _ImageSnapshotBarrier(transactions["image_gen"])
        return transactions

    monkeypatch.setattr(
        plugins_module,
        "_external_registry_transactions",
        transactions_with_barrier,
    )

    manager.discover_and_load(force=True)

    assert snapshot_entered.is_set()
    assert replacement_written.is_set()
    observed = _read_external_generation(replacement_values, isolated_registries)
    assert observed["image_gen"] is replacement_values["image_gen"]


def test_force_reload_preserves_same_key_tool_replacement_after_manager_snapshot(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "same-key-tool")
    _support(
        old_values,
        "generation-a",
        released=True,
        register_generation_state=True,
    )
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    new_values = _external_values("generation-b", "same-key-tool")
    _support(
        new_values,
        "generation-b",
        released=True,
        register_generation_state=True,
    )
    replacement_written = threading.Event()
    snapshot_calls = 0
    original_snapshot = registry._take_transaction_snapshot

    def replacement_handler(args, **kwargs):
        return "direct-replacement"

    def write_replacement() -> None:
        registry.register(
            name=_TOOL_NAME,
            toolset="h1_transaction",
            schema={
                "name": _TOOL_NAME,
                "description": "direct replacement",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=replacement_handler,
        )
        replacement_written.set()

    def snapshot_with_barrier():
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            writer = threading.Thread(target=write_replacement, daemon=True)
            writer.start()
            assert not replacement_written.wait(timeout=0.2)
        else:
            assert replacement_written.wait(timeout=5)
        return original_snapshot()

    monkeypatch.setattr(
        registry,
        "_take_transaction_snapshot",
        snapshot_with_barrier,
    )

    manager.discover_and_load(force=True)

    assert snapshot_calls >= 1
    assert replacement_written.wait(timeout=5)
    assert registry.get_entry(_TOOL_NAME).handler is replacement_handler


@pytest.mark.parametrize(
    "conflict_surface",
    [
        "platform",
        "browser",
        "dashboard",
        "image_gen",
        "secret",
        "stt",
        "tts",
        "video_gen",
        "web",
    ],
)
def test_external_registry_commit_conflict_preserves_concurrent_writer(
    tmp_path, monkeypatch, isolated_registries, conflict_surface
):
    home = tmp_path / "hermes-home"
    candidate_values = _external_values("candidate", "conflict")
    support = _support(candidate_values, "candidate")
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    thread, done, failures = _start_discovery(manager)
    assert support.started.wait(timeout=5)

    concurrent_values = _external_values(
        "concurrent-writer",
        "conflict-writer",
    )
    _register_external_direct(
        conflict_surface,
        concurrent_values[conflict_surface],
        isolated_registries,
    )

    support.release.set()
    _join(thread, done)
    assert failures == []
    observed_concurrent = _read_external_generation(
        concurrent_values,
        isolated_registries,
    )
    assert observed_concurrent[conflict_surface] is concurrent_values[conflict_surface]
    assert all(
        observed is None
        for observed in _read_external_generation(
            candidate_values,
            isolated_registries,
        ).values()
    )
    assert manager._plugins[_PLUGIN_NAME].enabled is False
    assert manager._plugins[_PLUGIN_NAME].error == "RuntimeError: plugin commit failed"


def test_context_retained_after_commit_targets_live_external_registries(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    values = _external_values("initial", "retained-context")
    support = _support(values, "initial", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    late = _ImageProvider("h1-retained-context-late", "late")
    support.saved_context.register_image_gen_provider(late)

    from agent.image_gen_registry import get_provider

    assert get_provider(late.name) is late


def test_failed_initial_publication_restores_candidate_context_targets(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    values = _external_values("candidate", "context-rollback")
    support = _support(values, "candidate", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    original_install = manager._install_owned_state_locked

    def interrupt_after_manager_install(state):
        original_install(state)
        raise KeyboardInterrupt("injected after manager install")

    monkeypatch.setattr(
        manager,
        "_install_owned_state_locked",
        interrupt_after_manager_install,
    )

    with pytest.raises(KeyboardInterrupt, match="injected after manager install"):
        manager.discover_and_load()

    candidate_context = support.saved_context
    late = _ImageProvider("h1-failed-context-retarget-image", "late")
    late_tool = "h1_failed_context_retarget_tool"
    late_hook = lambda **kwargs: "late"

    with pytest.raises(RuntimeError, match="transaction view is frozen"):
        candidate_context.register_image_gen_provider(late)
    with pytest.raises(RuntimeError, match="transaction view is frozen"):
        candidate_context.register_tool(
            late_tool,
            "h1_transaction",
            {
                "name": late_tool,
                "description": "late",
                "parameters": {"type": "object", "properties": {}},
            },
            lambda args, **kwargs: "late",
        )
    with pytest.raises(RuntimeError, match="registration view is frozen"):
        candidate_context.register_hook("post_tool_call", late_hook)

    from agent.image_gen_registry import get_provider

    assert get_provider(late.name) is None
    assert registry.get_entry(late_tool) is None
    assert late_hook not in manager._hooks.get("post_tool_call", [])


def test_failed_binding_rollback_leaves_retained_candidate_without_live_authority(
    tmp_path, monkeypatch, isolated_registries, caplog
):
    home = tmp_path / "hermes-home"
    values = _external_values("candidate", "binding-rollback-failure")
    support = _support(values, "candidate", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    primary = _InjectedBaseException("injected after tentative live binding")
    rollback_failure = _InjectedBaseException("injected binding rollback failure")
    original_install = manager._install_owned_state_locked

    def install_then_interrupt(state) -> None:
        original_install(state)
        raise primary

    monkeypatch.setattr(manager, "_install_owned_state_locked", install_then_interrupt)
    original_setattr = plugins_module.PluginContext.__setattr__
    binding_assignments = 0

    def fail_binding_rollback(self, name, value) -> None:
        nonlocal binding_assignments
        if self is support.saved_context and name == "_registration_binding":
            binding_assignments += 1
            if binding_assignments == 2:
                raise rollback_failure
        original_setattr(self, name, value)

    monkeypatch.setattr(plugins_module.PluginContext, "__setattr__", fail_binding_rollback)

    with pytest.raises(_InjectedBaseException) as raised:
        manager.discover_and_load()

    assert raised.value is primary
    assert binding_assignments == 2
    assert any(
        "Plugin publication rollback failed for context[0]" in record.message
        for record in caplog.records
    )
    assert any("rollback failed" in note.lower() for note in primary.__notes__)

    candidate_context = support.saved_context
    escaped_tool = "h1_failed_rollback_escape_tool"
    escaped_platform = "h1-failed-rollback-escape-platform"
    escaped_hook = "h1_failed_rollback_escape_hook"
    escaped_command = "h1-failed-rollback-escape-command"
    mutations = [
        lambda: candidate_context.register_tool(
            escaped_tool,
            "h1_transaction",
            {
                "name": escaped_tool,
                "description": "must not escape",
                "parameters": {"type": "object", "properties": {}},
            },
            lambda args, **kwargs: "escaped",
        ),
        lambda: candidate_context.register_platform(
            name=escaped_platform,
            label="Must not escape",
            adapter_factory=lambda config: "escaped",
            check_fn=lambda: True,
        ),
        lambda: candidate_context.register_hook(escaped_hook, lambda **kwargs: "escaped"),
        lambda: candidate_context.register_command(
            escaped_command,
            lambda raw_args: "escaped",
        ),
    ]
    escaped_surfaces = []
    for surface, mutate in zip(
        ("tool", "platform", "hook", "command"),
        mutations,
        strict=True,
    ):
        try:
            mutate()
        except RuntimeError as exc:
            assert "not published" in str(exc)
        else:  # pragma: no cover - deterministic RED assertion on the base
            escaped_surfaces.append(surface)

    assert escaped_surfaces == []

    binding = candidate_context._registration_binding
    assert binding.manager is manager
    assert binding.managed_generation == manager._live_context_generation
    assert binding.publication_capability is not None
    assert binding.publication_capability._published is False

    assert registry.get_entry(escaped_tool) is None
    assert isolated_registries.get(escaped_platform) is None
    assert escaped_hook not in manager._hooks
    assert escaped_command not in manager._plugin_commands


def _mutate_retained_candidate(context, surface: str, name: str) -> None:
    if surface == "tool":
        context.register_tool(
            name,
            "h1_transaction",
            {
                "name": name,
                "description": "concurrent retained candidate",
                "parameters": {"type": "object", "properties": {}},
            },
            lambda args, **kwargs: "concurrent",
        )
        return
    if surface == "hook":
        context.register_hook(name, lambda *args, **kwargs: "concurrent")
        return
    context.register_platform(
        name=name,
        label="Concurrent retained candidate",
        adapter_factory=lambda config: "concurrent",
        check_fn=lambda: True,
    )


@pytest.mark.parametrize("surface", ["tool", "platform", "hook"])
def test_initial_publication_hides_tentative_binding_from_retained_candidate_mutation(
    tmp_path, monkeypatch, isolated_registries, surface
):
    home = tmp_path / "hermes-home"
    values = _external_values("candidate", f"atomic-success-{surface}")
    support = _support(values, "candidate", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    publication_paused = threading.Event()
    release_publication = threading.Event()
    original_install = manager._install_owned_state_locked

    def pause_before_manager_install(state) -> None:
        publication_paused.set()
        assert release_publication.wait(timeout=5)
        original_install(state)

    monkeypatch.setattr(manager, "_install_owned_state_locked", pause_before_manager_install)
    discovery_thread, discovery_done, discovery_failures = _start_discovery(manager)
    assert publication_paused.wait(timeout=5)
    context = support.saved_context
    binding_captured = threading.Event()
    mutation_started = threading.Event()
    mutation_done = threading.Event()
    mutation_failures: list[BaseException] = []
    original_acquire = context._acquire_live_lease

    def observe_binding_capture(*args, **kwargs):
        binding_captured.set()
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(context, "_acquire_live_lease", observe_binding_capture)
    name = f"h1_atomic_success_{surface}"

    def mutate() -> None:
        mutation_started.set()
        try:
            _mutate_retained_candidate(context, surface, name)
        except BaseException as exc:  # pragma: no cover - asserted below
            mutation_failures.append(exc)
        finally:
            mutation_done.set()

    mutation_thread = threading.Thread(target=mutate, daemon=True)
    mutation_thread.start()
    assert mutation_started.wait(timeout=5)
    captured_before_publication = binding_captured.wait(timeout=0.2)
    release_publication.set()
    _join(discovery_thread, discovery_done)
    assert mutation_done.wait(timeout=5)
    mutation_thread.join(timeout=1)

    assert not captured_before_publication
    assert discovery_failures == []
    assert mutation_failures == []
    assert not mutation_thread.is_alive()
    if surface == "tool":
        assert registry.get_entry(name) is not None
        assert name in manager._plugin_tool_names
    elif surface == "platform":
        assert isolated_registries.get(name) is not None
        assert name in manager._plugin_external_names["platform"]
    else:
        assert len(manager._hooks.get(name, [])) == 1


@pytest.mark.parametrize("surface", ["tool", "platform", "hook"])
def test_failed_initial_publication_restores_binding_before_retained_candidate_mutation(
    tmp_path, monkeypatch, isolated_registries, surface
):
    home = tmp_path / "hermes-home"
    values = _external_values("candidate", f"atomic-rollback-{surface}")
    support = _support(values, "candidate", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    rollback_paused = threading.Event()
    release_rollback = threading.Event()
    original_install = manager._install_owned_state_locked

    def install_then_pause_and_interrupt(state) -> None:
        original_install(state)
        rollback_paused.set()
        assert release_rollback.wait(timeout=5)
        raise KeyboardInterrupt("injected after manager install")

    monkeypatch.setattr(
        manager,
        "_install_owned_state_locked",
        install_then_pause_and_interrupt,
    )
    discovery_thread, discovery_done, discovery_failures = _start_discovery(manager)
    assert rollback_paused.wait(timeout=5)
    context = support.saved_context
    binding_captured = threading.Event()
    mutation_started = threading.Event()
    mutation_done = threading.Event()
    mutation_failures: list[BaseException] = []
    original_acquire = context._acquire_live_lease

    def observe_binding_capture(*args, **kwargs):
        binding_captured.set()
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(context, "_acquire_live_lease", observe_binding_capture)
    name = f"h1_atomic_rollback_{surface}"

    def mutate() -> None:
        mutation_started.set()
        try:
            _mutate_retained_candidate(context, surface, name)
        except BaseException as exc:  # pragma: no cover - asserted below
            mutation_failures.append(exc)
        finally:
            mutation_done.set()

    mutation_thread = threading.Thread(target=mutate, daemon=True)
    mutation_thread.start()
    assert mutation_started.wait(timeout=5)
    captured_before_rollback = binding_captured.wait(timeout=0.2)
    release_rollback.set()
    _join(discovery_thread, discovery_done)
    assert mutation_done.wait(timeout=5)
    mutation_thread.join(timeout=1)

    assert not captured_before_rollback
    assert len(discovery_failures) == 1
    assert isinstance(discovery_failures[0], KeyboardInterrupt)
    assert len(mutation_failures) == 1
    assert "frozen" in str(mutation_failures[0])
    assert not mutation_thread.is_alive()
    if surface == "tool":
        assert registry.get_entry(name) is None
        assert name not in manager._plugin_tool_names
    elif surface == "platform":
        assert isolated_registries.get(name) is None
        assert name not in manager._plugin_external_names["platform"]
    else:
        assert name not in manager._hooks


def test_commit_does_not_deadlock_reverse_nested_candidate_context_registration(
    isolated_registries,
):
    manager = PluginManager()
    transaction = plugins_module._RegistrationTransaction(manager)
    manifest_a = plugins_module.PluginManifest(name="h1-lock-a")
    manifest_b = plugins_module.PluginManifest(name="h1-lock-b")
    context_kwargs = {
        "registration_manager": transaction.manager_view,
        "registration_registry": transaction.tool_view,
        "registration_external_registries": transaction.context_targets(),
    }
    context_a = plugins_module.PluginContext(manifest_a, manager, **context_kwargs)
    context_b = plugins_module.PluginContext(manifest_b, manager, **context_kwargs)
    transaction.contexts.extend((context_a, context_b))

    commit_attempted_context_lock = threading.Event()
    commit_ident: list[int] = []

    class _ObservedRLock:
        def __init__(self, lock) -> None:
            self._lock = lock

        def acquire(self, *args, **kwargs):
            if commit_ident and threading.get_ident() == commit_ident[0]:
                commit_attempted_context_lock.set()
            return self._lock.acquire(*args, **kwargs)

        def release(self) -> None:
            self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.release()

    wrappers = {}
    for context in (context_a, context_b):
        lock = context._registration_lock
        wrapper = wrappers.setdefault(id(lock), _ObservedRLock(lock))
        context._registration_lock = wrapper
    if hasattr(manager, "_context_registration_lock"):
        lock = manager._context_registration_lock
        manager._context_registration_lock = wrappers.setdefault(
            id(lock), _ObservedRLock(lock)
        )

    callback_entered = threading.Event()
    registration_done = threading.Event()
    registration_failures: list[BaseException] = []

    class _ReverseNestedSchema(dict):
        def get(self, key, default=None):
            if key == "description":
                callback_entered.set()
                assert commit_attempted_context_lock.wait(timeout=5)
                context_a.register_hook(
                    "h1_reverse_nested_hook",
                    lambda **kwargs: "must not publish",
                )
            return super().get(key, default)

    def register_from_context_b() -> None:
        try:
            context_b.register_tool(
                "h1_reverse_nested_tool",
                "h1_transaction",
                _ReverseNestedSchema(
                    {
                        "name": "h1_reverse_nested_tool",
                        "description": "must not publish",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ),
                lambda args, **kwargs: "must not publish",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            registration_failures.append(exc)
        finally:
            registration_done.set()

    registration_thread = threading.Thread(
        target=register_from_context_b,
        daemon=True,
    )
    registration_thread.start()
    assert callback_entered.wait(timeout=5)

    commit_done = threading.Event()
    commit_failures: list[BaseException] = []

    def commit() -> None:
        commit_ident.append(threading.get_ident())
        try:
            transaction.commit()
        except BaseException as exc:  # pragma: no cover - asserted below
            commit_failures.append(exc)
        finally:
            commit_done.set()

    commit_thread = threading.Thread(target=commit, daemon=True)
    commit_thread.start()

    assert commit_attempted_context_lock.wait(timeout=5)
    assert registration_done.wait(timeout=5), "reverse nested registration deadlocked"
    assert commit_done.wait(timeout=5), "commit deadlocked on reverse context order"
    registration_thread.join(timeout=1)
    commit_thread.join(timeout=1)
    assert not registration_thread.is_alive()
    assert not commit_thread.is_alive()
    assert len(commit_failures) == 1
    assert isinstance(commit_failures[0], plugins_module._ForceSweepAbort)
    assert len(registration_failures) == 1
    assert "frozen" in str(registration_failures[0])
    assert registry.get_entry("h1_reverse_nested_tool") is None
    assert "h1_reverse_nested_hook" not in manager._hooks


def test_retained_context_live_registration_does_not_validate_under_live_locks(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    values = _external_values("initial", "lock-validation")
    support = _support(values, "initial", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    from agent import image_gen_registry

    observations: list[tuple[str, bool, bool, bool]] = []

    def observe(label: str) -> None:
        observations.append(
            (
                label,
                manager._lock._is_owned(),
                registry._lock._is_owned(),
                image_gen_registry._lock._is_owned(),
            )
        )

    class _ObservedImageProvider(_ImageProvider):
        @property
        def name(self) -> str:
            observe("provider.name")
            return super().name

    class _ObservedSchema(dict):
        def get(self, key, default=None):
            if key == "description":
                observe("schema.get")
            return super().get(key, default)

    provider = _ObservedImageProvider("h1-lock-validation-late-image", "late")
    support.saved_context.register_image_gen_provider(provider)
    support.saved_context.register_tool(
        "h1_lock_validation_late_tool",
        "h1_transaction",
        _ObservedSchema(
            {
                "name": "h1_lock_validation_late_tool",
                "description": "late",
                "parameters": {"type": "object", "properties": {}},
            }
        ),
        lambda args, **kwargs: "late",
    )

    assert observations
    assert all(
        not any((manager_locked, tool_locked, external_locked))
        for _, manager_locked, tool_locked, external_locked in observations
    ), observations


def test_retained_context_reentrant_force_reload_during_validation_fails_closed(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    values = _external_values("initial", "reentrant-force")
    support = _support(values, "initial", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    provider_name = "h1-reentrant-force-validation-image"

    class _ReentrantImageProvider(_ImageProvider):
        @property
        def name(self) -> str:
            manager.discover_and_load(force=True)
            return super().name

    failures: list[BaseException] = []
    done = threading.Event()

    def register_from_retained_context() -> None:
        try:
            support.saved_context.register_image_gen_provider(
                _ReentrantImageProvider(provider_name, "late")
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=register_from_retained_context, daemon=True)
    thread.start()

    assert done.wait(timeout=5), "reentrant force reload self-deadlocked"
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "force reload cannot run from a live plugin registration" in str(failures[0])

    from agent.image_gen_registry import get_provider

    assert get_provider(provider_name) is None


def test_force_reload_fails_closed_instead_of_waiting_on_live_validation(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "lease-cycle")
    support = _support(old_values, "generation-a", released=True)
    support.saved_context = None
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_context = ctx\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_context = support.saved_context

    validation_started = threading.Event()
    force_done = threading.Event()
    registration_done = threading.Event()
    registration_failures: list[BaseException] = []

    class _WaitingImageProvider(_ImageProvider):
        @property
        def name(self) -> str:
            validation_started.set()
            if not force_done.wait(timeout=5):
                raise RuntimeError("force reload did not finish")
            return super().name

    provider_name = "h1-live-validation-waits-for-force"

    def register_provider() -> None:
        try:
            old_context.register_image_gen_provider(
                _WaitingImageProvider(provider_name, "late")
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            registration_failures.append(exc)
        finally:
            registration_done.set()

    registration_thread = threading.Thread(target=register_provider, daemon=True)
    registration_thread.start()
    assert validation_started.wait(timeout=5)

    new_values = _external_values("generation-b", "lease-cycle")
    support = _support(new_values, "generation-b", released=True)
    force_failures: list[BaseException] = []

    def force_reload() -> None:
        try:
            manager.discover_and_load(force=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            force_failures.append(exc)
        finally:
            force_done.set()

    force_thread = threading.Thread(target=force_reload, daemon=True)
    force_thread.start()

    assert force_done.wait(timeout=5), "force reload waited on live validation"
    assert registration_done.wait(timeout=5), "live validation did not resume"
    force_thread.join(timeout=1)
    registration_thread.join(timeout=1)
    assert not force_thread.is_alive()
    assert not registration_thread.is_alive()
    assert force_failures == []
    assert registration_failures == []
    assert manager._live_context_generation == 0
    assert manager._live_context_revoking == set()

    from agent.image_gen_registry import get_provider

    assert get_provider(provider_name) is not None


def test_successful_force_reload_revokes_prior_retained_context_before_mutation(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "revoked-context")
    support = _support(
        old_values,
        "generation-a",
        released=True,
        register_generation_state=True,
    )
    support.saved_contexts = []
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_contexts.append(ctx)\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_context = support.saved_contexts[-1]

    new_values = _external_values(
        "generation-b",
        "revoked-context",
    )
    support = _support(
        new_values,
        "generation-b",
        released=True,
        register_generation_state=True,
    )
    support.saved_contexts = []
    manager.discover_and_load(force=True)
    new_context = support.saved_contexts[-1]
    assert manager._live_context_revoking == set()

    stale_external = _ImageProvider("h1-revoked-context-late-image", "stale")
    stale_tool_name = "h1_revoked_context_late_tool"
    stale_platform_name = "h1-revoked-context-late-platform"
    stale_hook = lambda **kwargs: "stale"

    with pytest.raises(RuntimeError, match="no longer live"):
        old_context.register_image_gen_provider(stale_external)
    with pytest.raises(RuntimeError, match="no longer live"):
        old_context.register_tool(
            stale_tool_name,
            "h1_transaction",
            {
                "name": stale_tool_name,
                "description": "stale",
                "parameters": {"type": "object", "properties": {}},
            },
            lambda args, **kwargs: "stale",
        )
    with pytest.raises(RuntimeError, match="no longer live"):
        old_context.register_hook("post_tool_call", stale_hook)
    with pytest.raises(RuntimeError, match="no longer live"):
        old_context.register_platform(
            name=stale_platform_name,
            label="Stale",
            adapter_factory=lambda config: "stale",
            check_fn=lambda: True,
        )

    from agent.image_gen_registry import get_provider

    assert get_provider(stale_external.name) is None
    assert registry.get_entry(stale_tool_name) is None
    assert isolated_registries.get(stale_platform_name) is None
    assert stale_hook not in manager._hooks.get("post_tool_call", [])

    live_external = _ImageProvider("h1-revoked-context-live-image", "live")
    live_tool_name = "h1_revoked_context_live_tool"
    live_hook = lambda **kwargs: "live"
    new_context.register_image_gen_provider(live_external)
    new_context.register_tool(
        live_tool_name,
        "h1_transaction",
        {
            "name": live_tool_name,
            "description": "live",
            "parameters": {"type": "object", "properties": {}},
        },
        lambda args, **kwargs: "live",
    )
    new_context.register_hook("post_tool_call", live_hook)

    assert get_provider(live_external.name) is live_external
    assert registry.get_entry(live_tool_name) is not None
    assert live_hook in manager._hooks["post_tool_call"]


def test_failed_force_reload_keeps_prior_retained_context_live(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "failed-force-context")
    support = _support(old_values, "generation-a", released=True)
    support.saved_contexts = []
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_contexts.append(ctx)\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_context = support.saved_contexts[-1]

    new_values = _external_values("generation-b", "failed-force-context")
    support = _support(new_values, "generation-b", released=True, fail=True)
    support.saved_contexts = []
    manager.discover_and_load(force=True)

    retained_external = _ImageProvider("h1-failed-force-live-image", "retained")
    retained_hook = lambda **kwargs: "retained"
    old_context.register_image_gen_provider(retained_external)
    old_context.register_hook("post_tool_call", retained_hook)

    from agent.image_gen_registry import get_provider

    assert support.saved_contexts
    assert get_provider(retained_external.name) is retained_external
    assert retained_hook in manager._hooks["post_tool_call"]


def test_force_reload_interrupt_after_revocation_keeps_old_context_live(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "revocation-interrupt")
    support = _support(old_values, "generation-a", released=True)
    support.saved_contexts = []
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_contexts.append(ctx)\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_context = support.saved_contexts[-1]

    support = _support(
        _external_values("generation-b", "revocation-interrupt"),
        "generation-b",
        released=True,
    )
    support.saved_contexts = []
    original_begin = manager._begin_live_context_revocation

    def interrupt_after_begin(generation: int) -> None:
        original_begin(generation)
        raise KeyboardInterrupt("injected after revocation")

    monkeypatch.setattr(
        manager,
        "_begin_live_context_revocation",
        interrupt_after_begin,
    )

    with pytest.raises(KeyboardInterrupt, match="injected after revocation"):
        manager.discover_and_load(force=True)

    assert manager._live_context_generation == 0
    assert manager._live_context_revoking == set()
    retained_tool = "h1_revocation_interrupt_tool"
    retained_platform = "h1-revocation-interrupt-platform"
    old_context.register_tool(
        retained_tool,
        "h1_transaction",
        {
            "name": retained_tool,
            "description": "retained",
            "parameters": {"type": "object", "properties": {}},
        },
        lambda args, **kwargs: "retained",
    )
    old_context.register_platform(
        name=retained_platform,
        label="Retained",
        adapter_factory=lambda config: "retained",
        check_fn=lambda: True,
    )
    assert registry.get_entry(retained_tool) is not None
    assert isolated_registries.get(retained_platform) is not None


def test_registration_transaction_constructor_lock_interrupt_releases_prior_locks(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    values = _external_values("generation-a", "constructor-lock-interrupt")
    _support(values, "generation-a", released=True)
    _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()

    original_tool_lock = registry._lock
    attempted_tool_acquire = threading.Event()

    class _InterruptingToolLock:
        def acquire(self, *args, **kwargs):
            attempted_tool_acquire.set()
            raise _InjectedBaseException("injected constructor acquisition failure")

        def release(self) -> None:  # pragma: no cover - must not release unacquired lock
            original_tool_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.release()

    monkeypatch.setattr(registry, "_lock", _InterruptingToolLock())

    with pytest.raises(_InjectedBaseException, match="injected constructor acquisition failure"):
        plugins_module._RegistrationTransaction(manager)

    assert attempted_tool_acquire.is_set()
    monkeypatch.setattr(registry, "_lock", original_tool_lock)

    acquired = threading.Event()
    released = threading.Event()

    def acquire_manager_lock_from_other_thread() -> None:
        with manager._lock:
            acquired.set()
            released.wait(timeout=5)

    thread = threading.Thread(target=acquire_manager_lock_from_other_thread, daemon=True)
    thread.start()
    assert acquired.wait(timeout=5), "constructor leaked the manager lock"
    released.set()
    thread.join(timeout=1)
    assert not thread.is_alive()

    with original_tool_lock:
        assert True
    for transaction in plugins_module._external_registry_transactions().values():
        with transaction.lock:
            assert True


def test_force_reload_lock_acquisition_interrupt_releases_prior_locks(
    tmp_path, monkeypatch, isolated_registries
):
    home = tmp_path / "hermes-home"
    old_values = _external_values("generation-a", "lock-interrupt")
    support = _support(old_values, "generation-a", released=True)
    support.saved_contexts = []
    _write_plugin(home)
    plugin_file = home / "plugins" / _PLUGIN_NAME / "__init__.py"
    plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace(
            "values = support.values",
            "support.saved_contexts.append(ctx)\n    values = support.values",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    old_context = support.saved_contexts[-1]

    from agent import image_gen_registry

    original_lock = image_gen_registry._registry_state._lock

    class _InterruptingLock:
        def __init__(self) -> None:
            self.acquire_count = 0

        def acquire(self, *args, **kwargs):
            self.acquire_count += 1
            if self.acquire_count == 2:
                raise KeyboardInterrupt("injected lock acquisition failure")
            return original_lock.acquire(*args, **kwargs)

        def release(self) -> None:
            original_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.release()

    interrupting_lock = _InterruptingLock()
    monkeypatch.setattr(image_gen_registry._registry_state, "_lock", interrupting_lock)
    support = _support(
        _external_values("generation-b", "lock-interrupt"),
        "generation-b",
        released=True,
    )
    support.saved_contexts = []

    with pytest.raises(KeyboardInterrupt, match="injected lock acquisition failure"):
        manager.discover_and_load(force=True)

    assert interrupting_lock.acquire_count == 2
    assert manager._live_context_generation == 0
    assert manager._live_context_revoking == set()

    retained_tool = "h1_lock_interrupt_tool"
    done = threading.Event()
    failures: list[BaseException] = []

    def register_from_other_thread() -> None:
        try:
            old_context.register_tool(
                retained_tool,
                "h1_transaction",
                {
                    "name": retained_tool,
                    "description": "retained",
                    "parameters": {"type": "object", "properties": {}},
                },
                lambda args, **kwargs: "retained",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=register_from_other_thread, daemon=True)
    thread.start()
    assert done.wait(timeout=5), "previously acquired commit lock was leaked"
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert failures == []
    assert registry.get_entry(retained_tool) is not None
