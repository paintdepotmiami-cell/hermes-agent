"""Private secret declarations through public PluginManager discovery."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

import private_secret_policy as policy
from hermes_cli.plugins import PluginManager


_PRIVATE_A = "PLUGIN_ALPHA_PRIVATE_SECRET"
_PRIVATE_B = "PLUGIN_BETA_PRIVATE_SECRET"
_SECRET_VALUE = "SENTINEL_PLUGIN_SECRET_VALUE"


@pytest.fixture(autouse=True)
def isolated_discovery(tmp_path, monkeypatch):
    snapshot = policy.snapshot_private_secret_policy()
    empty_bundled = tmp_path / "empty-bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    try:
        yield
    finally:
        policy.restore_private_secret_policy(snapshot)
        for module_name in list(sys.modules):
            if module_name.startswith("hermes_plugins.private_policy_"):
                sys.modules.pop(module_name, None)


def _write_config(
    home: Path,
    *,
    enabled: list[str],
    disabled: list[str] | None = None,
) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config = {"plugins": {"enabled": enabled, "disabled": disabled or []}}
    (home / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )


def _write_plugin(
    home: Path,
    name: str,
    *,
    requires_env: list,
    source: str = "def register(ctx):\n    return None\n",
) -> Path:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": "1.0",
        "requires_env": requires_env,
    }
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _is_protected(name: str) -> bool:
    return name not in policy.scrub_private_secret_env({name: _SECRET_VALUE})


def test_enabled_winner_is_protected_before_import_and_context_has_no_value(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "hermes-home"
    plugin_name = "private_policy_enabled"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(_PRIVATE_A, _SECRET_VALUE)
    _write_plugin(
        home,
        plugin_name,
        requires_env=[{"name": _PRIVATE_A, "secret": True}],
        source=f'''from private_secret_policy import scrub_private_secret_env

private_probe = dict([({_PRIVATE_A!r}, "import-probe")])
protected_before_import = {_PRIVATE_A!r} not in scrub_private_secret_env(
    private_probe
)
context_has_secret_value = False

def register(ctx):
    global context_has_secret_value
    context_has_secret_value = any(
        value == {_SECRET_VALUE!r} for value in vars(ctx).values()
    )
''',
    )
    _write_config(home, enabled=[plugin_name])

    manager = PluginManager()
    manager.discover_and_load()

    module = sys.modules[f"hermes_plugins.{plugin_name}"]
    assert module.protected_before_import is True
    assert module.context_has_secret_value is False
    assert manager._plugins[plugin_name].enabled is True
    assert _is_protected(_PRIVATE_A)


def test_enabled_plugin_failure_remains_disabled_but_declared_name_is_protected(
    tmp_path, monkeypatch, caplog
) -> None:
    home = tmp_path / "hermes-home"
    plugin_name = "private_policy_failure"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(_PRIVATE_A, _SECRET_VALUE)
    _write_plugin(
        home,
        plugin_name,
        requires_env=[{"name": _PRIVATE_A, "secret": True}],
        source=f'raise RuntimeError({_SECRET_VALUE!r})\n',
    )
    _write_config(home, enabled=[plugin_name])

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager = PluginManager()
        manager.discover_and_load()

    loaded = manager._plugins[plugin_name]
    assert loaded.enabled is False
    assert loaded.module is None
    assert _is_protected(_PRIVATE_A)
    assert _SECRET_VALUE not in (loaded.error or "")
    assert _SECRET_VALUE not in caplog.text


def test_disabled_manifest_does_not_add_private_names(tmp_path, monkeypatch) -> None:
    home = tmp_path / "hermes-home"
    plugin_name = "private_policy_disabled"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_plugin(
        home,
        plugin_name,
        requires_env=[{"name": _PRIVATE_A, "secret": True}],
        source="raise AssertionError('disabled plugin imported')\n",
    )
    _write_config(home, enabled=[plugin_name], disabled=[plugin_name])

    manager = PluginManager()
    manager.discover_and_load()

    assert manager._plugins[plugin_name].error == "disabled via config"
    assert not _is_protected(_PRIVATE_A)
    assert f"hermes_plugins.{plugin_name}" not in sys.modules


def test_malformed_enabled_secret_declaration_fails_before_import_safely(
    tmp_path, monkeypatch, caplog
) -> None:
    home = tmp_path / "hermes-home"
    plugin_name = "private_policy_malformed"
    marker = tmp_path / "must-not-exist"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MALFORMED_PRIVATE_SECRET", _SECRET_VALUE)
    _write_plugin(
        home,
        plugin_name,
        requires_env=[
            {
                "name": "MALFORMED=PRIVATE=SECRET",
                "secret": True,
                "description": _SECRET_VALUE,
            }
        ],
        source=f'''from pathlib import Path
Path({str(marker)!r}).write_text("imported", encoding="utf-8")
def register(ctx):
    return None
''',
    )
    _write_config(home, enabled=[plugin_name])

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager = PluginManager()
        manager.discover_and_load()

    loaded = manager._plugins[plugin_name]
    assert loaded.enabled is False
    assert loaded.module is None
    assert loaded.error == (
        "PrivateSecretPolicyError: plugin private-secret declaration invalid"
    )
    assert not marker.exists()
    assert _SECRET_VALUE not in caplog.text
    assert _SECRET_VALUE not in loaded.error


def test_force_success_replaces_stale_policy_and_force_failure_restores_old(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "hermes-home"
    alpha = "private_policy_alpha"
    beta = "private_policy_beta"
    interrupting = "private_policy_interrupting"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_plugin(
        home,
        alpha,
        requires_env=[{"name": _PRIVATE_A, "secret": True}],
    )
    _write_config(home, enabled=[alpha])
    manager = PluginManager()
    manager.discover_and_load()
    assert _is_protected(_PRIVATE_A)

    _write_plugin(
        home,
        beta,
        requires_env=[{"name": _PRIVATE_B, "secret": True}],
    )
    _write_config(home, enabled=[beta], disabled=[alpha])
    manager.discover_and_load(force=True)
    assert not _is_protected(_PRIVATE_A)
    assert _is_protected(_PRIVATE_B)

    _write_plugin(
        home,
        interrupting,
        requires_env=[{"name": _PRIVATE_A, "secret": True}],
        source="raise KeyboardInterrupt('force policy rollback')\n",
    )
    _write_config(home, enabled=[interrupting], disabled=[alpha, beta])
    with pytest.raises(KeyboardInterrupt, match="force policy rollback"):
        manager.discover_and_load(force=True)

    assert not _is_protected(_PRIVATE_A)
    assert _is_protected(_PRIVATE_B)


def test_initial_base_exception_keeps_complete_winner_policy_and_is_retryable(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "hermes-home"
    committed = "private_policy_a_committed"
    interrupting = "private_policy_z_interrupting"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_plugin(
        home,
        committed,
        requires_env=[{"name": _PRIVATE_A, "secret": True}],
    )
    _write_plugin(
        home,
        interrupting,
        requires_env=[{"name": _PRIVATE_B, "secret": True}],
        source="raise SystemExit('initial policy sweep interrupted')\n",
    )
    _write_config(home, enabled=[committed, interrupting])
    manager = PluginManager()

    with pytest.raises(SystemExit, match="initial policy sweep interrupted"):
        manager.discover_and_load()

    assert manager._plugins[committed].enabled is True
    assert manager._discovered is False
    assert _is_protected(_PRIVATE_A)
    assert _is_protected(_PRIVATE_B)

    _write_plugin(
        home,
        interrupting,
        requires_env=[{"name": _PRIVATE_B, "secret": True}],
    )
    manager.discover_and_load()

    assert manager._discovered is True
    assert manager._plugins[interrupting].enabled is True
    assert _is_protected(_PRIVATE_A)
    assert _is_protected(_PRIVATE_B)
