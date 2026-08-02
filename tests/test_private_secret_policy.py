"""Host-owned private plugin secret policy invariants."""

from __future__ import annotations

import threading

import pytest

import private_secret_policy as policy


@pytest.fixture(autouse=True)
def restore_policy():
    snapshot = policy.snapshot_private_secret_policy()
    try:
        yield
    finally:
        policy.restore_private_secret_policy(snapshot)


def test_policy_starts_healthy_and_empty() -> None:
    env = {"PUBLIC_SETTING": "public-sentinel"}

    assert policy.scrub_private_secret_env(env) == env


def test_replacement_is_validated_before_old_policy_changes() -> None:
    policy.replace_private_secret_names(["OLD_PRIVATE_SECRET"])

    def broken_replacement():
        yield "NEW_PRIVATE_SECRET"
        raise RuntimeError("SENTINEL_SECRET_VALUE")

    with pytest.raises(
        policy.PrivateSecretPolicyError,
        match="private-secret policy replacement failed",
    ) as excinfo:
        policy.replace_private_secret_names(broken_replacement())

    assert "SENTINEL_SECRET_VALUE" not in str(excinfo.value)
    scrubbed = policy.scrub_private_secret_env(
        {
            "OLD_PRIVATE_SECRET": "SENTINEL_OLD_VALUE",
            "NEW_PRIVATE_SECRET": "SENTINEL_NEW_VALUE",
        }
    )
    assert "OLD_PRIVATE_SECRET" not in scrubbed
    assert scrubbed["NEW_PRIVATE_SECRET"] == "SENTINEL_NEW_VALUE"


@pytest.mark.parametrize(
    "invalid_name",
    ["", "9LEADING_DIGIT", "HAS-DASH", "HAS.DOT", "HAS=EQUALS", None],
)
def test_invalid_environment_names_raise_typed_sanitized_errors(invalid_name) -> None:
    with pytest.raises(
        policy.PrivateSecretPolicyError,
        match="invalid private secret environment name",
    ) as excinfo:
        policy.prepare_private_secret_names([invalid_name])

    if invalid_name not in ("", None):
        assert str(invalid_name) not in str(excinfo.value)


def test_snapshot_is_opaque_and_restore_is_atomic() -> None:
    policy.replace_private_secret_names(["FIRST_PRIVATE_SECRET"])
    snapshot = policy.snapshot_private_secret_policy()
    assert not hasattr(snapshot, "names")

    policy.replace_private_secret_names(["SECOND_PRIVATE_SECRET"])
    policy.restore_private_secret_policy(snapshot)

    scrubbed = policy.scrub_private_secret_env(
        {
            "FIRST_PRIVATE_SECRET": "SENTINEL_FIRST_VALUE",
            "SECOND_PRIVATE_SECRET": "SENTINEL_SECOND_VALUE",
        }
    )
    assert "FIRST_PRIVATE_SECRET" not in scrubbed
    assert scrubbed["SECOND_PRIVATE_SECRET"] == "SENTINEL_SECOND_VALUE"


def test_concurrent_reader_observes_old_or_complete_new_set_never_partial() -> None:
    policy.replace_private_secret_names(["OLD_PRIVATE_SECRET"])
    preparing = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def replacement_names():
        yield "NEW_PRIVATE_SECRET_A"
        preparing.set()
        if not release.wait(timeout=5):
            raise RuntimeError("replacement test timeout")
        yield "NEW_PRIVATE_SECRET_B"

    def replace() -> None:
        try:
            policy.replace_private_secret_names(replacement_names())
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    thread = threading.Thread(target=replace, daemon=True)
    thread.start()
    assert preparing.wait(timeout=5)
    during = policy.scrub_private_secret_env(
        {
            "OLD_PRIVATE_SECRET": "old",
            "NEW_PRIVATE_SECRET_A": "new-a",
            "NEW_PRIVATE_SECRET_B": "new-b",
        }
    )
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    after = policy.scrub_private_secret_env(
        {
            "OLD_PRIVATE_SECRET": "old",
            "NEW_PRIVATE_SECRET_A": "new-a",
            "NEW_PRIVATE_SECRET_B": "new-b",
        }
    )

    assert during == {
        "NEW_PRIVATE_SECRET_A": "new-a",
        "NEW_PRIVATE_SECRET_B": "new-b",
    }
    assert after == {"OLD_PRIVATE_SECRET": "old"}


def test_windows_matching_is_case_insensitive_but_posix_is_exact(monkeypatch) -> None:
    policy.replace_private_secret_names(["PLUGIN_PRIVATE_SECRET"])

    monkeypatch.setattr(policy, "_is_windows_platform", lambda: False)
    posix = policy.scrub_private_secret_env(
        {"plugin_private_secret": "SENTINEL_POSIX_VALUE"}
    )
    assert posix["plugin_private_secret"] == "SENTINEL_POSIX_VALUE"

    monkeypatch.setattr(policy, "_is_windows_platform", lambda: True)
    windows = policy.scrub_private_secret_env(
        {"plugin_private_secret": "SENTINEL_WINDOWS_VALUE"}
    )
    assert "plugin_private_secret" not in windows


def test_unhealthy_lookup_fails_closed_with_typed_error(monkeypatch) -> None:
    monkeypatch.setattr(policy, "_healthy", False)

    with pytest.raises(
        policy.PrivateSecretPolicyError,
        match="private-secret policy is unavailable",
    ):
        policy.scrub_private_secret_env({"PUBLIC_SETTING": "public-sentinel"})
