"""Host-owned policy for plugin-declared private environment names.

The policy stores names only.  It deliberately has no secret-value lookup or
PluginContext integration, and depends only on the Python standard library so
child-process builders and plugin discovery can import it without cycles.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable, Mapping


class PrivateSecretPolicyError(RuntimeError):
    """Raised when the private-secret deny policy cannot be applied safely."""


class _PrivateSecretPolicySnapshot:
    """Opaque, host-private checkpoint used by discovery transactions."""

    __slots__ = ("_private_names", "_was_healthy")

    def __init__(self, private_names: frozenset[str], was_healthy: bool) -> None:
        self._private_names = private_names
        self._was_healthy = was_healthy


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_lock = threading.RLock()
_private_names: frozenset[str] = frozenset()
_healthy = True


def _is_windows_platform() -> bool:
    """Small platform seam so case behavior is testable without patching os.name."""
    return os.name == "nt"


def _checked_names_locked() -> frozenset[str]:
    if not _healthy or not isinstance(_private_names, frozenset):
        raise PrivateSecretPolicyError("private-secret policy is unavailable")
    return _private_names


def prepare_private_secret_names(names: Iterable[object]) -> frozenset[str]:
    """Validate *names* completely without changing the live policy."""
    try:
        candidate = list(names)
    except Exception:
        raise PrivateSecretPolicyError(
            "private-secret policy replacement failed"
        ) from None

    prepared: set[str] = set()
    for name in candidate:
        if not isinstance(name, str) or _ENV_NAME_RE.fullmatch(name) is None:
            raise PrivateSecretPolicyError(
                "invalid private secret environment name"
            )
        prepared.add(name)
    return frozenset(prepared)


def replace_private_secret_names(names: Iterable[object]) -> None:
    """Atomically replace the live immutable deny set after full validation."""
    prepared = prepare_private_secret_names(names)
    global _private_names, _healthy
    with _lock:
        _private_names = prepared
        _healthy = True


def snapshot_private_secret_policy() -> object:
    """Return an opaque, immutable checkpoint of the current healthy policy."""
    with _lock:
        names = _checked_names_locked()
        return _PrivateSecretPolicySnapshot(names, True)


def restore_private_secret_policy(snapshot: object) -> None:
    """Atomically restore a checkpoint created by this module."""
    if not isinstance(snapshot, _PrivateSecretPolicySnapshot):
        raise PrivateSecretPolicyError("invalid private-secret policy snapshot")
    if not snapshot._was_healthy or not isinstance(
        snapshot._private_names, frozenset
    ):
        raise PrivateSecretPolicyError("invalid private-secret policy snapshot")

    global _private_names, _healthy
    with _lock:
        _private_names = snapshot._private_names
        _healthy = True


def scrub_private_secret_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return *env* without any currently protected private secret names.

    A policy lookup problem raises :class:`PrivateSecretPolicyError`; callers
    must let it propagate so the child process is not launched.
    """
    with _lock:
        names = _checked_names_locked()

    if _is_windows_platform():
        folded_names = frozenset(name.casefold() for name in names)
        return {
            key: value
            for key, value in env.items()
            if key.casefold() not in folded_names
        }
    return {key: value for key, value in env.items() if key not in names}
