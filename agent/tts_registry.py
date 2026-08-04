"""
TTS Provider Registry
=====================

Central map of registered TTS providers. Populated by plugins at
import-time via :meth:`PluginContext.register_tts_provider`; consumed
by :mod:`tools.tts_tool` to dispatch ``text_to_speech`` tool calls to
the active plugin backend **when** the configured ``tts.provider``
name is neither a built-in nor a command-type provider.

Built-ins-always-win
--------------------
Plugin names that collide with a built-in TTS provider (``edge``,
``openai``, ``elevenlabs``, ``minimax``, ``gemini``, ``mistral``,
``xai``, ``piper``, ``kittentts``, ``neutts``) are rejected at
registration with a warning. This invariant is also re-checked at
dispatch time in :func:`tools.tts_tool._dispatch_to_plugin_provider`.

Command-providers-win-over-plugins
----------------------------------
This registry doesn't enforce the command-vs-plugin precedence — that
lives in the dispatcher, which checks for a same-name
``tts.providers.<name>: type: command`` entry before consulting the
registry. The rationale is locality: a name declared in the user's
``config.yaml`` is more specific to their setup than a plugin that
happens to be installed.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.tts_provider import TTSProvider
from registry_transaction import MappingRegistry, RegistryTransactionSurface

logger = logging.getLogger(__name__)


# Names reserved for native built-in TTS handlers. Plugins cannot
# register a name in this set — the registration call is rejected with
# a warning. **Kept in sync with ``BUILTIN_TTS_PROVIDERS`` in
# :mod:`tools.tts_tool`** — a regression test in
# ``tests/agent/test_tts_registry.py::TestBuiltinSync`` fails if the
# two lists drift. Importing from ``tools.tts_tool`` directly would
# create a circular dependency (``tools.tts_tool`` imports
# ``agent.tts_registry`` for dispatch).
_BUILTIN_NAMES = frozenset({
    "edge",
    "elevenlabs",
    "openai",
    "minimax",
    "xai",
    "mistral",
    "gemini",
    "neutts",
    "kittentts",
    "piper",
    "deepinfra",
})


_providers: Dict[str, TTSProvider] = {}
_lock = threading.RLock()
_registry_state = MappingRegistry("tts", _providers, _lock)


def _register_provider_in(
    target: MappingRegistry,
    provider: TTSProvider,
) -> Optional[str]:
    """Validate and register into a live registry or isolated transaction view.

    Rejects:

    - Non-:class:`TTSProvider` instances (raises :class:`TypeError`).
    - Empty/whitespace ``.name`` (raises :class:`ValueError`).
    - Names colliding with a built-in (logs a warning, silently
      ignores — built-ins-always-win invariant).

    Re-registration (same ``name``) overwrites the previous entry and
    logs a debug message — makes hot-reload scenarios (tests, dev
    loops) behave predictably.
    """
    if not isinstance(provider, TTSProvider):
        raise TypeError(
            f"register_provider() expects a TTSProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("TTS provider .name must be a non-empty string")
    key = name.strip().lower()
    if key in _BUILTIN_NAMES:
        logger.warning(
            "TTS provider '%s' shadows a built-in name; registration ignored. "
            "Built-in TTS providers (%s) always win — pick a different name.",
            key, ", ".join(sorted(_BUILTIN_NAMES)),
        )
        return
    _added, existing = target.put(key, provider)
    if existing is not None:
        logger.debug(
            "TTS provider '%s' re-registered (was %r)",
            key, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered TTS provider '%s' (%s)",
            key, type(provider).__name__,
        )
    return key


def register_provider(provider: TTSProvider) -> None:
    """Register a TTS provider."""
    _register_provider_in(_registry_state, provider)


_plugin_transaction = RegistryTransactionSurface(
    _registry_state,
    _register_provider_in,
)


def list_providers() -> List[TTSProvider]:
    """Return all registered providers, sorted by name."""
    items = _registry_state.values()
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str) -> Optional[TTSProvider]:
    """Return the provider registered under *name*, or None.

    Name matching is case-insensitive and whitespace-tolerant — mirrors
    how ``tools.tts_tool._get_provider`` normalizes the configured
    ``tts.provider`` value.
    """
    if not isinstance(name, str):
        return None
    return _registry_state.get(name.strip().lower())


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    _registry_state.clear()
