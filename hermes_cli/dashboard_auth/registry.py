"""Module-level registry for DashboardAuthProvider instances.

Plugins call ``register_provider`` via the plugin context hook at startup.
The auth gate middleware iterates ``list_providers()`` and uses
``get_provider`` to dispatch on the session's ``provider`` field.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    assert_protocol_compliance,
)
from registry_transaction import MappingRegistry, RegistryTransactionSurface

_log = logging.getLogger(__name__)
_lock = threading.RLock()
_providers: dict[str, DashboardAuthProvider] = {}
_registry_state = MappingRegistry("dashboard", _providers, _lock)


def _register_provider_in(
    target: MappingRegistry,
    provider: DashboardAuthProvider,
) -> str:
    """Validate and register into a live registry or isolated transaction view.

    Raises:
        TypeError: on protocol violation.
        ValueError: if a provider with the same name is already registered.
    """
    assert_protocol_compliance(type(provider))
    name = provider.name
    display_name = provider.display_name
    added, _existing = target.put(name, provider, replace=False)
    if not added:
        raise ValueError(
            f"dashboard-auth provider already registered: {name!r}"
        )
    _log.info(
        "dashboard-auth: registered provider %r (%s)",
        name, display_name,
    )
    return name


def register_provider(provider: DashboardAuthProvider) -> None:
    """Register a provider."""
    _register_provider_in(_registry_state, provider)


_plugin_transaction = RegistryTransactionSurface(
    _registry_state,
    _register_provider_in,
)


def get_provider(name: str) -> Optional[DashboardAuthProvider]:
    """Return the registered provider for ``name``, or None if unknown."""
    return _registry_state.get(name)


def list_providers() -> List[DashboardAuthProvider]:
    """All registered providers, in registration order."""
    return _registry_state.values()


def list_token_providers() -> List[DashboardAuthProvider]:
    """Registered providers that support non-interactive token auth.

    The subset of ``list_providers()`` whose ``supports_token`` flag is True,
    in registration order. The ``token_auth`` middleware seam consults these
    (and only these) when a token-authable route is hit, so OAuth/password-only
    providers are never asked to ``verify_token``. Returns an empty list when
    no token provider is registered — a token-authable route then fails
    closed (401), never open.
    """
    providers = _registry_state.values()
    return [p for p in providers if getattr(p, "supports_token", False)]


def list_session_providers() -> List[DashboardAuthProvider]:
    """Registered providers with supports_session True (interactive cookie
    sessions). The login page, /auth/login, and the gate's verify/refresh loops
    consult only these. Mirror of list_token_providers.
    """
    providers = _registry_state.values()
    return [p for p in providers if getattr(p, "supports_session", True)]


def clear_providers() -> None:
    """Test-only: drop all registrations."""
    _registry_state.clear()
