"""Client-credentials OAuth for autonomous MCP agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
import pytest

from tools.mcp_oauth_manager import (
    ClientCredentialsAuth,
    MCPOAuthManager,
    OAuthClientCredentialsError,
)


def _config(**overrides):
    config = {
        "grant_type": "client_credentials",
        "token_endpoint": "https://tenant.auth0.com/oauth/token",
        "client_id": "hermes-client",
        "client_secret": "top-secret",
        "audience": "https://dps-work.example/mcp",
        "scope": "dps.read dps.action.internal",
    }
    config.update(overrides)
    return config


def test_manager_selects_client_credentials_without_browser_oauth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = MCPOAuthManager()

    provider = manager.get_or_build_provider(
        "dps-work", "https://dps-work.example/mcp", _config()
    )

    assert isinstance(provider, ClientCredentialsAuth)
    assert not (tmp_path / "mcp-tokens").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_endpoint", "http://tenant.auth0.com/oauth/token"),
        ("token_endpoint", "https://user:pass@tenant.auth0.com/oauth/token"),
        ("client_id", ""),
        ("client_secret", ""),
        ("audience", ""),
        ("scope", ""),
    ],
)
def test_client_credentials_config_fails_closed(field, value, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with pytest.raises(OAuthClientCredentialsError, match=field):
        MCPOAuthManager().get_or_build_provider(
            "dps-work", "https://dps-work.example/mcp", _config(**{field: value})
        )


def test_m2m_failure_never_falls_back_to_browser_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "tools.mcp_oauth_manager._HERMES_PROVIDER_CLS",
        lambda **_kwargs: pytest.fail("browser OAuth fallback was attempted"),
    )

    with pytest.raises(OAuthClientCredentialsError):
        MCPOAuthManager().get_or_build_provider(
            "dps-work",
            "https://dps-work.example/mcp",
            _config(client_secret=""),
        )


@dataclass
class _Clock:
    value: float = 1_000.0

    def __call__(self) -> float:
        return self.value


@pytest.mark.asyncio
async def test_token_request_is_exact_cached_and_never_persisted(tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "access_token": "ACCESS-ONE",
                "token_type": "Bearer",
                "expires_in": 300,
            },
        )

    transport = httpx.MockTransport(handler)
    clock = _Clock()
    auth = ClientCredentialsAuth(
        server_name="dps-work",
        token_endpoint="https://tenant.auth0.com/oauth/token",
        client_id="hermes-client",
        client_secret="top-secret",
        audience="https://dps-work.example/mcp",
        scope="dps.read dps.action.internal",
        clock=clock,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=transport, **kwargs
        ),
    )

    first = await auth.get_access_token()
    second = await auth.get_access_token()

    assert first == second == "ACCESS-ONE"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://tenant.auth0.com/oauth/token"
    assert dict(httpx.QueryParams(requests[0].content.decode())) == {
        "grant_type": "client_credentials",
        "client_id": "hermes-client",
        "client_secret": "top-secret",
        "audience": "https://dps-work.example/mcp",
        "scope": "dps.read dps.action.internal",
    }
    assert not (tmp_path / "mcp-tokens").exists()


@pytest.mark.asyncio
async def test_token_refreshes_before_expiry_and_deduplicates_concurrent_fetches():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            request=request,
            json={
                "access_token": f"ACCESS-{calls}",
                "token_type": "Bearer",
                "expires_in": 120,
            },
        )

    clock = _Clock()
    auth = ClientCredentialsAuth(
        server_name="dps-work",
        token_endpoint="https://tenant.auth0.com/oauth/token",
        client_id="hermes-client",
        client_secret="top-secret",
        audience="https://dps-work.example/mcp",
        scope="dps.read",
        refresh_skew=30,
        clock=clock,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )

    assert await auth.get_access_token() == "ACCESS-1"
    clock.value += 91
    refreshed = await asyncio.gather(*(auth.get_access_token() for _ in range(8)))

    assert refreshed == ["ACCESS-2"] * 8
    assert calls == 2


@pytest.mark.asyncio
async def test_401_invalidates_m2m_token_for_manager_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = MCPOAuthManager()
    provider = manager.get_or_build_provider(
        "dps-work", "https://dps-work.example/mcp", _config()
    )
    provider._access_token = "FAILED"
    provider._expires_at = float("inf")

    recovered = await manager.handle_401(
        "dps-work", failed_access_token="FAILED"
    )

    assert recovered is True
    assert provider._access_token is None


@pytest.mark.asyncio
async def test_401_ignores_stale_browser_token_file_after_m2m_migration(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir()
    (token_dir / "dps-work.json").write_text(
        '{"access_token":"STALE-PKCE"}', encoding="utf-8"
    )
    manager = MCPOAuthManager()
    provider = manager.get_or_build_provider(
        "dps-work", "https://dps-work.example/mcp", _config()
    )
    provider._access_token = "FAILED-M2M"
    provider._expires_at = float("inf")

    recovered = await manager.handle_401(
        "dps-work", failed_access_token="FAILED-M2M"
    )

    assert recovered is True
    assert provider._access_token is None


@pytest.mark.asyncio
async def test_http_auth_retries_401_once_with_a_fresh_token():
    token_calls = 0
    mcp_authorizations: list[str] = []

    def token_handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        token_calls += 1
        return httpx.Response(
            200,
            request=request,
            json={
                "access_token": f"ACCESS-{token_calls}",
                "token_type": "Bearer",
                "expires_in": 300,
            },
        )

    def mcp_handler(request: httpx.Request) -> httpx.Response:
        mcp_authorizations.append(request.headers["Authorization"])
        return httpx.Response(
            401 if len(mcp_authorizations) == 1 else 200,
            request=request,
        )

    auth = ClientCredentialsAuth(
        server_name="dps-work",
        token_endpoint="https://tenant.auth0.com/oauth/token",
        client_id="hermes-client",
        client_secret="top-secret",
        audience="https://dps-work.example/mcp",
        scope="dps.read",
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(token_handler), **kwargs
        ),
    )

    async with httpx.AsyncClient(
        auth=auth, transport=httpx.MockTransport(mcp_handler)
    ) as client:
        response = await client.post("https://dps-work.example/mcp")

    assert response.status_code == 200
    assert token_calls == 2
    assert mcp_authorizations == ["Bearer ACCESS-1", "Bearer ACCESS-2"]


@pytest.mark.asyncio
async def test_token_errors_are_sanitized_and_do_not_leak_secret():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            request=request,
            text='{"error":"invalid_client","client_secret":"top-secret"}',
        )

    auth = ClientCredentialsAuth(
        server_name="dps-work",
        token_endpoint="https://tenant.auth0.com/oauth/token",
        client_id="hermes-client",
        client_secret="top-secret",
        audience="https://dps-work.example/mcp",
        scope="dps.read",
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )

    with pytest.raises(OAuthClientCredentialsError) as raised:
        await auth.get_access_token()

    assert "401" in str(raised.value)
    assert "top-secret" not in str(raised.value)


def test_manager_rebuilds_provider_when_credentials_change(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = MCPOAuthManager()
    first = manager.get_or_build_provider(
        "dps-work", "https://dps-work.example/mcp", _config()
    )
    second = manager.get_or_build_provider(
        "dps-work",
        "https://dps-work.example/mcp",
        _config(client_secret="rotated-secret"),
    )

    assert first is not second
