"""Generic governed MCP dispatch through plugin discovery and live handlers."""

from __future__ import annotations

import asyncio
import builtins
from contextlib import contextmanager
import dataclasses
import hashlib
import hmac
import inspect
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from agent._trusted_interaction_issuer import (
    _bind_trusted_interaction,
    _clear_trusted_interactions,
    _issue_trusted_interaction,
    _publish_trusted_interaction,
    _reset_trusted_interaction,
)
from agent.trusted_interaction import get_current_trusted_interaction
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli import plugins as plugin_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import fresh_approval, mcp_tool
from tools.approval import register_gateway_notify, unregister_gateway_notify
from tools.governed_mcp import (
    ApprovalReceipt,
    BoundOperation,
    GovernedDispatchPlan,
    GovernedMCPHostServices,
    GovernedOutcome,
    MCPGovernanceRequest,
    MCPToolDescriptor,
)
from tools.registry import registry


SERVER = "neutral-governed-server"
EXECUTE = "execute-operation"
PREPARE = "prepare-operation"
PREFLIGHT = "inspect-operation"
READ = "read-operation"
SECRET_NAME = "NEUTRAL_GOVERNOR_SENTINEL_SECRET"
SECRET_VALUE = "SENTINEL_GOVERNOR_SECRET_VALUE"
META_FIELD = "transport.attestation"
BOUND_META_FIELD = "transport.bound"
PREPARE_META_SENTINEL = "PREPARE_TRANSPORT_META_SENTINEL"
BOUND_META_SENTINEL = "BOUND_TRANSPORT_META_SENTINEL"
EXECUTE_META_SENTINEL = "EXECUTE_TRANSPORT_META_SENTINEL"
FULL_PREVIEW_TAIL = "FULL_PREVIEW_TAIL_SENTINEL"
FULL_DISPLAY_TAIL = "FULL_DISPLAY_TAIL_SENTINEL"
FULL_PREVIEW = "p" * 5_000 + FULL_PREVIEW_TAIL
FULL_DISPLAY = "d" * 5_000 + FULL_DISPLAY_TAIL


def _run_direct(coro_or_factory, timeout=30):
    del timeout
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def isolated_governed_state(tmp_path, monkeypatch):
    tool_snapshot = registry._take_transaction_snapshot()
    server_snapshot = dict(mcp_tool._servers)
    old_manager = plugin_module._plugin_manager
    old_coordinator = fresh_approval._DEFAULT_COORDINATOR
    empty_bundled = tmp_path / "empty-bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    monkeypatch.setenv(SECRET_NAME, SECRET_VALUE)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", _run_direct)
    _clear_trusted_interactions()
    try:
        yield
    finally:
        _clear_trusted_interactions()
        fresh_approval._DEFAULT_COORDINATOR = old_coordinator
        plugin_module._plugin_manager = old_manager
        for session_key in (
            "neutral-desktop-session",
            "neutral-telegram-session",
        ):
            unregister_gateway_notify(session_key)
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._servers.update(server_snapshot)
        registry._restore_transaction_snapshot(tool_snapshot)
        for module_name in list(sys.modules):
            if module_name.startswith("hermes_plugins.neutral_governor_"):
                sys.modules.pop(module_name, None)


def _write_plugin(
    home: Path,
    name: str,
    source: str,
    *,
    allow_governance: bool = True,
    server_name: str = SERVER,
) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "requires_env": [{"name": SECRET_NAME, "secret": True}],
                "version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(source, encoding="utf-8")
    config = {
        "plugins": {
            "enabled": [name],
            "entries": {
                name: {"allow_mcp_governance": allow_governance},
            },
        }
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )


def _discover_plugin(home: Path, monkeypatch, name: str) -> tuple[PluginManager, object]:
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    plugin_module._plugin_manager = manager
    manager.discover_and_load()
    loaded = manager._plugins[name]
    return manager, loaded


def _server_with_tools(*tools: Tool, call_tool=None):
    server = mcp_tool.MCPServerTask(SERVER)
    session = SimpleNamespace(
        call_tool=call_tool
        or AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="unused")]
            )
        )
    )
    server.session = session
    server._tools = list(tools)
    server._config = {}
    server.tool_timeout = 3
    with mcp_tool._lock:
        mcp_tool._servers[SERVER] = server
    registered = mcp_tool._register_server_tools(SERVER, server, {})
    server._registered_tool_names = list(registered)
    return server, session, registered


def _tool(
    name: str,
    *,
    annotations: ToolAnnotations | None = None,
    meta: dict | None = None,
) -> Tool:
    values = {
        "name": name,
        "description": f"Exact description for {name}",
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "signature": {"type": "string"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    }
    if annotations is not None:
        values["annotations"] = annotations
    if meta is not None:
        values["_meta"] = meta
    return Tool(**values)


@contextmanager
def _trusted_scope(*, platform: str):
    if platform == "telegram":
        values = {
            "surface": "gateway",
            "platform": "telegram",
            "actor_id": "neutral-actor",
            "channel_id": "neutral-chat",
            "chat_id": "neutral-chat",
            "thread_id": "neutral-thread",
            "session_key": "neutral-telegram-session",
            "session_id": "neutral-telegram-session",
            "message_id": "neutral-origin-message",
            "profile_id": "default",
        }
    else:
        values = {
            "surface": "desktop",
            "platform": "local",
            "actor_id": "neutral-local-transport",
            "channel_id": "neutral-local-transport",
            "chat_id": "neutral-runtime",
            "thread_id": None,
            "session_key": "neutral-desktop-session",
            "session_id": "neutral-runtime",
            "message_id": "neutral-origin-message",
            "profile_id": "default",
        }
    interaction = _issue_trusted_interaction(
        **values,
        received_at=time.time(),
    )
    _publish_trusted_interaction(interaction)
    token = _bind_trusted_interaction(interaction)
    session_tokens = set_session_vars(
        platform=interaction.platform,
        source=interaction.surface,
        chat_id=interaction.chat_id,
        thread_id=interaction.thread_id or "",
        user_id=interaction.actor_id,
        session_key=interaction.session_key,
        session_id=interaction.session_id,
        ui_session_id=(interaction.chat_id if interaction.platform == "local" else ""),
        message_id=interaction.message_id,
        profile=interaction.profile_id,
    )
    try:
        yield interaction
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)
        _clear_trusted_interactions(interaction=interaction)


def _capture_policy_source() -> str:
    return f'''seen = []

class Policy:
    def prepare(self, request, services):
        global seen
        seen.append(request)
        raise RuntimeError("policy stopped before provider call")

    def finalize(self, request, operation, result):
        raise AssertionError("finalize must not run")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


def _positive_policy_source() -> str:
    return f'''from tools.governed_mcp import GovernedDispatchPlan, GovernedOutcome

seen = []
saved_receipt = None

class Policy:
    def prepare(self, request, services):
        seen.append((request, services))
        preflight = services.preflight_call(
            {PREPARE!r},
            {{"probe": "sentinel"}},
            meta={{{META_FIELD!r}: {PREPARE_META_SENTINEL!r}}},
        )
        seen.append(("preflight", preflight))
        proposal_hash = services.hmac_sha256(
            {SECRET_NAME!r},
            '{{"proposal":"sentinel"}}',
        )
        base_meta = {{{BOUND_META_FIELD!r}: {BOUND_META_SENTINEL!r}}}
        operation = services.bind_operation(
            {EXECUTE!r},
            request.arguments,
            proposal_hash=proposal_hash,
            preview={FULL_PREVIEW!r},
            display_text={FULL_DISPLAY!r},
            base_meta=base_meta,
            allowed_post_approval_patch_fields=("signature",),
            allowed_post_approval_meta_fields=({META_FIELD!r},),
        )
        base_meta[{BOUND_META_FIELD!r}] = "MUTATED_AFTER_BIND_SENTINEL"
        receipt = services.request_fresh_approval(operation)
        global saved_receipt
        saved_receipt = receipt
        return GovernedDispatchPlan(
            bound_operation=operation,
            approval_receipt=receipt,
            post_approval_patch={{"signature": proposal_hash}},
            post_approval_meta_patch={{{META_FIELD!r}: {EXECUTE_META_SENTINEL!r}}},
        )

    def finalize(self, request, operation, result):
        seen.append((request, operation, result))
        if result.structuredContent["verified"] is not True:
            return GovernedOutcome(status="unknown", summary="verification absent")
        return GovernedOutcome(
            status="verified",
            summary="neutral provider verified the operation",
            evidence={{"receipt": result.structuredContent["receipt"]}},
        )

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


def _read_only_policy_source(*, decision: str = "valid") -> str:
    if decision == "valid":
        expression = "ReadOnlyPassThrough(request.descriptor)"
    elif decision == "foreign":
        expression = (
            "ReadOnlyPassThrough(dataclasses.replace("
            "request.descriptor, fingerprint='0' * 64))"
        )
    else:
        expression = "{'descriptor': request.descriptor}"
    return f'''import dataclasses
from tools.governed_mcp import ReadOnlyPassThrough

seen = []

class Policy:
    def prepare(self, request, services):
        seen.append((request, services))
        return {expression}

    def finalize(self, request, operation, result):
        raise AssertionError("read-only pass-through must not finalize")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


def _invalid_target_policy_source(target_raw_name: str) -> str:
    return f'''class Policy:
    def prepare(self, request, services):
        services.preflight_call({PREPARE!r}, {{"probe": "sentinel"}})
        services.bind_operation(
            {target_raw_name!r},
            {{"value": "sentinel"}},
            proposal_hash="proposal-sentinel",
            preview="neutral preview",
            display_text="neutral display",
        )
        raise AssertionError("invalid target unexpectedly bound")

    def finalize(self, request, operation, result):
        raise AssertionError("invalid target must not finalize")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


def _direct_target_block_policy_source() -> str:
    return f'''seen = []

class Policy:
    def prepare(self, request, services):
        seen.append(request.descriptor.raw_tool_name)
        if request.descriptor.raw_tool_name == {EXECUTE!r}:
            raise RuntimeError("plugin policy blocks direct target invocation")
        raise AssertionError("unexpected invocation")

    def finalize(self, request, operation, result):
        raise AssertionError("direct target invocation must not finalize")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


def _auto_approve_notifier(coordinator, clock, observed):
    def notify(data):
        observed.append(dict(data))
        request = coordinator.get_request(data["approval_id"])
        clock[0] += 1
        binding = request.subject.conversation
        responder = coordinator._mint_authenticated_responder_for_host(
            assurance=(
                "telegram_authenticated"
                if binding.platform == "telegram"
                else "local_desktop_session"
            ),
            principal_id=(binding.actor_id if binding.platform == "telegram" else None),
            binding=binding,
            inbound_id="neutral-later-approval-message",
        )
        coordinator.resolve(
            request.approval_id,
            decision="approve_once",
            responder=responder,
        )

    return notify


def test_public_contract_is_immutable_and_host_capabilities_are_host_only():
    from tools.governed_mcp import ReadOnlyPassThrough

    assert dataclasses.is_dataclass(MCPToolDescriptor)
    assert dataclasses.is_dataclass(MCPGovernanceRequest)
    assert dataclasses.is_dataclass(ApprovalReceipt)
    assert dataclasses.is_dataclass(GovernedDispatchPlan)
    assert dataclasses.is_dataclass(GovernedOutcome)
    assert dataclasses.is_dataclass(ReadOnlyPassThrough)

    with pytest.raises(TypeError, match="host-owned"):
        BoundOperation()
    with pytest.raises(TypeError, match="host-issued"):
        ApprovalReceipt(
            approval_request_id="fabricated-request",
            approval_message_id="fabricated-message",
            approval_event_id="fabricated-event",
            approved_at="2026-08-01T00:00:00Z",
            evidence_hash="0" * 64,
            grant=fresh_approval.ApprovalGrant(),
        )

    assert tuple(field.name for field in dataclasses.fields(ApprovalReceipt)) == (
        "approval_request_id",
        "approval_message_id",
        "approval_event_id",
        "approved_at",
        "evidence_hash",
        "grant",
    )
    return_annotation = inspect.signature(
        GovernedMCPHostServices.request_fresh_approval
    ).return_annotation
    assert return_annotation in {ApprovalReceipt, "ApprovalReceipt"}

    outcome = GovernedOutcome(
        status="verified",
        summary="safe summary",
        evidence={"nested": ["value"]},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.status = "unknown"  # type: ignore[misc]
    with pytest.raises(TypeError):
        outcome.evidence["nested"] = ()  # type: ignore[index]

    descriptor = MCPToolDescriptor(
        server_name=SERVER,
        raw_tool_name=PREPARE,
        normalized_name=mcp_tool.mcp_prefixed_tool_name(SERVER, PREPARE),
        description="read-only descriptor",
        input_schema={"type": "object", "properties": {}},
        annotations={"readOnlyHint": True},
        meta=None,
        fingerprint="f" * 64,
    )
    decision = ReadOnlyPassThrough(descriptor)
    assert decision.descriptor is descriptor
    assert decision.fingerprint == descriptor.fingerprint
    assert not any(
        callable(getattr(decision, field.name))
        for field in dataclasses.fields(decision)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.fingerprint = "changed"  # type: ignore[misc]


def test_public_protocol_and_installed_sdk_expose_keyword_only_transport_meta():
    sdk_meta = inspect.signature(ClientSession.call_tool).parameters["meta"]
    assert sdk_meta.kind is inspect.Parameter.KEYWORD_ONLY
    assert sdk_meta.default is None

    preflight_meta = inspect.signature(
        GovernedMCPHostServices.preflight_call
    ).parameters["meta"]
    assert preflight_meta.kind is inspect.Parameter.KEYWORD_ONLY
    assert preflight_meta.default is None

    bind_signature = inspect.signature(GovernedMCPHostServices.bind_operation)
    assert bind_signature.parameters["base_meta"].default == {}
    assert (
        bind_signature.parameters["allowed_post_approval_meta_fields"].default
        == ()
    )

    plan_field = next(
        field
        for field in dataclasses.fields(GovernedDispatchPlan)
        if field.name == "post_approval_meta_patch"
    )
    assert plan_field.default_factory() == {}


def test_descriptor_preserves_real_sdk_aliases_and_absence(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_metadata"
    _write_plugin(home, name, _capture_policy_source())
    manager, loaded = _discover_plugin(home, monkeypatch, name)
    assert loaded.enabled is True
    assert manager.get_mcp_governor_attribution(SERVER) == name

    annotations = ToolAnnotations(
        destructiveHint=True,
        idempotentHint=False,
    )
    execute = _tool(
        EXECUTE,
        annotations=annotations,
        meta={"neutralAlias": {"present": True}},
    )
    session_call = AsyncMock()
    _server, _session, registered = _server_with_tools(execute)
    assert mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE) in registered

    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "blocked"
    session_call.assert_not_awaited()
    request = loaded.module.seen[0]
    descriptor = request.descriptor
    assert descriptor.server_name == SERVER
    assert descriptor.raw_tool_name == EXECUTE
    assert descriptor.normalized_name == mcp_tool.mcp_prefixed_tool_name(
        SERVER, EXECUTE
    )
    assert descriptor.input_schema["type"] == execute.inputSchema["type"]
    assert descriptor.input_schema["required"] == tuple(
        execute.inputSchema["required"]
    )
    assert descriptor.input_schema["properties"]["value"] == {
        "type": "string"
    }
    assert descriptor.annotations["readOnlyHint"] is None
    assert descriptor.annotations["destructiveHint"] is True
    assert descriptor.annotations["idempotentHint"] is False
    assert descriptor.annotations["openWorldHint"] is None
    assert descriptor.meta == {"neutralAlias": {"present": True}}
    assert len(descriptor.fingerprint) == 64
    with pytest.raises(TypeError):
        descriptor.input_schema["type"] = "array"
    with pytest.raises(TypeError):
        request.arguments["value"] = "changed"

    missing_name = "neutral_governor_metadata_missing"
    missing_home = tmp_path / "home-missing"
    _write_plugin(missing_home, missing_name, _capture_policy_source())
    manager, loaded = _discover_plugin(missing_home, monkeypatch, missing_name)
    assert manager.get_mcp_governor_attribution(SERVER) == missing_name
    mcp_tool._servers.pop(SERVER, None)
    _server_with_tools(_tool(EXECUTE))
    with _trusted_scope(platform="telegram"):
        registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )
    missing_descriptor = loaded.module.seen[0].descriptor
    assert missing_descriptor.annotations is None
    assert missing_descriptor.meta is None


def test_descriptor_fingerprint_changes_with_every_contract_field():
    from tools.governed_mcp_dispatch import build_mcp_tool_descriptor

    base_tool = _tool(
        EXECUTE,
        annotations=ToolAnnotations(readOnlyHint=True),
        meta={"neutral": "base"},
    )
    descriptors = [
        build_mcp_tool_descriptor(SERVER, "normalized-base", base_tool),
        build_mcp_tool_descriptor("different-server", "normalized-base", base_tool),
        build_mcp_tool_descriptor(
            SERVER,
            "normalized-base",
            Tool(
                name="different-tool",
                description=base_tool.description,
                inputSchema=base_tool.inputSchema,
                annotations=base_tool.annotations,
                _meta=base_tool.meta,
            ),
        ),
        build_mcp_tool_descriptor(SERVER, "normalized-different", base_tool),
        build_mcp_tool_descriptor(
            SERVER,
            "normalized-base",
            Tool(
                name=EXECUTE,
                description="different description",
                inputSchema=base_tool.inputSchema,
                annotations=base_tool.annotations,
                _meta=base_tool.meta,
            ),
        ),
        build_mcp_tool_descriptor(
            SERVER,
            "normalized-base",
            Tool(
                name=EXECUTE,
                description=base_tool.description,
                inputSchema={"type": "object", "properties": {}},
                annotations=base_tool.annotations,
                _meta=base_tool.meta,
            ),
        ),
        build_mcp_tool_descriptor(
            SERVER,
            "normalized-base",
            Tool(
                name=EXECUTE,
                description=base_tool.description,
                inputSchema=base_tool.inputSchema,
                annotations=ToolAnnotations(readOnlyHint=False),
                _meta=base_tool.meta,
            ),
        ),
        build_mcp_tool_descriptor(
            SERVER,
            "normalized-base",
            Tool(
                name=EXECUTE,
                description=base_tool.description,
                inputSchema=base_tool.inputSchema,
                annotations=base_tool.annotations,
                _meta={"neutral": "different"},
            ),
        ),
    ]
    assert len({descriptor.fingerprint for descriptor in descriptors}) == len(
        descriptors
    )


def test_operation_hash_has_fixed_meta_and_patch_allowlist_canonical_vector():
    from tools.governed_mcp_dispatch import _operation_binding_hash

    binding = {
        "server_name": "vector-server",
        "invocation_raw_tool_name": "vector-prepare",
        "invocation_descriptor_hash": "i" * 64,
        "target_raw_tool_name": "vector-execute",
        "target_normalized_name": "mcp__vector_server__vector_execute",
        "target_descriptor_hash": "t" * 64,
        "base_arguments": {"alpha": 1, "nested": {"z": True}},
        "base_meta": {META_FIELD: "BASE_META_SENTINEL"},
        "proposal_hash": "proposal-sentinel",
        "preview": "preview sentinel",
        "display_text": "display sentinel",
        "interaction_hash": "x" * 64,
        "preflight_facts": {
            "arguments": {"probe": "vector"},
            "descriptor_hash": "p" * 64,
            "meta": {META_FIELD: "PREFLIGHT_META_SENTINEL"},
            "normalized_name": "mcp__vector_server__vector_prepare",
            "raw_tool_name": "vector-prepare",
        },
        "allowed_post_approval_patch_fields": ("signature",),
        "allowed_post_approval_meta_fields": (META_FIELD,),
    }
    expected = "34349a3fb394b4cbda7bf629fc7231c820b16a9238aab9fd2b36cfca4e06d491"
    assert _operation_binding_hash(**binding) == expected

    changed_base_meta = {**binding, "base_meta": {META_FIELD: "CHANGED"}}
    changed_argument_allowlist = {
        **binding,
        "allowed_post_approval_patch_fields": ("different",),
    }
    changed_meta_allowlist = {
        **binding,
        "allowed_post_approval_meta_fields": ("transport.different",),
    }
    changed_invocation_descriptor = {
        **binding,
        "invocation_descriptor_hash": "j" * 64,
    }
    changed_target_descriptor = {
        **binding,
        "target_descriptor_hash": "u" * 64,
    }
    assert _operation_binding_hash(**changed_base_meta) != expected
    assert _operation_binding_hash(**changed_argument_allowlist) != expected
    assert _operation_binding_hash(**changed_meta_allowlist) != expected
    assert _operation_binding_hash(**changed_invocation_descriptor) != expected
    assert _operation_binding_hash(**changed_target_descriptor) != expected
    for field, changed_value in (
        ("invocation_raw_tool_name", "different-prepare"),
        ("target_raw_tool_name", "different-execute"),
        ("target_normalized_name", "mcp__vector_server__different_execute"),
        ("base_arguments", {"alpha": 2}),
        ("proposal_hash", "different-proposal"),
        ("preview", "different preview"),
        ("display_text", "different display"),
        ("interaction_hash", "y" * 64),
        ("preflight_facts", {}),
    ):
        assert (
            _operation_binding_hash(**{**binding, field: changed_value})
            != expected
        )


@pytest.mark.parametrize("platform", ["local", "telegram"])
def test_positive_two_phase_flow_has_one_preflight_one_dispatch_and_fresh_approval(
    tmp_path, monkeypatch, caplog, platform
):
    home = tmp_path / f"home-{platform}"
    name = f"neutral_governor_positive_{platform}"
    _write_plugin(home, name, _positive_policy_source())
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    assert loaded.enabled is True

    calls = []

    async def call_tool(raw_name, arguments, *, meta=None):
        calls.append(
            (
                raw_name,
                dict(arguments),
                None if meta is None else dict(meta),
            )
        )
        if raw_name == PREPARE:
            return CallToolResult(
                content=[TextContent(type="text", text="proposal")],
                structuredContent={"proposal": "sentinel"},
                _meta={"neutral": "preflight-meta"},
            )
        assert raw_name == EXECUTE
        return CallToolResult(
            content=[TextContent(type="text", text="sensitive raw response")],
            structuredContent={"verified": True, "receipt": "safe-receipt"},
            _meta={
                "neutral": "dispatch-meta",
                META_FIELD: EXECUTE_META_SENTINEL,
            },
        )

    _server_with_tools(_tool(PREPARE), _tool(EXECUTE), call_tool=call_tool)
    clock = [1000.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    notifications = []
    session_key = (
        "neutral-telegram-session"
        if platform == "telegram"
        else "neutral-desktop-session"
    )
    register_gateway_notify(
        session_key,
        _auto_approve_notifier(coordinator, clock, notifications),
    )

    with _trusted_scope(platform=platform):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, PREPARE),
            {"value": "sentinel"},
        )
    result = json.loads(raw)

    assert result == {
        "dispatch_count": 1,
        "dispatch_started": True,
        "dispatch_tool": EXECUTE,
        "invoked_tool": PREPARE,
        "operation_hash": result["operation_hash"],
        "outcome": {
            "evidence": {"receipt": "safe-receipt"},
            "summary": "neutral provider verified the operation",
        },
        "preflight_completed": True,
        "preflight_count": 1,
        "preflight_started": True,
        "server": SERVER,
        "status": "verified",
    }
    assert len(result["operation_hash"]) == 64
    assert [call[0] for call in calls] == [PREPARE, EXECUTE]
    assert calls[0] == (
        PREPARE,
        {"probe": "sentinel"},
        {META_FIELD: PREPARE_META_SENTINEL},
    )
    assert calls[1][1]["value"] == "sentinel"
    assert len(calls[1][1]["signature"]) == 64
    assert calls[1][1]["signature"] == hmac.new(
        SECRET_VALUE.encode("utf-8"),
        b'{"proposal":"sentinel"}',
        hashlib.sha256,
    ).hexdigest()
    assert set(calls[1][1]) == {"value", "signature"}
    assert calls[1][2] == {
        BOUND_META_FIELD: BOUND_META_SENTINEL,
        META_FIELD: EXECUTE_META_SENTINEL,
    }
    assert "_meta" not in calls[1][1]
    assert META_FIELD not in calls[1][1]
    for host_only_value in (
        PREPARE_META_SENTINEL,
        BOUND_META_SENTINEL,
        EXECUTE_META_SENTINEL,
        SECRET_VALUE,
    ):
        assert host_only_value not in raw
        assert host_only_value not in json.dumps(
            dict(loaded.module.seen[0][0].arguments), sort_keys=True
        )
        assert host_only_value not in json.dumps(notifications, sort_keys=True)
        assert host_only_value not in caplog.text
    assert "sensitive raw response" not in raw
    assert len(notifications) == 1
    assert notifications[0]["fresh_only"] is True
    assert notifications[0]["choices"] == ["once", "deny"]
    assert notifications[0]["allow_session"] is False
    assert notifications[0]["allow_permanent"] is False
    assert notifications[0]["command"] == FULL_PREVIEW
    assert notifications[0]["description"] == FULL_DISPLAY
    assert notifications[0]["command"].endswith(FULL_PREVIEW_TAIL)
    assert notifications[0]["description"].endswith(FULL_DISPLAY_TAIL)
    assert notifications[0]["approval_id"].startswith("fa_")
    assert result["operation_hash"] not in notifications[0]["approval_id"]
    assert "always" not in json.dumps(notifications[0]).lower()
    receipt = loaded.module.saved_receipt
    resolution = coordinator.await_resolution(notifications[0]["approval_id"])
    assert type(receipt) is ApprovalReceipt
    assert receipt.approval_request_id == notifications[0]["approval_id"]
    assert receipt.approval_message_id == "neutral-later-approval-message"
    assert receipt.approval_message_id != "neutral-origin-message"
    assert receipt.approved_at == "1970-01-01T00:16:41Z"
    assert resolution is not None
    assert resolution.evidence is not None
    assert receipt.approval_event_id == resolution.evidence.evidence_hash
    assert receipt.evidence_hash == resolution.evidence.evidence_hash
    assert type(receipt.grant) is fresh_approval.ApprovalGrant
    assert repr(receipt.grant) == "<ApprovalGrant opaque>"
    assert not hasattr(receipt, "__dict__")
    for unsafe_name in (
        "nonce",
        "host_nonce",
        "responder",
        "binding",
        "conversation",
        "subject",
    ):
        assert not hasattr(receipt, unsafe_name)
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.evidence_hash = "changed"  # type: ignore[misc]
    preflight_result = loaded.module.seen[1][1]
    assert preflight_result.structuredContent == {"proposal": "sentinel"}
    assert preflight_result.meta == {"neutral": "preflight-meta"}
    with pytest.raises(TypeError):
        preflight_result.meta["neutral"] = "changed"
    finalizer_request = loaded.module.seen[2][0]
    assert finalizer_request is loaded.module.seen[0][0]
    assert dict(finalizer_request.arguments) == {"value": "sentinel"}
    assert (
        loaded.module.seen[2][2].meta[META_FIELD]
        == EXECUTE_META_SENTINEL
    )


def test_real_sdk_read_only_hint_true_uses_closed_legacy_handler_exactly_once(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_read_only"
    _write_plugin(home, name, _read_only_policy_source())
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    calls = []

    async def legacy_read(raw_name, arguments):
        calls.append((raw_name, dict(arguments)))
        return CallToolResult(
            content=[TextContent(type="text", text="read-safe-result")]
        )

    _server_with_tools(
        _tool(READ, annotations=ToolAnnotations(readOnlyHint=True)),
        call_tool=legacy_read,
    )
    with _trusted_scope(platform="local"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, READ),
            {"value": "read-sentinel"},
        )

    assert json.loads(raw) == {"result": "read-safe-result"}
    assert calls == [(READ, {"value": "read-sentinel"})]
    _request, services = loaded.module.seen[0]
    assert services.preflight_count == 0
    assert services.approval_receipt is None
    assert services.dispatch_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        services.preflight_call(READ, {"value": "detached"})
    assert calls == [(READ, {"value": "read-sentinel"})]


def test_read_only_pass_through_preserves_legacy_retry_path(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_read_retry"
    _write_plugin(home, name, _read_only_policy_source())
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    calls = []

    async def flaky_read(raw_name, arguments):
        calls.append((raw_name, dict(arguments)))
        if len(calls) == 1:
            raise RuntimeError("synthetic auth failure")
        return CallToolResult(
            content=[TextContent(type="text", text="retried-read-result")]
        )

    def retry_once(server_name, exc, call_once, operation):
        assert server_name == SERVER
        assert isinstance(exc, RuntimeError)
        assert operation == f"tools/call {READ}"
        return call_once()

    monkeypatch.setattr(mcp_tool, "_handle_auth_error_and_retry", retry_once)
    _server_with_tools(
        _tool(READ, annotations=ToolAnnotations(readOnlyHint=True)),
        call_tool=flaky_read,
    )
    with _trusted_scope(platform="local"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, READ),
            {"value": "read-sentinel"},
        )

    assert json.loads(raw) == {"result": "retried-read-result"}
    assert calls == [
        (READ, {"value": "read-sentinel"}),
        (READ, {"value": "read-sentinel"}),
    ]


@pytest.mark.parametrize("reconnect_origin", ["manual", "auth", "stdio"])
def test_governed_read_only_rebinds_clean_reconnect_without_list_changed(
    tmp_path, monkeypatch, reconnect_origin
):
    home = tmp_path / f"home-{reconnect_origin}"
    name = f"neutral_governor_reconnect_{reconnect_origin}"
    _write_plugin(home, name, _read_only_policy_source())
    _discover_plugin(home, monkeypatch, name)

    original_call = AsyncMock()
    server, original_session, registered = _server_with_tools(
        _tool(READ, annotations=ToolAnnotations(readOnlyHint=True)),
        call_tool=original_call,
    )
    replacement_call = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text=f"{reconnect_origin}-reconnected")]
        )
    )
    server.session = SimpleNamespace(call_tool=replacement_call)

    server._register_discovered_tools_if_needed()

    with _trusted_scope(platform="local"):
        raw = registry.dispatch(
            registered[0],
            {"value": "read-sentinel"},
        )

    assert json.loads(raw) == {"result": f"{reconnect_origin}-reconnected"}
    original_call.assert_not_awaited()
    replacement_call.assert_awaited_once_with(
        READ, arguments={"value": "read-sentinel"}
    )


@pytest.mark.parametrize("descriptor_drift", ["description", "duplicate"])
def test_governed_reconnect_descriptor_drift_blocks_without_sdk_call(
    tmp_path, monkeypatch, descriptor_drift
):
    home = tmp_path / f"home-{descriptor_drift}"
    name = f"neutral_governor_reconnect_drift_{descriptor_drift}"
    _write_plugin(home, name, _read_only_policy_source())
    _discover_plugin(home, monkeypatch, name)

    original_call = AsyncMock()
    server, original_session, registered = _server_with_tools(
        _tool(READ, annotations=ToolAnnotations(readOnlyHint=True)),
        call_tool=original_call,
    )
    replacement_call = AsyncMock()
    server.session = SimpleNamespace(call_tool=replacement_call)
    if descriptor_drift == "description":
        server._tools = [
            _tool(
                READ,
                annotations=ToolAnnotations(readOnlyHint=True),
                meta={"drift": "changed"},
            )
        ]
    else:
        server._tools = [
            _tool(READ, annotations=ToolAnnotations(readOnlyHint=True)),
            _tool(READ, annotations=ToolAnnotations(readOnlyHint=True)),
        ]

    server._register_discovered_tools_if_needed()

    with _trusted_scope(platform="local"):
        result = json.loads(
            registry.dispatch(
                registered[0],
                {"value": "read-sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["dispatch_count"] == 0
    assert result["dispatch_started"] is False
    original_call.assert_not_awaited()
    replacement_call.assert_not_awaited()
    assert original_session is not server.session


@pytest.mark.parametrize(
    "annotation_value",
    [pytest.param(None, id="missing"), False, "true", 1],
)
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_read_only_pass_through_requires_exact_boolean_true_annotation(
    tmp_path, monkeypatch, annotation_value
):
    home = tmp_path / f"home-{annotation_value!s}"
    name = f"neutral_governor_non_read_{type(annotation_value).__name__}"
    _write_plugin(home, name, _read_only_policy_source())
    _discover_plugin(home, monkeypatch, name)
    annotations = None
    if annotation_value is not None:
        annotations = ToolAnnotations.model_construct(
            readOnlyHint=annotation_value
        )
    call_tool = AsyncMock()
    if type(annotation_value) in {str, int}:
        with pytest.warns(UserWarning, match="Expected `bool`"):
            _server_with_tools(
                _tool(READ, annotations=annotations), call_tool=call_tool
            )
    else:
        _server_with_tools(
            _tool(READ, annotations=annotations), call_tool=call_tool
        )

    with _trusted_scope(platform="local"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, READ),
                {"value": "read-sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["invoked_tool"] == READ
    assert result["dispatch_tool"] is None
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


@pytest.mark.parametrize("decision", ["malformed", "foreign"])
def test_malformed_or_foreign_read_only_decision_blocks_without_provider_call(
    tmp_path, monkeypatch, decision
):
    home = tmp_path / f"home-{decision}"
    name = f"neutral_governor_read_{decision}"
    _write_plugin(home, name, _read_only_policy_source(decision=decision))
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(
        _tool(READ, annotations=ToolAnnotations(readOnlyHint=True)),
        call_tool=call_tool,
    )

    with _trusted_scope(platform="local"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, READ),
                {"value": "read-sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["invoked_tool"] == READ
    assert result["dispatch_tool"] is None
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "drift",
    ["metadata", "session", "duplicate", "absent", "ambiguous"],
)
def test_distinct_target_drift_blocks_after_prepare_without_target_dispatch(
    tmp_path, monkeypatch, drift
):
    home = tmp_path / f"home-{drift}"
    name = f"neutral_governor_target_drift_{drift}"
    _write_plugin(home, name, _positive_policy_source())
    _discover_plugin(home, monkeypatch, name)
    calls = []

    async def call_tool(raw_name, arguments, *, meta=None):
        calls.append((raw_name, dict(arguments), meta))
        assert raw_name == PREPARE
        return CallToolResult(
            content=[TextContent(type="text", text="proposal")],
            structuredContent={"proposal": "sentinel"},
        )

    server, original_session, _registered = _server_with_tools(
        _tool(PREPARE),
        _tool(EXECUTE),
        call_tool=call_tool,
    )
    replacement_call = AsyncMock()
    clock = [1000.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    auto_approve = _auto_approve_notifier(coordinator, clock, [])

    def notify(data):
        if drift == "metadata":
            server._tools = [
                _tool(PREPARE),
                Tool(
                    name=EXECUTE,
                    description="drifted target description",
                    inputSchema=_tool(EXECUTE).inputSchema,
                ),
            ]
        elif drift == "session":
            server.session = SimpleNamespace(call_tool=replacement_call)
        elif drift == "duplicate":
            server._tools.append(_tool(EXECUTE))
        elif drift == "absent":
            server._tools = [_tool(PREPARE)]
        else:
            server._tools.append(_tool(EXECUTE.replace("-", "_")))
        auto_approve(data)

    register_gateway_notify("neutral-desktop-session", notify)
    with _trusted_scope(platform="local"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, PREPARE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["invoked_tool"] == PREPARE
    assert result["dispatch_tool"] == EXECUTE
    assert result["preflight_completed"] is True
    assert result["preflight_count"] == 1
    assert result["dispatch_started"] is False
    assert result["dispatch_count"] == 0
    assert [call[0] for call in calls] == [PREPARE]
    assert original_session.call_tool is call_tool
    replacement_call.assert_not_awaited()


@pytest.mark.parametrize(
    ("target_kind", "target_raw_name"),
    [
        ("absent", "absent-operation"),
        ("other-server", "remote-operation"),
        ("generated-utility", "list_resources"),
    ],
)
def test_target_must_be_one_registered_raw_tool_on_current_server(
    tmp_path, monkeypatch, target_kind, target_raw_name
):
    home = tmp_path / f"home-{target_kind}"
    name = f"neutral_governor_invalid_target_{target_kind}"
    _write_plugin(home, name, _invalid_target_policy_source(target_raw_name))
    _discover_plugin(home, monkeypatch, name)
    calls = []

    async def call_tool(raw_name, arguments, *, meta=None):
        calls.append(raw_name)
        return CallToolResult(
            content=[TextContent(type="text", text="proposal")],
            structuredContent={"proposal": "sentinel"},
        )

    server, _session, _registered = _server_with_tools(
        _tool(PREPARE),
        call_tool=call_tool,
    )
    if target_kind == "other-server":
        other_name = "neutral-other-server"
        other = mcp_tool.MCPServerTask(other_name)
        other.session = SimpleNamespace(call_tool=AsyncMock())
        other._tools = [_tool(target_raw_name)]
        other._config = {}
        other.tool_timeout = 3
        with mcp_tool._lock:
            mcp_tool._servers[other_name] = other
        other._registered_tool_names = mcp_tool._register_server_tools(
            other_name, other, {}
        )
    elif target_kind == "generated-utility":
        server.session.list_resources = AsyncMock()
        server._registered_tool_names = mcp_tool._register_server_tools(
            SERVER, server, {}
        )

    with _trusted_scope(platform="local"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, PREPARE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["invoked_tool"] == PREPARE
    assert result["dispatch_tool"] == target_raw_name
    assert result["preflight_completed"] is True
    assert result["preflight_count"] == 1
    assert result["dispatch_count"] == 0
    assert calls == [PREPARE]


def test_plugin_policy_blocks_direct_target_invocation_without_host_classification(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_direct_target"
    _write_plugin(home, name, _direct_target_block_policy_source())
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(PREPARE), _tool(EXECUTE), call_tool=call_tool)

    with _trusted_scope(platform="local"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "direct-sentinel"},
            )
        )

    assert loaded.module.seen == [EXECUTE]
    assert result["status"] == "blocked"
    assert result["invoked_tool"] == EXECUTE
    assert result["dispatch_tool"] is None
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


def test_legacy_non_governed_mcp_keeps_two_argument_sdk_call_behavior():
    plugin_module._plugin_manager = None
    calls = []

    async def legacy_call_tool(raw_name, arguments):
        calls.append((raw_name, dict(arguments)))
        return CallToolResult(
            content=[TextContent(type="text", text="legacy-safe-result")]
        )

    _server_with_tools(_tool(EXECUTE), call_tool=legacy_call_tool)
    raw = registry.dispatch(
        mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
        {"value": "legacy-sentinel"},
    )

    assert json.loads(raw) == {"result": "legacy-safe-result"}
    assert calls == [(EXECUTE, {"value": "legacy-sentinel"})]


def test_no_trusted_provenance_blocks_before_preflight_or_dispatch(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_no_provenance"
    _write_plugin(home, name, _positive_policy_source())
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)

    result = json.loads(
        registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )
    )

    assert result["status"] == "blocked"
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


def test_model_visible_meta_argument_is_rejected_before_policy_or_sdk_call(
    tmp_path, monkeypatch
):
    home = tmp_path / "model-meta"
    name = "neutral_governor_model_meta"
    _write_plugin(home, name, _positive_policy_source())
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)

    with _trusted_scope(platform="telegram"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {
                "value": "sentinel",
                "_meta": {META_FIELD: "MODEL_META_SENTINEL"},
            },
        )
    result = json.loads(raw)

    assert result["status"] == "blocked"
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    assert loaded.module.seen == []
    assert "MODEL_META_SENTINEL" not in raw
    call_tool.assert_not_awaited()


def test_mismatched_trusted_provenance_blocks_before_provider_work(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_mismatched_provenance"
    _write_plugin(home, name, _positive_policy_source())
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)

    with _trusted_scope(platform="telegram"):
        mismatch_tokens = set_session_vars(chat_id="different-chat")
        try:
            result = json.loads(
                registry.dispatch(
                    mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                    {"value": "sentinel"},
                )
            )
        finally:
            clear_session_vars(mismatch_tokens)

    assert result["status"] == "blocked"
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


def test_expired_trusted_provenance_blocks_before_provider_work(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_expired_provenance"
    _write_plugin(home, name, _positive_policy_source())
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)
    interaction = _issue_trusted_interaction(
        surface="gateway",
        platform="telegram",
        actor_id="neutral-actor",
        channel_id="neutral-chat",
        chat_id="neutral-chat",
        thread_id="neutral-thread",
        session_key="neutral-telegram-session",
        session_id="neutral-telegram-session",
        message_id="neutral-origin-message",
        received_at=time.time(),
        profile_id="default",
    )
    _publish_trusted_interaction(interaction, ttl_seconds=0.001)
    token = _bind_trusted_interaction(interaction)
    session_tokens = set_session_vars(
        platform="telegram",
        source="gateway",
        chat_id=interaction.chat_id,
        thread_id=interaction.thread_id or "",
        user_id=interaction.actor_id,
        session_key=interaction.session_key,
        session_id=interaction.session_id,
        message_id=interaction.message_id,
        profile=interaction.profile_id,
    )
    try:
        deadline = time.monotonic() + 1.0
        while (
            get_current_trusted_interaction() is not None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert get_current_trusted_interaction() is None
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )
    finally:
        clear_session_vars(session_tokens)
        _reset_trusted_interaction(token)
        _clear_trusted_interactions(interaction=interaction)

    assert result["status"] == "blocked"
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


def test_descriptor_drift_and_session_swap_block_without_dispatch(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_drift"
    source = f'''from tools.governed_mcp import GovernedDispatchPlan

class Policy:
    def prepare(self, request, services):
        operation = services.bind_operation(
            {EXECUTE!r}, request.arguments,
            proposal_hash="sentinel-proposal",
            preview="neutral preview",
            display_text="neutral display",
            base_meta={{{META_FIELD!r}: "DRIFT_META_SENTINEL"}},
        )
        return GovernedDispatchPlan(operation, None, {{}})
    def finalize(self, request, operation, result):
        raise AssertionError("must not finalize")
def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''
    _write_plugin(home, name, source)
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    server, session, _registered = _server_with_tools(
        _tool(EXECUTE), call_tool=call_tool
    )
    entry = registry.get_entry(mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE))
    assert entry is not None

    server._tools = [
        Tool(
            name=EXECUTE,
            description="drifted description",
            inputSchema=_tool(EXECUTE).inputSchema,
        )
    ]
    with _trusted_scope(platform="telegram"):
        drift = json.loads(entry.handler({"value": "sentinel"}))
    assert drift["status"] == "blocked"
    assert drift["preflight_count"] == 0
    assert drift["dispatch_count"] == 0
    assert "DRIFT_META_SENTINEL" not in json.dumps(drift)
    call_tool.assert_not_awaited()

    server._tools = [_tool(EXECUTE)]
    server.session = SimpleNamespace(call_tool=call_tool)
    with _trusted_scope(platform="telegram"):
        swap = json.loads(entry.handler({"value": "sentinel"}))
    assert swap["status"] == "blocked"
    assert swap["preflight_count"] == 0
    assert swap["dispatch_count"] == 0
    assert "DRIFT_META_SENTINEL" not in json.dumps(swap)
    call_tool.assert_not_awaited()
    assert session is not server.session


def test_governor_lookup_failure_blocks_without_sdk_call(monkeypatch, caplog):
    call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="provider must not run")]
        )
    )
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    entry = registry.get_entry(mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE))
    assert entry is not None

    def fail_lookup(_server_name: str):
        raise RuntimeError("SENSITIVE_LOOKUP_DETAIL_DO_NOT_LOG")

    monkeypatch.setattr(plugin_module, "get_mcp_governor", fail_lookup)

    raw = entry.handler({"value": "sentinel"})
    result = json.loads(raw)

    assert result["status"] == "blocked"
    assert result["preflight_started"] is False
    assert result["preflight_count"] == 0
    assert result["dispatch_started"] is False
    assert result["dispatch_count"] == 0
    assert "SENSITIVE_LOOKUP_DETAIL_DO_NOT_LOG" not in raw
    assert "SENSITIVE_LOOKUP_DETAIL_DO_NOT_LOG" not in caplog.text
    call_tool.assert_not_awaited()


def test_governor_import_failure_blocks_without_sdk_call(monkeypatch, caplog):
    call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="provider must not run")]
        )
    )
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    entry = registry.get_entry(mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE))
    assert entry is not None
    real_import = builtins.__import__

    def fail_plugin_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hermes_cli.plugins" and "get_mcp_governor" in fromlist:
            raise ImportError("SENSITIVE_IMPORT_DETAIL_DO_NOT_LOG")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_plugin_import)

    raw = entry.handler({"value": "sentinel"})
    result = json.loads(raw)

    assert result["status"] == "blocked"
    assert result["preflight_started"] is False
    assert result["preflight_count"] == 0
    assert result["dispatch_started"] is False
    assert result["dispatch_count"] == 0
    assert "SENSITIVE_IMPORT_DETAIL_DO_NOT_LOG" not in raw
    assert "SENSITIVE_IMPORT_DETAIL_DO_NOT_LOG" not in caplog.text
    call_tool.assert_not_awaited()


@pytest.mark.parametrize("drift_kind", ["session", "descriptor"])
def test_preflight_rechecks_session_and_descriptor_drift_under_rpc_lock(
    tmp_path, monkeypatch, drift_kind
):
    home = tmp_path / f"preflight-{drift_kind}-drift"
    name = f"neutral_governor_preflight_{drift_kind}_drift"
    _write_plugin(home, name, _positive_policy_source())
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    server, _session, _registered = _server_with_tools(
        _tool(EXECUTE),
        _tool(PREFLIGHT),
        call_tool=call_tool,
    )
    replacement_call = AsyncMock()

    class DriftOnEnter:
        async def __aenter__(self):
            if drift_kind == "session":
                server.session = SimpleNamespace(call_tool=replacement_call)
            else:
                drifted_execute = Tool(
                    name=EXECUTE,
                    description="drifted under lock",
                    inputSchema=_tool(EXECUTE).inputSchema,
                )
                server._tools = [drifted_execute, _tool(PREFLIGHT)]

        async def __aexit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    server._rpc_lock = DriftOnEnter()
    with _trusted_scope(platform="telegram"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )
    result = json.loads(raw)

    assert result["status"] == "blocked"
    assert result["preflight_started"] is False
    assert result["preflight_count"] == 0
    assert result["dispatch_started"] is False
    assert result["dispatch_count"] == 0
    assert PREPARE_META_SENTINEL not in raw
    call_tool.assert_not_awaited()
    replacement_call.assert_not_awaited()


@pytest.mark.parametrize(
    "server_name",
    ["", " ", "neutral-*", "neutral-?", "neutral-[x]"],
)
def test_registration_rejects_empty_or_wildcard_servers(
    tmp_path, monkeypatch, server_name
):
    home = tmp_path / f"invalid-{abs(hash(server_name))}"
    name = "neutral_governor_invalid_server"
    source = f'''class Policy:
    def prepare(self, request, services):
        return None
    def finalize(self, request, operation, result):
        return None
def register(ctx):
    ctx.register_mcp_governor({server_name!r}, Policy())
'''
    _write_plugin(home, name, source, server_name=server_name)
    manager, loaded = _discover_plugin(home, monkeypatch, name)
    assert loaded.enabled is False
    assert manager.get_mcp_governor_attribution(SERVER) is None


def test_governor_requires_explicit_operator_opt_in_and_rolls_back_on_failure(
    tmp_path, monkeypatch
):
    denied_home = tmp_path / "denied"
    denied_name = "neutral_governor_denied"
    _write_plugin(
        denied_home,
        denied_name,
        _capture_policy_source(),
        allow_governance=False,
    )
    manager, loaded = _discover_plugin(denied_home, monkeypatch, denied_name)
    assert loaded.enabled is False
    assert manager.get_mcp_governor_attribution(SERVER) is None

    failed_home = tmp_path / "failed"
    failed_name = "neutral_governor_failed"
    _write_plugin(
        failed_home,
        failed_name,
        _capture_policy_source()
        + "\nraise RuntimeError('registration fails after governor')\n",
    )
    manager, loaded = _discover_plugin(failed_home, monkeypatch, failed_name)
    assert loaded.enabled is False
    assert manager.get_mcp_governor_attribution(SERVER) is None


def test_two_plugins_cannot_govern_the_same_exact_server(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    first = "neutral_governor_first"
    second = "neutral_governor_second"
    for name in (first, second):
        plugin_dir = home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            yaml.safe_dump({"name": name, "version": "1.0"}),
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            _capture_policy_source(), encoding="utf-8"
        )
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "enabled": [first, second],
                    "entries": {
                        first: {"allow_mcp_governance": True},
                        second: {"allow_mcp_governance": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    plugin_module._plugin_manager = manager
    manager.discover_and_load()

    enabled = [
        name for name in (first, second) if manager._plugins[name].enabled
    ]
    assert enabled == [first]
    assert manager.get_mcp_governor_attribution(SERVER) == first


def _prepare_case_source(body: str) -> str:
    return f'''seen_services = []

class Policy:
    def prepare(self, request, services):
        seen_services.append(services)
{body}

    def finalize(self, request, operation, result):
        raise AssertionError("finalize must not run")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


def test_bound_base_arguments_reject_top_level_meta_without_sdk_call(
    tmp_path, monkeypatch
):
    home = tmp_path / "bound-argument-meta"
    name = "neutral_governor_bound_argument_meta"
    body = f'''        services.bind_operation(
            {EXECUTE!r},
            {{"value": "sentinel", "_meta": {{{META_FIELD!r}: "FORBIDDEN"}}}},
            proposal_hash="neutral-proposal",
            preview="neutral preview",
            display_text="neutral display",
        )'''
    _write_plugin(home, name, _prepare_case_source(body))
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)

    with _trusted_scope(platform="telegram"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )
    result = json.loads(raw)

    assert result["status"] == "blocked"
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    assert "FORBIDDEN" not in raw
    call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "meta_expr",
    [
        "[]",
        "{1: 'NON_STRING_KEY_SENTINEL'}",
        "{'bad': object()}",
        "{'nan': float('nan')}",
        "{'oversize': 'x' * (1024 * 1024 + 1)}",
    ],
    ids=["not-mapping", "non-string-key", "non-json", "nan", "oversize"],
)
def test_malformed_or_oversize_preflight_meta_blocks_before_sdk_call(
    tmp_path, monkeypatch, meta_expr
):
    home = tmp_path / "preflight-meta"
    name = f"neutral_governor_preflight_meta_{meta_expr[:4].encode().hex()}"
    body = f'''        services.preflight_call(
            {PREFLIGHT!r},
            {{"probe": "sentinel"}},
            meta={meta_expr},
        )'''
    _write_plugin(home, name, _prepare_case_source(body))
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)

    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    ("keyword", "value_expr"),
    [
        ("base_meta", "[]"),
        ("base_meta", "{1: 'NON_STRING_KEY_SENTINEL'}"),
        ("base_meta", "{'bad': object()}"),
        ("base_meta", "{'oversize': 'x' * (1024 * 1024 + 1)}"),
        ("allowed_post_approval_meta_fields", "('',)"),
        ("allowed_post_approval_meta_fields", "('duplicate', 'duplicate')"),
        ("allowed_post_approval_meta_fields", "('x' * 257,)"),
        ("preview", "'x' * 64_001"),
        ("display_text", "'x' * 64_001"),
        (
            "allowed_post_approval_meta_fields",
            "tuple(f'field-{{index}}' for index in range(129))",
        ),
    ],
    ids=[
        "base-not-mapping",
        "base-non-string-key",
        "base-non-json",
        "base-oversize",
        "empty-field",
        "duplicate-field",
        "oversize-field",
        "oversize-preview",
        "oversize-display",
        "too-many-fields",
    ],
)
def test_bound_transport_meta_and_allowlist_are_strictly_validated(
    tmp_path, monkeypatch, keyword, value_expr
):
    home = tmp_path / f"bound-meta-{keyword}-{len(value_expr)}"
    name = f"neutral_governor_bound_meta_{len(value_expr)}_{keyword[-4:]}"
    preview_expr = value_expr if keyword == "preview" else repr("neutral preview")
    display_expr = value_expr if keyword == "display_text" else repr("neutral display")
    extra_keyword = "" if keyword in {"preview", "display_text"} else f"{keyword}={value_expr},"
    body = f'''        services.bind_operation(
            {EXECUTE!r},
            request.arguments,
            proposal_hash="neutral-proposal",
            preview={preview_expr},
            display_text={display_expr},
            {extra_keyword}
        )'''
    _write_plugin(home, name, _prepare_case_source(body))
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)

    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["preflight_count"] == 0
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    ("case_name", "body", "expected_preflight"),
    [
        (
            "policy_exception",
            '        raise RuntimeError("prepare failed")',
            0,
        ),
        (
            "after_preflight",
            f'''        services.preflight_call({PREFLIGHT!r}, {{"probe": "one"}})
        raise RuntimeError("prepare failed after preflight")''',
            1,
        ),
        (
            "second_preflight",
            f'''        services.preflight_call(
            {PREFLIGHT!r}, {{"probe": "one"}},
            meta={{{META_FIELD!r}: "FIRST_META_SENTINEL"}},
        )
        services.preflight_call(
            {PREFLIGHT!r}, {{"probe": "two"}},
            meta={{{META_FIELD!r}: "SECOND_META_SENTINEL"}},
        )''',
            1,
        ),
        (
            "undeclared_hmac",
            '''        services.hmac_sha256(
            "UNDECLARED_NEUTRAL_SECRET", "{}"
        )''',
            0,
        ),
        (
            "malformed_plan",
            '        return {"not": "a governed plan"}',
            0,
        ),
    ],
)
def test_prepare_failures_block_with_zero_dispatch_and_no_retry(
    tmp_path,
    monkeypatch,
    case_name,
    body,
    expected_preflight,
):
    home = tmp_path / case_name
    name = f"neutral_governor_{case_name}"
    _write_plugin(home, name, _prepare_case_source(body))
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    calls = []

    async def call_tool(raw_name, arguments, *, meta=None):
        del meta
        calls.append((raw_name, dict(arguments)))
        return CallToolResult(
            content=[TextContent(type="text", text="preflight complete")],
            structuredContent={"proposal": "sentinel"},
        )

    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)
    with _trusted_scope(platform="telegram"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )
    result = json.loads(raw)

    assert result["status"] == "blocked"
    assert result["dispatch_count"] == 0
    assert result["dispatch_started"] is False
    assert result["preflight_count"] == expected_preflight
    assert len(calls) == expected_preflight
    assert SECRET_VALUE not in raw
    if case_name in {"after_preflight", "second_preflight"}:
        assert result["preflight_completed"] is True
    if case_name == "undeclared_hmac":
        assert "UNDECLARED_NEUTRAL_SECRET" not in raw
    assert loaded.enabled is True


def test_capability_is_detached_and_closed_after_policy_returns(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_detached"
    _write_plugin(
        home,
        name,
        _prepare_case_source('        raise RuntimeError("stop")'),
    )
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)
    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )
    assert result["status"] == "blocked"

    services = loaded.module.seen_services[0]
    with pytest.raises(RuntimeError, match="closed"):
        services.preflight_call(
            PREFLIGHT,
            {"probe": "detached"},
            meta={META_FIELD: "DETACHED_META_SENTINEL"},
        )
    call_tool.assert_not_awaited()


@pytest.mark.parametrize("secret_value", [None, ""])
def test_declared_hmac_secret_must_be_materialized_and_non_empty(
    tmp_path, monkeypatch, secret_value
):
    home = tmp_path / ("missing" if secret_value is None else "empty")
    name = "neutral_governor_missing_hmac_material"
    body = f'''        services.hmac_sha256(
            {SECRET_NAME!r}, "{{}}"
        )'''
    _write_plugin(home, name, _prepare_case_source(body))
    _discover_plugin(home, monkeypatch, name)
    if secret_value is None:
        monkeypatch.delenv(SECRET_NAME, raising=False)
    else:
        monkeypatch.setenv(SECRET_NAME, secret_value)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)

    with _trusted_scope(platform="telegram"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )
    result = json.loads(raw)
    assert result["status"] == "blocked"
    assert result["dispatch_count"] == 0
    assert SECRET_VALUE not in raw
    call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("preflight provider error"),
        TimeoutError("preflight timeout"),
        InterruptedError("preflight interrupted"),
        KeyboardInterrupt("preflight keyboard interrupt"),
        asyncio.CancelledError(),
    ],
    ids=["error", "timeout", "interrupt", "keyboard-interrupt", "cancel"],
)
def test_preflight_failure_after_sdk_call_begins_is_unknown_and_never_retried(
    tmp_path, monkeypatch, failure
):
    home = tmp_path / f"home-{type(failure).__name__}"
    name = f"neutral_governor_preflight_{type(failure).__name__.lower()}"
    body = f'''        services.preflight_call(
            {PREFLIGHT!r}, {{"probe": "sentinel"}}
        )'''
    _write_plugin(home, name, _prepare_case_source(body))
    _discover_plugin(home, monkeypatch, name)
    calls = []

    async def call_tool(raw_name, arguments, *, meta=None):
        del meta
        calls.append((raw_name, dict(arguments)))
        raise failure

    _server_with_tools(_tool(EXECUTE), _tool(PREFLIGHT), call_tool=call_tool)
    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "unknown"
    assert result["preflight_started"] is True
    assert result["preflight_completed"] is False
    assert result["preflight_count"] == 1
    assert result["dispatch_started"] is False
    assert result["dispatch_count"] == 0
    assert len(calls) == 1


def _approval_policy_source(
    *,
    patch: str = "{}",
    meta_patch: str = "{}",
    receipt_expr: str = "receipt",
    allowed_patch_fields: str = "()",
    allowed_meta_fields: str = "()",
) -> str:
    return f'''import dataclasses
from tools.fresh_approval import ApprovalGrant
from tools.governed_mcp import ApprovalReceipt, GovernedDispatchPlan, GovernedOutcome

saved_receipt = None

def clone_receipt(receipt):
    clone = object.__new__(ApprovalReceipt)
    for receipt_field in dataclasses.fields(ApprovalReceipt):
        object.__setattr__(clone, receipt_field.name, getattr(receipt, receipt_field.name))
    return clone

def fabricate_receipt():
    receipt = object.__new__(ApprovalReceipt)
    object.__setattr__(receipt, "approval_request_id", "fabricated-request")
    object.__setattr__(receipt, "approval_message_id", "fabricated-message")
    object.__setattr__(receipt, "approval_event_id", "fabricated-event")
    object.__setattr__(receipt, "approved_at", "2026-08-01T00:00:00Z")
    object.__setattr__(receipt, "evidence_hash", "0" * 64)
    object.__setattr__(receipt, "grant", ApprovalGrant())
    return receipt

class Policy:
    def prepare(self, request, services):
        operation = services.bind_operation(
            {EXECUTE!r}, request.arguments,
            proposal_hash="neutral-proposal-hash",
            preview="neutral preview",
            display_text="neutral display",
            base_meta={{}},
            allowed_post_approval_patch_fields={allowed_patch_fields},
            allowed_post_approval_meta_fields={allowed_meta_fields},
        )
        receipt = services.request_fresh_approval(operation)
        global saved_receipt
        saved_receipt = receipt
        return GovernedDispatchPlan(
            operation,
            {receipt_expr},
            {patch},
            post_approval_meta_patch={meta_patch},
        )

    def finalize(self, request, operation, result):
        return GovernedOutcome("verified", "verified", {{"receipt": "safe"}})

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


def _approval_notifier_for_mode(coordinator, clock, mode):
    def notify(data):
        request = coordinator.get_request(data["approval_id"])
        binding = request.subject.conversation
        if mode == "timeout":
            return
        if mode == "expiry":
            clock[0] += 301
        else:
            clock[0] += 1
        if mode == "binding_mismatch":
            binding = dataclasses.replace(binding, chat_id="different-chat")
        inbound_id = (
            request.subject.origin_message_id
            if mode == "initial_message"
            else "later-approval-message"
        )
        responder = coordinator._mint_authenticated_responder_for_host(
            assurance="telegram_authenticated",
            principal_id=binding.actor_id,
            binding=binding,
            inbound_id=inbound_id,
        )
        coordinator.resolve(
            request.approval_id,
            decision="deny" if mode == "deny" else "approve_once",
            responder=responder,
        )
        if mode == "receipt_expiry":
            clock[0] += 301

    return notify


@pytest.mark.parametrize(
    "mode",
    [
        "missing_notifier",
        "deny",
        "timeout",
        "expiry",
        "initial_message",
        "binding_mismatch",
        "receipt_expiry",
    ],
)
def test_fresh_approval_failures_block_before_dispatch(
    tmp_path, monkeypatch, mode
):
    home = tmp_path / mode
    name = f"neutral_governor_approval_{mode}"
    _write_plugin(home, name, _approval_policy_source())
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    clock = [2000.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    if mode != "missing_notifier":
        register_gateway_notify(
            "neutral-telegram-session",
            _approval_notifier_for_mode(coordinator, clock, mode),
        )
    if mode == "timeout":
        monkeypatch.setattr(
            "tools.governed_mcp_dispatch._FRESH_APPROVAL_TTL_SECONDS",
            0.01,
        )

    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["dispatch_count"] == 0
    assert result["dispatch_started"] is False
    call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    (
        "case_name",
        "patch",
        "meta_patch",
        "receipt_expr",
        "allowed_patch_fields",
        "allowed_meta_fields",
    ),
    [
        (
            "undeclared_patch",
            '{"not_allowed": "sentinel"}',
            "{}",
            "receipt",
            "()",
            "()",
        ),
        ("fabricated_receipt", "{}", "{}", "fabricate_receipt()", "()", "()"),
        ("cloned_receipt", "{}", "{}", "clone_receipt(receipt)", "()", "()"),
        (
            "altered_receipt",
            "{}",
            "{}",
            "(object.__setattr__(receipt, 'evidence_hash', '0' * 64) or receipt)",
            "()",
            "()",
        ),
        (
            "undeclared_meta_patch",
            "{}",
            f'{{{META_FIELD!r}: "UNDECLARED_META_SENTINEL"}}',
            "receipt",
            "()",
            "()",
        ),
        (
            "meta_patch_in_argument_patch",
            f'{{{META_FIELD!r}: "WRONG_CHANNEL_SENTINEL"}}',
            "{}",
            "receipt",
            "()",
            f"({META_FIELD!r},)",
        ),
        (
            "argument_patch_in_meta_patch",
            "{}",
            '{"signature": "WRONG_CHANNEL_SENTINEL"}',
            "receipt",
            "('signature',)",
            "()",
        ),
        (
            "overlapping_patches",
            '{"shared": "ARGUMENT_SENTINEL"}',
            '{"shared": "META_SENTINEL"}',
            "receipt",
            "('shared',)",
            "('shared',)",
        ),
        (
            "top_level_meta_argument_patch",
            '{"_meta": {"forbidden": "PATCH_META_SENTINEL"}}',
            "{}",
            "receipt",
            "('_meta',)",
            "()",
        ),
        (
            "malformed_meta_patch",
            "{}",
            "[]",
            "receipt",
            "()",
            "()",
        ),
        (
            "oversize_meta_patch",
            "{}",
            "{'oversize': 'x' * (1024 * 1024 + 1)}",
            "receipt",
            "()",
            "('oversize',)",
        ),
    ],
)
def test_malformed_or_foreign_approved_plan_never_dispatches(
    tmp_path,
    monkeypatch,
    case_name,
    patch,
    meta_patch,
    receipt_expr,
    allowed_patch_fields,
    allowed_meta_fields,
):
    home = tmp_path / case_name
    name = f"neutral_governor_plan_{case_name}"
    _write_plugin(
        home,
        name,
        _approval_policy_source(
            patch=patch,
            meta_patch=meta_patch,
            receipt_expr=receipt_expr,
            allowed_patch_fields=allowed_patch_fields,
            allowed_meta_fields=allowed_meta_fields,
        ),
    )
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    clock = [3000.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    notifications = []
    register_gateway_notify(
        "neutral-telegram-session",
        _auto_approve_notifier(coordinator, clock, notifications),
    )

    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )
    assert result["status"] == "blocked"
    assert result["dispatch_count"] == 0
    assert result["dispatch_started"] is False
    for sentinel in (
        "UNDECLARED_META_SENTINEL",
        "WRONG_CHANNEL_SENTINEL",
        "ARGUMENT_SENTINEL",
        "META_SENTINEL",
        "PATCH_META_SENTINEL",
    ):
        assert sentinel not in json.dumps(result)
    call_tool.assert_not_awaited()
    if notifications:
        request = coordinator.get_request(notifications[0]["approval_id"])
        assert coordinator.consume(
            loaded.module.saved_receipt.grant,
            request.subject,
        )
    else:
        assert loaded.module.saved_receipt is None


def test_foreign_replayed_receipt_from_prior_operation_blocks_before_dispatch(
    tmp_path, monkeypatch
):
    home = tmp_path / "foreign-replayed-receipt"
    name = "neutral_governor_foreign_replayed_receipt"
    source = f'''from tools.governed_mcp import GovernedDispatchPlan, GovernedOutcome

saved_receipt = None

class Policy:
    def prepare(self, request, services):
        operation = services.bind_operation(
            {EXECUTE!r}, request.arguments,
            proposal_hash="neutral-proposal-hash",
            preview="neutral preview",
            display_text="neutral display",
        )
        receipt = services.request_fresh_approval(operation)
        global saved_receipt
        if saved_receipt is None:
            saved_receipt = receipt
            plan_receipt = receipt
        else:
            plan_receipt = saved_receipt
        return GovernedDispatchPlan(operation, plan_receipt, {{}})

    def finalize(self, request, operation, result):
        return GovernedOutcome("verified", "verified")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''
    _write_plugin(home, name, source)
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="ok")]
        )
    )
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    clock = [3250.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    notifications = []
    register_gateway_notify(
        "neutral-telegram-session",
        _auto_approve_notifier(coordinator, clock, notifications),
    )

    with _trusted_scope(platform="telegram"):
        first = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "first"},
            )
        )
    with _trusted_scope(platform="telegram"):
        second = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "second"},
            )
        )

    assert first["status"] == "verified"
    assert first["dispatch_count"] == 1
    assert second["status"] == "blocked"
    assert second["dispatch_count"] == 0
    assert second["dispatch_started"] is False
    assert len(notifications) == 2
    call_tool.assert_awaited_once()


def test_failure_after_receipt_consume_before_sdk_call_stays_blocked_and_consumed(
    tmp_path, monkeypatch
):
    home = tmp_path / "consume-before-call"
    name = "neutral_governor_consume_before_call"
    _write_plugin(home, name, _approval_policy_source())
    _manager, loaded = _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock()
    server, original_session, _registered = _server_with_tools(
        _tool(EXECUTE), call_tool=call_tool
    )
    clock = [3500.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    notifications = []
    register_gateway_notify(
        "neutral-telegram-session",
        _auto_approve_notifier(coordinator, clock, notifications),
    )
    replacement_session = SimpleNamespace(call_tool=AsyncMock())

    class SwapSessionOnEnter:
        async def __aenter__(self):
            server.session = replacement_session

        async def __aexit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    server._rpc_lock = SwapSessionOnEnter()
    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "blocked"
    assert result["dispatch_started"] is False
    assert result["dispatch_count"] == 0
    call_tool.assert_not_awaited()
    replacement_session.call_tool.assert_not_awaited()
    request = coordinator.get_request(notifications[0]["approval_id"])
    with pytest.raises(
        fresh_approval.ApprovalRejected,
        match="unknown or consumed approval grant",
    ):
        coordinator.consume(loaded.module.saved_receipt.grant, request.subject)
    assert original_session is not server.session


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("dispatch provider error"),
        TimeoutError("dispatch timeout"),
        InterruptedError("dispatch interrupted"),
        KeyboardInterrupt("dispatch keyboard interrupt"),
        asyncio.CancelledError(),
    ],
    ids=["error", "timeout", "interrupt", "keyboard-interrupt", "cancel"],
)
def test_dispatch_failure_after_sdk_call_begins_is_unknown_exactly_once(
    tmp_path, monkeypatch, failure
):
    home = tmp_path / f"dispatch-{type(failure).__name__}"
    name = f"neutral_governor_dispatch_{type(failure).__name__.lower()}"
    _write_plugin(home, name, _approval_policy_source())
    _discover_plugin(home, monkeypatch, name)
    calls = []

    async def call_tool(raw_name, arguments, *, meta=None):
        del meta
        calls.append((raw_name, dict(arguments)))
        raise failure

    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    clock = [4000.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    register_gateway_notify(
        "neutral-telegram-session",
        _auto_approve_notifier(coordinator, clock, []),
    )
    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )

    assert result["status"] == "unknown"
    assert result["dispatch_started"] is True
    assert result["dispatch_count"] == 1
    assert len(calls) == 1


def _finalizer_policy_source(finalizer_body: str) -> str:
    return f'''from tools.governed_mcp import GovernedDispatchPlan, GovernedOutcome

class Policy:
    def prepare(self, request, services):
        operation = services.bind_operation(
            {EXECUTE!r}, request.arguments,
            proposal_hash="neutral-proposal-hash",
            preview="neutral preview",
            display_text="neutral display",
        )
        receipt = services.request_fresh_approval(operation)
        return GovernedDispatchPlan(operation, receipt, {{}})

    def finalize(self, request, operation, result):
{finalizer_body}

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''


@pytest.mark.parametrize(
    ("case_name", "finalizer_body", "expected"),
    [
        (
            "exception",
            '        raise RuntimeError("finalizer failed")',
            "unknown",
        ),
        (
            "malformed",
            '        return {"status": "verified"}',
            "unknown",
        ),
        (
            "is_error_verified",
            '''        assert result.isError is True
        return GovernedOutcome("verified", "single response verified")''',
            "verified",
        ),
        (
            "is_error_blocked",
            '''        assert result.isError is True
        return GovernedOutcome("blocked", "provider rejected operation")''',
            "blocked",
        ),
    ],
)
def test_finalizer_owns_is_error_classification_and_malformed_is_unknown(
    tmp_path, monkeypatch, case_name, finalizer_body, expected
):
    home = tmp_path / case_name
    name = f"neutral_governor_finalizer_{case_name}"
    _write_plugin(home, name, _finalizer_policy_source(finalizer_body))
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="raw provider error")],
            isError=True,
        )
    )
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    clock = [5000.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    register_gateway_notify(
        "neutral-telegram-session",
        _auto_approve_notifier(coordinator, clock, []),
    )
    with _trusted_scope(platform="telegram"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )
    result = json.loads(raw)

    assert result["status"] == expected
    assert result["dispatch_count"] == 1
    call_tool.assert_awaited_once()
    assert "raw provider error" not in raw


def test_real_call_tool_result_meta_is_read_only_for_finalizer(
    tmp_path, monkeypatch
):
    home = tmp_path / "result-meta"
    name = "neutral_governor_result_meta"
    finalizer = '''        assert result.meta == {"neutral": "meta-value"}
        try:
            result.meta["neutral"] = "changed"
        except TypeError:
            pass
        else:
            raise AssertionError("provider meta was mutable")
        return GovernedOutcome("verified", "meta preserved")'''
    _write_plugin(home, name, _finalizer_policy_source(finalizer))
    _discover_plugin(home, monkeypatch, name)
    call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent={"safe": True},
            _meta={"neutral": "meta-value"},
        )
    )
    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    clock = [6000.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    register_gateway_notify(
        "neutral-telegram-session",
        _auto_approve_notifier(coordinator, clock, []),
    )
    with _trusted_scope(platform="telegram"):
        result = json.loads(
            registry.dispatch(
                mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
                {"value": "sentinel"},
            )
        )
    assert result["status"] == "verified"


def test_finalizer_cannot_reflect_host_transport_meta_into_audit_output(
    tmp_path, monkeypatch
):
    home = tmp_path / "result-meta-reflection"
    name = "neutral_governor_result_meta_reflection"
    source = f'''from tools.governed_mcp import GovernedDispatchPlan, GovernedOutcome

class Policy:
    def prepare(self, request, services):
        operation = services.bind_operation(
            {EXECUTE!r},
            request.arguments,
            proposal_hash="neutral-proposal-hash",
            preview="neutral preview {EXECUTE_META_SENTINEL}",
            display_text="neutral display {EXECUTE_META_SENTINEL}",
            base_meta={{{META_FIELD!r}: {EXECUTE_META_SENTINEL!r}}},
        )
        receipt = services.request_fresh_approval(operation)
        return GovernedDispatchPlan(operation, receipt, {{}})

    def finalize(self, request, operation, result):
        del request, operation
        reflected = result.meta[{META_FIELD!r}]
        return GovernedOutcome(
            "verified",
            f"provider reflected {{reflected}}",
            {{"reflected": reflected}},
        )

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
'''
    _write_plugin(home, name, source)
    _discover_plugin(home, monkeypatch, name)
    calls = []

    async def call_tool(raw_name, arguments, *, meta=None):
        calls.append((raw_name, dict(arguments), dict(meta or {})))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            _meta={META_FIELD: EXECUTE_META_SENTINEL},
        )

    _server_with_tools(_tool(EXECUTE), call_tool=call_tool)
    clock = [6100.0]
    coordinator = fresh_approval.FreshApprovalCoordinator(clock=lambda: clock[0])
    monkeypatch.setattr(fresh_approval, "_DEFAULT_COORDINATOR", coordinator)
    notifications = []
    register_gateway_notify(
        "neutral-telegram-session",
        _auto_approve_notifier(coordinator, clock, notifications),
    )
    with _trusted_scope(platform="telegram"):
        raw = registry.dispatch(
            mcp_tool.mcp_prefixed_tool_name(SERVER, EXECUTE),
            {"value": "sentinel"},
        )

    assert calls == [
        (
            EXECUTE,
            {"value": "sentinel"},
            {META_FIELD: EXECUTE_META_SENTINEL},
        )
    ]
    assert EXECUTE_META_SENTINEL not in raw
    assert EXECUTE_META_SENTINEL not in json.dumps(json.loads(raw))
    assert EXECUTE_META_SENTINEL not in json.dumps(notifications)


def _governor_source_for(server_name: str) -> str:
    return f'''class Policy:
    def prepare(self, request, services):
        raise RuntimeError("not invoked")
    def finalize(self, request, operation, result):
        raise RuntimeError("not invoked")
def register(ctx):
    ctx.register_mcp_governor({server_name!r}, Policy())
'''


def test_force_reload_replaces_governors_and_interrupt_rolls_back_exact_state(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_force_reload"
    replacement_server = "neutral-replacement-server"
    _write_plugin(home, name, _governor_source_for(SERVER))
    manager, loaded = _discover_plugin(home, monkeypatch, name)
    assert loaded.enabled is True
    assert manager.get_mcp_governor_attribution(SERVER) == name

    plugin_file = home / "plugins" / name / "__init__.py"
    plugin_file.write_text(
        _governor_source_for(replacement_server), encoding="utf-8"
    )
    manager.discover_and_load(force=True)
    assert manager.get_mcp_governor_attribution(SERVER) is None
    assert manager.get_mcp_governor_attribution(replacement_server) == name

    plugin_file.write_text(
        "raise KeyboardInterrupt('force reload interruption')\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyboardInterrupt, match="force reload interruption"):
        manager.discover_and_load(force=True)
    assert manager.get_mcp_governor_attribution(SERVER) is None
    assert manager.get_mcp_governor_attribution(replacement_server) == name


def test_readers_see_no_staged_governor_until_atomic_commit(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    name = "neutral_governor_atomic_reader"
    source = f'''import threading

started = threading.Event()
release = threading.Event()

class Policy:
    def prepare(self, request, services):
        raise RuntimeError("not invoked")
    def finalize(self, request, operation, result):
        raise RuntimeError("not invoked")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
    started.set()
    if not release.wait(timeout=5):
        raise RuntimeError("release timeout")
'''
    _write_plugin(home, name, source)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    plugin_module._plugin_manager = manager
    failures = []

    def discover():
        try:
            manager.discover_and_load()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=discover, daemon=True)
    thread.start()
    module_name = f"hermes_plugins.{name}"
    deadline = time.monotonic() + 5
    module = None
    while time.monotonic() < deadline:
        module = sys.modules.get(module_name)
        if (
            module is not None
            and hasattr(module, "started")
            and module.started.wait(timeout=0.01)
        ):
            break
    assert module is not None
    assert manager.get_mcp_governor_attribution(SERVER) is None
    module.release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
    assert manager.get_mcp_governor_attribution(SERVER) == name


def test_concurrent_exact_server_claim_conflict_fails_closed_at_commit(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    staged_name = "neutral_governor_concurrent_staged"
    live_name = "neutral_governor_concurrent_live"
    source = f'''import threading

started = threading.Event()
release = threading.Event()

class Policy:
    def prepare(self, request, services):
        raise RuntimeError("not invoked")
    def finalize(self, request, operation, result):
        raise RuntimeError("not invoked")

def register(ctx):
    ctx.register_mcp_governor({SERVER!r}, Policy())
    started.set()
    if not release.wait(timeout=5):
        raise RuntimeError("release timeout")
'''
    _write_plugin(home, staged_name, source)
    config_path = home / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["plugins"]["entries"][live_name] = {
        "allow_mcp_governance": True
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    plugin_module._plugin_manager = manager
    thread = threading.Thread(target=manager.discover_and_load, daemon=True)
    thread.start()
    module_name = f"hermes_plugins.{staged_name}"
    deadline = time.monotonic() + 5
    module = None
    while time.monotonic() < deadline:
        module = sys.modules.get(module_name)
        if (
            module is not None
            and hasattr(module, "started")
            and module.started.wait(timeout=0.01)
        ):
            break
    assert module is not None

    live_context = PluginContext(
        PluginManifest(name=live_name, source="user"), manager
    )
    live_context.register_mcp_governor(
        SERVER,
        SimpleNamespace(
            prepare=lambda request, services: None,
            finalize=lambda request, operation, result: None,
        ),
    )
    module.release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert manager.get_mcp_governor_attribution(SERVER) == live_name
    assert manager._plugins[staged_name].enabled is False
    assert "conflict" not in (manager._plugins[staged_name].error or "").lower()
