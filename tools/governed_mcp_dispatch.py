"""Host-private execution engine for exact-server governed MCP tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextvars
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
import threading
from types import MappingProxyType
from typing import Any

from agent.trusted_interaction import (
    TrustedInteraction,
    get_current_trusted_interaction,
)
from tools.fresh_approval import (
    ApprovalGrant,
    ApprovalRejected,
    ConversationBinding,
    InvocationSubject,
    get_fresh_approval_coordinator,
)
from tools.governance_violation_reasons import sanitized_violation_reason
from tools.governed_mcp import (
    ApprovalReceipt,
    BoundOperation,
    GovernedDispatchPlan,
    GovernedOutcome,
    MCPGovernanceRequest,
    MCPToolDescriptor,
    ReadOnlyPassThrough,
    _BoundOperationState,
    _canonical_json_bytes,
    _deep_freeze,
    _deep_thaw,
    _host_get_bound_operation_state,
    _host_issue_approval_receipt,
    _host_issue_bound_operation,
    _host_validate_approval_receipt,
)


_MAX_HASH_CHARS = 256
_MAX_TEXT_CHARS = 4096
_MAX_PREVIEW_CHARS = 64_000
_MAX_FIELD_NAME_CHARS = 256
_MAX_PATCH_FIELDS = 128
_MAX_JSON_BYTES = 1024 * 1024
_MAX_TRANSPORT_META_BYTES = 64 * 1024
_MAX_TRANSPORT_META_FIELDS = 128
_MAX_EVIDENCE_BYTES = 16 * 1024
_FRESH_APPROVAL_TTL_SECONDS = 300.0
_BREAKER_ACTION_IGNORE = "ignore"
_BREAKER_ACTION_BUMP = "bump"
_BREAKER_ACTION_RESET = "reset"
_SAFE_NOTIFIER_EXCEPTION_TYPES = frozenset(
    {
        "ConnectionError",
        "ImportError",
        "OSError",
        "RuntimeError",
        "TimeoutError",
    }
)
_ANNOTATION_HINTS = (
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)
_EMPTY_TRANSPORT_META: Mapping[str, Any] = MappingProxyType({})
_READ_ONLY_PASS_THROUGH = object()
logger = logging.getLogger(__name__)


def _safe_notifier_exception_type(exc: Exception) -> str:
    name = type(exc).__name__
    if name in _SAFE_NOTIFIER_EXCEPTION_TYPES:
        return name
    return "unrecognized_exception"


def _approved_at_rfc3339(confirmed_at: int | float) -> str:
    """Render a fresh-approval timestamp as canonical RFC3339 UTC."""

    approved = datetime.fromtimestamp(confirmed_at, tz=timezone.utc)
    return approved.isoformat().replace("+00:00", "Z")


class _GovernanceViolation(RuntimeError):
    pass


class _PreflightIndeterminate(RuntimeError):
    pass


class _GovernedDispatchResult(str):
    def __new__(cls, payload: str, *, breaker_action: str):
        obj = str.__new__(cls, payload)
        obj.breaker_action = breaker_action
        return obj


class _ReadOnlyObject:
    """Attribute and mapping view over a detached, recursively frozen model."""

    __slots__ = ("__data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_ReadOnlyObject__data", data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_ReadOnlyObject__data")
        if name in data:
            return data[name]
        if name == "meta" and "_meta" in data:
            return data["_meta"]
        raise AttributeError(name)

    def __getitem__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_ReadOnlyObject__data")
        return data[key]

    def __iter__(self):
        data = object.__getattribute__(self, "_ReadOnlyObject__data")
        return iter(data)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("provider result is read-only")

    def model_dump(self, *, by_alias: bool = True) -> dict[str, Any]:
        del by_alias
        data = object.__getattribute__(self, "_ReadOnlyObject__data")
        return _deep_thaw(data)


def _model_dump(value: Any, *, exclude_unset: bool) -> dict[str, Any] | None:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            result = dump(by_alias=True, exclude_unset=exclude_unset)
        except TypeError:
            result = dump(by_alias=True)
        return result if isinstance(result, dict) else None
    dump = getattr(value, "dict", None)
    if callable(dump):
        try:
            result = dump(by_alias=True, exclude_unset=exclude_unset)
        except TypeError:
            result = dump()
        return result if isinstance(result, dict) else None
    return None


def _plain_model_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_model_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_model_value(item) for item in value]
    dumped = _model_dump(value, exclude_unset=False)
    if dumped is not None:
        return _plain_model_value(dumped)
    values = getattr(value, "__dict__", None)
    if isinstance(values, dict):
        return {
            str(key): _plain_model_value(item)
            for key, item in values.items()
            if not str(key).startswith("__")
        }
    raise TypeError(f"unsupported provider result value: {type(value).__name__}")


def _readonly_provider_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _readonly_provider_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_provider_value(item) for item in value)
    plain = _plain_model_value(value)
    if isinstance(plain, dict):
        frozen = MappingProxyType(
            {str(key): _readonly_provider_value(item) for key, item in plain.items()}
        )
        return _ReadOnlyObject(frozen)
    return _readonly_provider_value(plain)


def _tool_meta(tool: Any, dumped: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "_meta" in dumped:
        value = dumped["_meta"]
        return value if isinstance(value, Mapping) else None
    fields_set = getattr(tool, "model_fields_set", None)
    if fields_set is not None and "meta" not in fields_set:
        return None
    sentinel = object()
    value = getattr(tool, "meta", sentinel)
    if value is sentinel:
        value = getattr(tool, "_meta", sentinel)
    if value is sentinel or value is None:
        return None
    return value if isinstance(value, Mapping) else None


def build_mcp_tool_descriptor(
    server_name: str,
    normalized_name: str,
    tool: Any,
) -> MCPToolDescriptor:
    """Build the canonical exact metadata snapshot from a real SDK Tool."""

    raw_tool_name = getattr(tool, "name", None)
    if not isinstance(server_name, str) or not server_name:
        raise ValueError("descriptor server name must be non-empty")
    if not isinstance(raw_tool_name, str) or not raw_tool_name:
        raise ValueError("descriptor raw tool name must be non-empty")
    if not isinstance(normalized_name, str) or not normalized_name:
        raise ValueError("descriptor normalized name must be non-empty")
    input_schema = getattr(tool, "inputSchema", None)
    if not isinstance(input_schema, Mapping):
        raise ValueError("descriptor input schema must be an object")
    output_schema = getattr(tool, "outputSchema", None)
    if output_schema is not None and not isinstance(output_schema, Mapping):
        raise ValueError("descriptor output schema must be an object or None")

    dumped = _model_dump(tool, exclude_unset=True) or {}
    raw_annotations = getattr(tool, "annotations", None)
    annotations: Mapping[str, Any] | None
    if raw_annotations is None:
        annotations = None
    else:
        annotation_dump = _model_dump(raw_annotations, exclude_unset=True)
        if annotation_dump is None:
            annotation_dump = _plain_model_value(raw_annotations)
        if not isinstance(annotation_dump, dict):
            raise ValueError("descriptor annotations must be an object")
        for hint in _ANNOTATION_HINTS:
            annotation_dump.setdefault(hint, None)
        annotations = annotation_dump

    meta = _tool_meta(tool, dumped)
    description = getattr(tool, "description", None)
    if description is not None and not isinstance(description, str):
        raise ValueError("descriptor description must be a string or None")
    payload = {
        "annotations": annotations,
        "description": description,
        "input_schema": input_schema,
        "meta": meta,
        "normalized_name": normalized_name,
        "output_schema": output_schema,
        "raw_tool_name": raw_tool_name,
        "server_name": server_name,
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return MCPToolDescriptor(
        server_name=server_name,
        raw_tool_name=raw_tool_name,
        normalized_name=normalized_name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        annotations=annotations,
        meta=meta,
        fingerprint=fingerprint,
    )


def _redact_text(value: str) -> str:
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    except Exception:
        return "redacted text unavailable"


def _validate_text(
    label: str,
    value: object,
    *,
    allow_empty: bool = False,
    maximum: int = _MAX_TEXT_CHARS,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _GovernanceViolation(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise _GovernanceViolation(f"{label} is too large")
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise _GovernanceViolation(f"{label} contains control characters")
    return value


def _validate_hash(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _GovernanceViolation(f"{label} must be non-empty")
    if len(value) > _MAX_HASH_CHARS:
        raise _GovernanceViolation(f"{label} is too large")
    if any(ord(char) < 33 or ord(char) == 127 for char in value):
        raise _GovernanceViolation(f"{label} contains invalid characters")
    return value


def _validate_json_mapping(
    label: str,
    value: object,
    *,
    forbid_transport_meta: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _GovernanceViolation(f"{label} must be an object")
    if forbid_transport_meta and "_meta" in value:
        raise _GovernanceViolation(
            f"{label} cannot contain top-level _meta"
        )
    try:
        encoded = _canonical_json_bytes(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _GovernanceViolation(f"{label} is not canonical JSON") from exc
    if len(encoded) > _MAX_JSON_BYTES:
        raise _GovernanceViolation(f"{label} is too large")
    return value


def _validate_transport_meta(
    label: str,
    value: object | None,
    *,
    allow_none: bool = False,
) -> Mapping[str, Any]:
    if value is None:
        if not allow_none:
            raise _GovernanceViolation(f"{label} must be an object")
        value = {}
    if not isinstance(value, Mapping):
        raise _GovernanceViolation(f"{label} must be an object")
    if len(value) > _MAX_TRANSPORT_META_FIELDS:
        raise _GovernanceViolation(f"{label} has too many fields")

    def _validate_keys(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise _GovernanceViolation(
                        f"{label} field names must be strings"
                    )
                _validate_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                _validate_keys(nested)
        elif item is None or isinstance(item, (bool, int, float, str)):
            return
        else:
            raise _GovernanceViolation(
                f"{label} contains a non-JSON value"
            )

    for field_name in value:
        if (
            not isinstance(field_name, str)
            or not field_name
            or len(field_name) > _MAX_FIELD_NAME_CHARS
            or any(
                ord(char) < 32 or ord(char) == 127
                for char in field_name
            )
        ):
            raise _GovernanceViolation(f"{label} has an invalid field name")
    _validate_keys(value)
    try:
        frozen = _deep_freeze(value)
        encoded = _canonical_json_bytes(frozen)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _GovernanceViolation(f"{label} is not canonical JSON") from exc
    if len(encoded) > _MAX_TRANSPORT_META_BYTES:
        raise _GovernanceViolation(f"{label} is too large")
    return frozen


def _validate_patch_fields(
    label: str,
    value: object,
    *,
    forbid_transport_meta: bool = False,
) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _GovernanceViolation(f"{label} must be a sequence")
    if len(value) > _MAX_PATCH_FIELDS:
        raise _GovernanceViolation(f"{label} has too many fields")
    fields: list[str] = []
    for field_name in value:
        if (
            not isinstance(field_name, str)
            or not field_name
            or len(field_name) > _MAX_FIELD_NAME_CHARS
            or any(
                ord(char) < 32 or ord(char) == 127
                for char in field_name
            )
        ):
            raise _GovernanceViolation(f"{label} has an invalid field")
        if forbid_transport_meta and field_name == "_meta":
            raise _GovernanceViolation(f"{label} cannot contain _meta")
        if field_name in fields:
            raise _GovernanceViolation(f"{label} contains a duplicate field")
        fields.append(field_name)
    return frozenset(fields)


def _operation_binding_hash(
    *,
    server_name: str,
    invocation_raw_tool_name: str,
    invocation_descriptor_hash: str,
    target_raw_tool_name: str,
    target_normalized_name: str,
    target_descriptor_hash: str,
    base_arguments: Mapping[str, Any],
    base_meta: Mapping[str, Any],
    proposal_hash: str,
    preview: str,
    display_text: str,
    interaction_hash: str,
    preflight_facts: Mapping[str, Any],
    allowed_post_approval_patch_fields: Sequence[str],
    allowed_post_approval_meta_fields: Sequence[str],
) -> str:
    payload = {
        "allowed_post_approval_meta_fields": sorted(
            allowed_post_approval_meta_fields
        ),
        "allowed_post_approval_patch_fields": sorted(
            allowed_post_approval_patch_fields
        ),
        "base_arguments": base_arguments,
        "base_meta": base_meta,
        "display_text": display_text,
        "interaction_hash": interaction_hash,
        "invocation_descriptor_hash": invocation_descriptor_hash,
        "invocation_raw_tool_name": invocation_raw_tool_name,
        "preflight_facts": preflight_facts,
        "preview": preview,
        "proposal_hash": proposal_hash,
        "server_name": server_name,
        "target_descriptor_hash": target_descriptor_hash,
        "target_normalized_name": target_normalized_name,
        "target_raw_tool_name": target_raw_tool_name,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _transport_meta_atoms(value: object) -> tuple[object, ...]:
    atoms: list[object] = []

    def _visit(item: object) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                _visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                _visit(nested)
        elif item is not None:
            atoms.append(item)

    _visit(value)
    return tuple(atoms)


def _host_meta_string_atoms(atoms: Sequence[object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                atom
                for atom in atoms
                if isinstance(atom, str) and atom
            },
            key=len,
            reverse=True,
        )
    )


def _redact_host_meta_text(
    value: str,
    atoms: Sequence[object],
) -> str:
    item = value
    for atom in _host_meta_string_atoms(atoms):
        item = item.replace(atom, "[REDACTED]")
    return item


def _sanitize_host_meta_evidence(
    value: Any,
    atoms: Sequence[object],
) -> Any:
    string_atoms = _host_meta_string_atoms(atoms)

    def _sanitize(item: Any) -> Any:
        if isinstance(item, str):
            for atom in string_atoms:
                item = item.replace(atom, "[REDACTED]")
            return item
        if isinstance(item, Mapping):
            return {
                key: _sanitize(nested)
                for key, nested in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [_sanitize(nested) for nested in item]
        for atom in atoms:
            if type(item) is type(atom) and item == atom:
                return "[REDACTED]"
        return item

    return _sanitize(value)


async def _call_with_pending_context(server: Any, invoke: Any) -> Any:
    previous_context = getattr(server, "_pending_call_context", None)
    server._pending_call_context = contextvars.copy_context()
    try:
        result = await invoke()
        mark_proven = getattr(server, "_mark_session_proven", None)
        if mark_proven is not None:
            mark_proven()
        return result
    finally:
        server._pending_call_context = previous_context


def _current_server(server_name: str) -> Any | None:
    from tools import mcp_tool

    with mcp_tool._lock:
        return mcp_tool._servers.get(server_name)


def _exact_raw_tool(server: Any, raw_tool_name: str) -> Any | None:
    matches = [
        tool
        for tool in list(getattr(server, "_tools", ()) or ())
        if getattr(tool, "name", None) == raw_tool_name
    ]
    return matches[0] if len(matches) == 1 else None


def _resolve_live_tool_descriptor(
    server: Any,
    session: Any,
    server_name: str,
    raw_tool_name: str,
) -> MCPToolDescriptor | None:
    """Resolve one real, unambiguous, currently registered raw MCP tool."""

    from tools import mcp_tool
    from tools.registry import registry

    if _current_server(server_name) is not server:
        return None
    if getattr(server, "session", None) is not session or session is None:
        return None
    raw_tool = _exact_raw_tool(server, raw_tool_name)
    if raw_tool is None:
        return None
    try:
        normalized_name = mcp_tool._convert_mcp_schema(
            server_name, raw_tool
        )["name"]
        normalized_raw_matches = [
            candidate
            for candidate in list(getattr(server, "_tools", ()) or ())
            if mcp_tool._convert_mcp_schema(server_name, candidate)["name"]
            == normalized_name
        ]
        if len(normalized_raw_matches) != 1:
            return None
        utility_entries = mcp_tool._select_utility_schemas(
            server_name,
            server,
            getattr(server, "_config", {}) or {},
        )
        if any(
            entry.get("schema", {}).get("name") == normalized_name
            for entry in utility_entries
        ):
            return None
        if registry.get_toolset_for_tool(normalized_name) != f"mcp-{server_name}":
            return None
        with mcp_tool._lock:
            if mcp_tool._mcp_tool_server_names.get(normalized_name) != server_name:
                return None
        return build_mcp_tool_descriptor(
            server_name,
            normalized_name,
            raw_tool,
        )
    except Exception:
        return None


def _target_is_current(
    server: Any,
    session: Any,
    descriptor: MCPToolDescriptor,
) -> bool:
    current = _resolve_live_tool_descriptor(
        server,
        session,
        descriptor.server_name,
        descriptor.raw_tool_name,
    )
    return bool(
        current is not None
        and current.normalized_name == descriptor.normalized_name
        and current.fingerprint == descriptor.fingerprint
    )


class _GovernedServices:
    def __init__(
        self,
        *,
        registration: Any,
        request: MCPGovernanceRequest,
        server: Any,
        session: Any,
        tool_timeout: float,
    ) -> None:
        self._registration = registration
        self._request = request
        self._server = server
        self._session = session
        self._tool_timeout = tool_timeout
        self._owner_thread = threading.get_ident()
        self._active = True
        self._lock = threading.RLock()
        self._violation: str | None = None
        self._preflight_used = False
        self.preflight_started = False
        self.preflight_completed = False
        self.preflight_count = 0
        self.preflight_indeterminate = False
        self._preflight_facts: Mapping[str, Any] = _EMPTY_TRANSPORT_META
        self.bound_operation: BoundOperation | None = None
        self.dispatch_tool_name: str | None = None
        self.approval_receipt: ApprovalReceipt | None = None
        self.approval_subject: InvocationSubject | None = None
        self._approval_coordinator: Any | None = None
        self._approval_receipt_issuer = object()
        self.dispatch_started = False
        self.dispatch_count = 0
        self.transport_failed = False
        self.transport_succeeded = False
        self._host_meta_atoms: list[object] = []

    @property
    def violation(self) -> str | None:
        with self._lock:
            return self._violation

    @property
    def host_meta_atoms(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._host_meta_atoms)

    def _remember_transport_meta(self, meta: Mapping[str, Any]) -> None:
        with self._lock:
            for atom in _transport_meta_atoms(meta):
                if not any(
                    type(atom) is type(existing) and atom == existing
                    for existing in self._host_meta_atoms
                ):
                    self._host_meta_atoms.append(atom)

    def _fail(self, reason: str) -> None:
        with self._lock:
            if self._violation is None:
                self._violation = reason
        raise _GovernanceViolation(reason)

    def _guard(self) -> None:
        if threading.get_ident() != self._owner_thread:
            self._fail("governance capability cannot be used from a detached context")
        with self._lock:
            if not self._active:
                self._fail("governance capability is closed")

    def close(self) -> None:
        with self._lock:
            self._active = False

    def preflight_call(
        self,
        raw_tool_name: str,
        arguments: Mapping[str, Any],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> Any:
        self._guard()
        with self._lock:
            if self._preflight_used:
                self._fail("only one preflight call is permitted")
            self._preflight_used = True
        if not isinstance(raw_tool_name, str) or not raw_tool_name:
            self._fail("preflight raw tool name must be non-empty")
        args = _validate_json_mapping(
            "preflight arguments",
            arguments,
            forbid_transport_meta=True,
        )
        frozen_meta = _validate_transport_meta(
            "preflight meta", meta, allow_none=True
        )
        self._remember_transport_meta(frozen_meta)
        call_arguments = _deep_thaw(args)
        call_meta = _deep_thaw(frozen_meta)
        if not _target_is_current(
            self._server, self._session, self._request.descriptor
        ):
            self._fail("MCP target changed before preflight")
        preflight_descriptor = _resolve_live_tool_descriptor(
            self._server,
            self._session,
            self._request.descriptor.server_name,
            raw_tool_name,
        )
        if preflight_descriptor is None:
            self._fail("preflight must target a real raw tool on the same server")
        self._preflight_facts = _deep_freeze(
            {
                "arguments": args,
                "descriptor_hash": preflight_descriptor.fingerprint,
                "meta": frozen_meta,
                "normalized_name": preflight_descriptor.normalized_name,
                "raw_tool_name": preflight_descriptor.raw_tool_name,
            }
        )

        async def _call() -> Any:
            async with self._server._rpc_lock:
                if not _target_is_current(
                    self._server,
                    self._session,
                    self._request.descriptor,
                ):
                    raise _GovernanceViolation(
                        "MCP target changed before preflight"
                    )
                current_preflight = _resolve_live_tool_descriptor(
                    self._server,
                    self._session,
                    self._request.descriptor.server_name,
                    raw_tool_name,
                )
                if (
                    current_preflight is None
                    or current_preflight.fingerprint
                    != preflight_descriptor.fingerprint
                    or current_preflight.normalized_name
                    != preflight_descriptor.normalized_name
                ):
                    raise _GovernanceViolation(
                        "preflight target changed before SDK call"
                    )
                mark_tool_call = getattr(self._server, "mark_tool_call", None)
                if not callable(mark_tool_call):
                    logger.error(
                        "Governed MCP preflight blocked for server %s: required "
                        "mark_tool_call activity marker is unavailable",
                        self._request.descriptor.server_name,
                    )
                    raise _GovernanceViolation(
                        "MCP server activity marker is unavailable"
                    )
                try:
                    mark_tool_call()
                except Exception:
                    logger.error(
                        "Governed MCP preflight blocked for server %s: "
                        "mark_tool_call activity marker failed",
                        self._request.descriptor.server_name,
                    )
                    raise _GovernanceViolation(
                        "MCP server activity marker failed"
                    ) from None
                with self._lock:
                    self.preflight_started = True
                    self.preflight_count = 1
                result = await _call_with_pending_context(
                    self._server,
                    lambda: self._session.call_tool(
                        raw_tool_name,
                        arguments=call_arguments,
                        meta=call_meta or None,
                    ),
                )
                with self._lock:
                    self.preflight_completed = True
                    self.transport_succeeded = True
                return result

        from tools.mcp_tool import _run_on_mcp_loop

        try:
            result = _run_on_mcp_loop(_call, timeout=self._tool_timeout)
        except BaseException as exc:
            with self._lock:
                started = self.preflight_started
                if not isinstance(exc, _GovernanceViolation):
                    self.transport_failed = True
                if started:
                    self.preflight_indeterminate = True
                    if self._violation is None:
                        self._violation = "preflight outcome is unknown"
            if started:
                raise _PreflightIndeterminate(
                    "preflight outcome is unknown"
                ) from None
            if isinstance(exc, _GovernanceViolation):
                raise
            self._fail("preflight could not start")
        return _readonly_provider_value(result)

    def bind_operation(
        self,
        raw_tool_name: str,
        base_arguments: Mapping[str, Any],
        *,
        proposal_hash: str,
        preview: str,
        display_text: str,
        base_meta: Mapping[str, Any] = _EMPTY_TRANSPORT_META,
        allowed_post_approval_patch_fields: Sequence[str] = (),
        allowed_post_approval_meta_fields: Sequence[str] = (),
    ) -> BoundOperation:
        self._guard()
        if self.bound_operation is not None:
            self._fail("only one operation may be bound")
        raw_tool_name = _validate_text(
            "bound operation target raw tool name",
            raw_tool_name,
        )
        self.dispatch_tool_name = raw_tool_name
        target_descriptor = _resolve_live_tool_descriptor(
            self._server,
            self._session,
            self._request.descriptor.server_name,
            raw_tool_name,
        )
        if target_descriptor is None:
            self._fail(
                "bound operation target is not one exact registered raw tool "
                "on the current server"
            )
        base = _validate_json_mapping(
            "base dispatch arguments",
            base_arguments,
            forbid_transport_meta=True,
        )
        frozen_base_meta = _validate_transport_meta(
            "base dispatch meta", base_meta
        )
        self._remember_transport_meta(frozen_base_meta)
        proposal = _validate_hash("proposal hash", proposal_hash)
        safe_preview = _redact_host_meta_text(
            _redact_text(
                _validate_text(
                    "preview",
                    preview,
                    maximum=_MAX_PREVIEW_CHARS,
                )
            ),
            self.host_meta_atoms,
        )
        safe_display = _redact_host_meta_text(
            _redact_text(
                _validate_text(
                    "display text",
                    display_text,
                    maximum=_MAX_PREVIEW_CHARS,
                )
            ),
            self.host_meta_atoms,
        )
        argument_patch_fields = _validate_patch_fields(
            "allowed argument patch fields",
            allowed_post_approval_patch_fields,
            forbid_transport_meta=True,
        )
        meta_patch_fields = _validate_patch_fields(
            "allowed meta patch fields",
            allowed_post_approval_meta_fields,
        )

        operation_hash = _operation_binding_hash(
            server_name=self._request.descriptor.server_name,
            invocation_raw_tool_name=self._request.descriptor.raw_tool_name,
            invocation_descriptor_hash=self._request.descriptor.fingerprint,
            target_raw_tool_name=target_descriptor.raw_tool_name,
            target_normalized_name=target_descriptor.normalized_name,
            target_descriptor_hash=target_descriptor.fingerprint,
            base_arguments=base,
            base_meta=frozen_base_meta,
            proposal_hash=proposal,
            preview=safe_preview,
            display_text=safe_display,
            interaction_hash=self._request.interaction.evidence_hash,
            preflight_facts=self._preflight_facts,
            allowed_post_approval_patch_fields=argument_patch_fields,
            allowed_post_approval_meta_fields=meta_patch_fields,
        )
        state = _BoundOperationState(
            server_name=self._request.descriptor.server_name,
            invocation_raw_tool_name=self._request.descriptor.raw_tool_name,
            invocation_descriptor_hash=self._request.descriptor.fingerprint,
            target_descriptor=target_descriptor,
            preflight_facts=self._preflight_facts,
            base_arguments=_deep_freeze(base),
            base_meta=frozen_base_meta,
            proposal_hash=proposal,
            preview=safe_preview,
            display_text=safe_display,
            interaction_hash=self._request.interaction.evidence_hash,
            operation_hash=operation_hash,
            allowed_argument_patch_fields=argument_patch_fields,
            allowed_meta_patch_fields=meta_patch_fields,
        )
        self.bound_operation = _host_issue_bound_operation(state)
        return self.bound_operation

    def _conversation_binding(self) -> ConversationBinding:
        interaction = self._request.interaction
        return ConversationBinding(
            profile_id=interaction.profile_id,
            platform=interaction.platform,
            channel_id=interaction.channel_id,
            chat_id=interaction.chat_id,
            session_key=interaction.session_key,
            session_id=interaction.session_id,
            thread_id=interaction.thread_id,
            actor_id=interaction.actor_id,
        )

    def request_fresh_approval(
        self, operation: BoundOperation
    ) -> ApprovalReceipt:
        self._guard()
        if operation is not self.bound_operation or operation is None:
            self._fail("approval must reference this service's bound operation")
        if self.approval_subject is not None:
            self._fail("only one fresh approval may be requested")
        state = _host_get_bound_operation_state(operation)
        try:
            from tools.approval import get_current_observability_context

            turn_id, tool_call_id = get_current_observability_context()
        except Exception:
            turn_id, tool_call_id = "", ""
        subject = InvocationSubject(
            invocation_id=state.operation_hash,
            tool_call_id=tool_call_id or f"tool_{state.operation_hash}",
            turn_id=turn_id or self._request.interaction.interaction_id,
            origin_message_id=self._request.interaction.message_id,
            args_hash=state.operation_hash,
            proposal_hash=state.proposal_hash,
            conversation=self._conversation_binding(),
        )
        coordinator = get_fresh_approval_coordinator()
        request = coordinator.request(
            subject,
            ttl_seconds=_FRESH_APPROVAL_TTL_SECONDS,
        )
        self.approval_subject = subject
        payload = {
            "allow_permanent": False,
            "allow_session": False,
            "approval_id": request.approval_id,
            "choices": ["once", "deny"],
            "command": state.preview,
            "description": state.display_text,
            "fresh_only": True,
        }
        notifier_failure: tuple[str, str] | None = None
        try:
            from tools.approval import notify_fresh_approval
        except Exception as exc:
            notifier_failure = (
                "import_exception",
                _safe_notifier_exception_type(exc),
            )
            notified = False
        else:
            try:
                notified = notify_fresh_approval(
                    self._request.interaction.session_key,
                    payload,
                )
            except Exception as exc:
                notifier_failure = (
                    "callback_exception",
                    _safe_notifier_exception_type(exc),
                )
                notified = False
        if not notified:
            if notifier_failure is None:
                logger.warning(
                    "fresh approval notifier unavailable: outcome=not_registered"
                )
            else:
                outcome, exception_type = notifier_failure
                logger.warning(
                    "fresh approval notifier unavailable: outcome=%s "
                    "exception_type=%s",
                    outcome,
                    exception_type,
                )
            try:
                coordinator.cancel(request.approval_id)
            except ApprovalRejected:
                pass
            self._fail("fresh approval notifier is unavailable")

        resolution = coordinator.await_resolution(
            request.approval_id,
            timeout=_FRESH_APPROVAL_TTL_SECONDS,
        )
        if resolution is None:
            try:
                coordinator.cancel(request.approval_id)
            except ApprovalRejected:
                pass
            self._fail("fresh approval timed out")
        if (
            resolution.outcome != "approved"
            or resolution.subject != subject
            or not isinstance(resolution.grant, ApprovalGrant)
            or resolution.evidence is None
            or resolution.evidence.decision != "approve_once"
            or resolution.evidence.approval_id != request.approval_id
            or not coordinator.verify_evidence(resolution.evidence)
        ):
            self._fail("fresh approval was not granted")
        evidence = resolution.evidence
        try:
            receipt = _host_issue_approval_receipt(
                issuer=self._approval_receipt_issuer,
                bound_operation=operation,
                approval_request_id=evidence.approval_id,
                approval_message_id=evidence.responder.inbound_id,
                approval_event_id=evidence.evidence_hash,
                approved_at=_approved_at_rfc3339(evidence.confirmed_at),
                evidence_hash=evidence.evidence_hash,
                grant=resolution.grant,
            )
        except (TypeError, ValueError, OSError, OverflowError):
            self._fail("fresh approval evidence could not issue a receipt")
        self._approval_coordinator = coordinator
        self.approval_receipt = receipt
        return receipt

    def hmac_sha256(
        self, secret_env_name: str, canonical_payload: bytes | str
    ) -> str:
        self._guard()
        if (
            not isinstance(secret_env_name, str)
            or not secret_env_name
            or secret_env_name not in self._registration.secret_env_names
        ):
            self._fail("HMAC secret is not declared for this governor")
        if isinstance(canonical_payload, str):
            payload = canonical_payload.encode("utf-8")
        elif isinstance(canonical_payload, bytes):
            payload = bytes(canonical_payload)
        else:
            self._fail("HMAC payload must be bytes or string")
        if len(payload) > _MAX_JSON_BYTES:
            self._fail("HMAC payload is too large")
        secret = os.environ.get(secret_env_name)
        if not isinstance(secret, str) or not secret:
            self._fail("HMAC secret is unavailable")
        return hmac.new(
            secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()


def _audit_json(
    *,
    descriptor: MCPToolDescriptor,
    status: str,
    services: _GovernedServices | None,
    summary: str,
    operation_hash: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    breaker_action: str = _BREAKER_ACTION_IGNORE,
) -> str:
    preflight_started = bool(services and services.preflight_started)
    preflight_completed = bool(services and services.preflight_completed)
    dispatch_started = bool(services and services.dispatch_started)
    dispatch_tool_name = services.dispatch_tool_name if services else None
    host_meta_atoms = services.host_meta_atoms if services else ()
    payload: dict[str, Any] = {
        "dispatch_count": services.dispatch_count if services else 0,
        "dispatch_started": dispatch_started,
        "dispatch_tool": dispatch_tool_name,
        "invoked_tool": descriptor.raw_tool_name,
        "operation_hash": operation_hash,
        "outcome": {
            "summary": _redact_host_meta_text(
                _redact_text(summary)[:_MAX_TEXT_CHARS],
                host_meta_atoms,
            )
        },
        "preflight_completed": preflight_completed,
        "preflight_count": services.preflight_count if services else 0,
        "preflight_started": preflight_started,
        "server": descriptor.server_name,
        "status": status,
    }
    if evidence is not None:
        payload["outcome"]["evidence"] = _sanitize_host_meta_evidence(
            _deep_thaw(evidence),
            host_meta_atoms,
        )
    return _GovernedDispatchResult(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        breaker_action=breaker_action,
    )


def _services_breaker_action(services: _GovernedServices | None) -> str:
    if services is None:
        return _BREAKER_ACTION_IGNORE
    if services.transport_failed:
        return _BREAKER_ACTION_BUMP
    if services.transport_succeeded:
        return _BREAKER_ACTION_RESET
    return _BREAKER_ACTION_IGNORE


def _safe_outcome(outcome: object) -> tuple[str, str, Mapping[str, Any] | None]:
    if type(outcome) is not GovernedOutcome:
        raise _GovernanceViolation("finalizer returned a malformed outcome")
    if outcome.status not in {"verified", "blocked", "unknown"}:
        raise _GovernanceViolation("finalizer returned an invalid status")
    summary = _validate_text("outcome summary", outcome.summary)
    summary = _redact_text(summary)
    evidence = outcome.evidence
    if evidence is not None:
        evidence = _validate_json_mapping("outcome evidence", evidence)
        safe_evidence = json.loads(
            _redact_text(
                json.dumps(
                    _deep_thaw(evidence),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        )
        encoded = _canonical_json_bytes(safe_evidence)
        if len(encoded) > _MAX_EVIDENCE_BYTES:
            raise _GovernanceViolation("outcome evidence is too large")
        evidence = _deep_freeze(safe_evidence)
    return outcome.status, summary, evidence


def dispatch_governed_mcp(
    *,
    registration: Any,
    descriptor: MCPToolDescriptor,
    server: Any,
    session: Any,
    arguments: Mapping[str, Any],
    tool_timeout: float,
) -> Any:
    """Execute one governed invocation with no provider or reconnect retry."""

    interaction = get_current_trusted_interaction()
    if not isinstance(interaction, TrustedInteraction):
        return _audit_json(
            descriptor=descriptor,
            status="blocked",
            services=None,
            summary="trusted interaction provenance is required",
        )
    if not _target_is_current(server, session, descriptor):
        return _audit_json(
            descriptor=descriptor,
            status="blocked",
            services=None,
            summary="MCP target metadata or session changed",
        )
    try:
        args = _validate_json_mapping(
            "provider arguments",
            arguments,
            forbid_transport_meta=True,
        )
        request = MCPGovernanceRequest(
            descriptor=descriptor,
            arguments=args,
            interaction=interaction,
        )
    except Exception:
        return _audit_json(
            descriptor=descriptor,
            status="blocked",
            services=None,
            summary="provider arguments are invalid",
        )

    services = _GovernedServices(
        registration=registration,
        request=request,
        server=server,
        session=session,
        tool_timeout=tool_timeout,
    )
    plan: object = None
    prepare_error: BaseException | None = None
    try:
        plan = registration.policy.prepare(request, services)
    except BaseException as exc:
        prepare_error = exc
    finally:
        services.close()

    operation_hash = None
    if services.bound_operation is not None:
        try:
            operation_hash = _host_get_bound_operation_state(
                services.bound_operation
            ).operation_hash
        except Exception:
            operation_hash = None
    if services.preflight_indeterminate:
        return _audit_json(
            descriptor=descriptor,
            status="unknown",
            services=services,
            summary="preflight outcome is unknown; target dispatch did not start",
            operation_hash=operation_hash,
            breaker_action=_services_breaker_action(services),
        )
    if prepare_error is not None or services.violation is not None:
        if services.violation is not None:
            logger.warning(
                "governed MCP violation for %s: reason=%s",
                descriptor.raw_tool_name,
                sanitized_violation_reason(services.violation),
            )
        return _audit_json(
            descriptor=descriptor,
            status="blocked",
            services=services,
            summary="governor preparation blocked the operation",
            operation_hash=operation_hash,
            breaker_action=_services_breaker_action(services),
        )

    if type(plan) is ReadOnlyPassThrough:
        try:
            if plan.descriptor is not descriptor:
                raise _GovernanceViolation(
                    "read-only decision references a foreign descriptor"
                )
            if plan.fingerprint != descriptor.fingerprint:
                raise _GovernanceViolation("read-only descriptor fingerprint changed")
            if (
                descriptor.annotations is None
                or descriptor.annotations.get("readOnlyHint") is not True
            ):
                raise _GovernanceViolation(
                    "read-only pass-through requires exact boolean true annotation"
                )
            if (
                services._preflight_used
                or services.preflight_count
                or services.bound_operation is not None
                or services.approval_subject is not None
                or services.approval_receipt is not None
                or services.dispatch_count
            ):
                raise _GovernanceViolation(
                    "read-only pass-through cannot follow governed provider work"
                )
            if not _target_is_current(server, session, descriptor):
                raise _GovernanceViolation(
                    "read-only target changed before legacy dispatch"
                )
            if services.violation is not None:
                raise _GovernanceViolation(
                    "governance capability was used after prepare"
                )
        except BaseException:
            return _audit_json(
                descriptor=descriptor,
                status="blocked",
                services=services,
                summary="read-only pass-through validation failed",
                operation_hash=operation_hash,
                breaker_action=_services_breaker_action(services),
            )
        return _READ_ONLY_PASS_THROUGH

    try:
        if type(plan) is not GovernedDispatchPlan:
            raise _GovernanceViolation("governor returned a malformed plan")
        if plan.bound_operation is not services.bound_operation:
            raise _GovernanceViolation("plan references a foreign operation")
        state = _host_get_bound_operation_state(plan.bound_operation)
        if state.server_name != descriptor.server_name:
            raise _GovernanceViolation("bound server changed")
        if state.invocation_raw_tool_name != descriptor.raw_tool_name:
            raise _GovernanceViolation("bound invocation changed")
        if state.invocation_descriptor_hash != descriptor.fingerprint:
            raise _GovernanceViolation("bound invocation descriptor changed")
        if type(state.target_descriptor) is not MCPToolDescriptor:
            raise _GovernanceViolation("bound target descriptor is malformed")
        if state.target_descriptor.server_name != descriptor.server_name:
            raise _GovernanceViolation("bound target server changed")
        if services.dispatch_tool_name != state.target_descriptor.raw_tool_name:
            raise _GovernanceViolation("bound target name changed")
        if state.preflight_facts != services._preflight_facts:
            raise _GovernanceViolation("bound preflight facts changed")
        if state.interaction_hash != interaction.evidence_hash:
            raise _GovernanceViolation("bound interaction changed")
        if plan.approval_receipt is not services.approval_receipt:
            raise _GovernanceViolation("plan references a foreign approval receipt")
        approval_grant = _host_validate_approval_receipt(
            plan.approval_receipt,
            issuer=services._approval_receipt_issuer,
            bound_operation=plan.bound_operation,
        )
        if services.approval_subject is None:
            raise _GovernanceViolation("fresh approval subject is missing")
        if services._approval_coordinator is None:
            raise _GovernanceViolation("fresh approval issuer is missing")
        patch = _validate_json_mapping(
            "post-approval argument patch",
            plan.post_approval_patch,
            forbid_transport_meta=True,
        )
        meta_patch = _validate_transport_meta(
            "post-approval meta patch",
            plan.post_approval_meta_patch,
        )
        services._remember_transport_meta(meta_patch)
        argument_patch_fields = set(patch)
        meta_patch_fields = set(meta_patch)
        if argument_patch_fields & meta_patch_fields:
            raise _GovernanceViolation(
                "post-approval argument and meta patches overlap"
            )
        if not argument_patch_fields.issubset(
            state.allowed_argument_patch_fields
        ):
            raise _GovernanceViolation(
                "post-approval argument patch contains undeclared fields"
            )
        if not meta_patch_fields.issubset(state.allowed_meta_patch_fields):
            raise _GovernanceViolation(
                "post-approval meta patch contains undeclared fields"
            )
        merged_arguments = _deep_thaw(state.base_arguments)
        merged_arguments.update(_deep_thaw(patch))
        _validate_json_mapping(
            "merged provider arguments",
            merged_arguments,
            forbid_transport_meta=True,
        )
        merged_meta = _deep_thaw(state.base_meta)
        merged_meta.update(_deep_thaw(meta_patch))
        frozen_merged_meta = _validate_transport_meta(
            "merged provider meta", merged_meta
        )
        merged_meta = _deep_thaw(frozen_merged_meta)
        current_interaction = get_current_trusted_interaction()
        if current_interaction is not interaction:
            raise _GovernanceViolation("trusted interaction changed before dispatch")
        if not _target_is_current(server, session, descriptor):
            raise _GovernanceViolation("MCP invocation changed before dispatch")
        if not _target_is_current(server, session, state.target_descriptor):
            raise _GovernanceViolation("MCP dispatch target changed before dispatch")
        services._approval_coordinator.consume(
            approval_grant,
            services.approval_subject,
        )
    except BaseException:
        return _audit_json(
            descriptor=descriptor,
            status="blocked",
            services=services,
            summary="bound plan validation or approval consumption failed",
            operation_hash=operation_hash,
            breaker_action=_services_breaker_action(services),
        )

    async def _dispatch() -> Any:
        async with server._rpc_lock:
            if not _target_is_current(server, session, descriptor):
                raise _GovernanceViolation(
                    "MCP invocation changed after grant consumption"
                )
            if not _target_is_current(server, session, state.target_descriptor):
                raise _GovernanceViolation(
                    "MCP dispatch target changed after grant consumption"
                )
            mark_tool_call = getattr(server, "mark_tool_call", None)
            if callable(mark_tool_call):
                mark_tool_call()
            # The marker is intentionally adjacent to the one SDK call:
            # any failure above this point remains blocked, while every
            # exception/cancellation from the call itself is unknown.
            with services._lock:
                services.dispatch_started = True
                services.dispatch_count = 1
            return await _call_with_pending_context(
                server,
                lambda: session.call_tool(
                    state.target_descriptor.raw_tool_name,
                    arguments=merged_arguments,
                    meta=merged_meta or None,
                ),
            )

    from tools.mcp_tool import _run_on_mcp_loop

    try:
        raw_result = _run_on_mcp_loop(_dispatch, timeout=tool_timeout)
        with services._lock:
            services.transport_succeeded = True
    except BaseException:
        with services._lock:
            if services.dispatch_started:
                services.transport_failed = True
        return _audit_json(
            descriptor=descriptor,
            status="unknown" if services.dispatch_started else "blocked",
            services=services,
            summary=(
                "provider dispatch outcome is unknown"
                if services.dispatch_started
                else "provider dispatch did not start after approval consumption"
            ),
            operation_hash=operation_hash,
            breaker_action=_services_breaker_action(services),
        )

    try:
        readonly_result = _readonly_provider_value(raw_result)
        plugin_outcome = registration.policy.finalize(
            request,
            plan.bound_operation,
            readonly_result,
        )
        if services.violation is not None:
            raise _GovernanceViolation(
                "governance capability was used after prepare"
            )
        status, summary, evidence = _safe_outcome(plugin_outcome)
    except BaseException:
        status = "unknown"
        summary = "provider returned but final verification was unavailable"
        evidence = None
    return _audit_json(
        descriptor=descriptor,
        status=status,
        services=services,
        summary=summary,
        operation_hash=operation_hash,
        evidence=evidence,
        breaker_action=_services_breaker_action(services),
    )


__all__ = ["build_mcp_tool_descriptor", "dispatch_governed_mcp"]
