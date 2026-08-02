"""Public, product-neutral contracts for two-phase governed MCP dispatch.

Plugins receive immutable snapshots and a narrow stateful host capability.
They never receive the provider callable, MCP session, approval coordinator,
or secret values.  Host-only constructors and inspectors are underscored: the
separation makes the supported plugin API explicit, but is not a claim of
memory isolation from trusted in-process Python code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import threading
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from agent.trusted_interaction import TrustedInteraction
from tools.fresh_approval import ApprovalGrant


GovernedStatus = Literal["verified", "blocked", "unknown"]
_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


def _deep_freeze(value: Any) -> Any:
    """Detach and recursively freeze JSON-shaped public contract data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("immutable contract object keys must be strings")
            frozen[key] = _deep_freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    raise TypeError(f"unsupported immutable contract value: {type(value).__name__}")


def _deep_thaw(value: Any) -> Any:
    """Return a detached mutable JSON-shaped copy of frozen contract data."""

    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return sorted((_deep_thaw(item) for item in value), key=repr)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _deep_thaw(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """Exact immutable metadata snapshot for one discovered raw MCP tool."""

    server_name: str
    raw_tool_name: str
    normalized_name: str
    description: str | None
    input_schema: Mapping[str, Any]
    annotations: Mapping[str, Any] | None
    meta: Mapping[str, Any] | None
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", _deep_freeze(self.input_schema))
        if self.annotations is not None:
            object.__setattr__(
                self, "annotations", _deep_freeze(self.annotations)
            )
        if self.meta is not None:
            object.__setattr__(self, "meta", _deep_freeze(self.meta))


@dataclass(frozen=True, slots=True)
class MCPGovernanceRequest:
    """Read-only exact invocation facts supplied to a governor policy."""

    descriptor: MCPToolDescriptor
    arguments: Mapping[str, Any]
    interaction: TrustedInteraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _deep_freeze(self.arguments))


@dataclass(frozen=True, slots=True)
class ReadOnlyPassThrough:
    """Authorize one exact annotated read through the legacy MCP handler.

    The decision carries no provider callable or downstream capability.  It is
    valid only for the same descriptor object supplied in the current
    :class:`MCPGovernanceRequest`; the host independently verifies the frozen
    fingerprint and the SDK annotation before honoring it.
    """

    descriptor: MCPToolDescriptor
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.descriptor) is not MCPToolDescriptor:
            raise TypeError("read-only pass-through requires an MCP tool descriptor")
        object.__setattr__(self, "fingerprint", self.descriptor.fingerprint)


@dataclass(frozen=True, slots=True)
class _BoundOperationState:
    server_name: str
    invocation_raw_tool_name: str
    invocation_descriptor_hash: str
    target_descriptor: MCPToolDescriptor
    preflight_facts: Mapping[str, Any]
    base_arguments: Mapping[str, Any]
    base_meta: Mapping[str, Any]
    proposal_hash: str
    preview: str
    display_text: str
    interaction_hash: str
    operation_hash: str
    allowed_argument_patch_fields: frozenset[str]
    allowed_meta_patch_fields: frozenset[str]


_BOUND_OPERATION_ISSUER = object()


class BoundOperation:
    """Opaque, immutable host-owned operation approved as one exact unit."""

    __slots__ = ("__issuer", "__state")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("BoundOperation is host-owned and cannot be constructed")

    def __repr__(self) -> str:
        return "<BoundOperation opaque>"

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("BoundOperation is immutable")


def _host_issue_bound_operation(state: _BoundOperationState) -> BoundOperation:
    if not isinstance(state, _BoundOperationState):
        raise TypeError("invalid bound operation state")
    operation = object.__new__(BoundOperation)
    object.__setattr__(operation, "_BoundOperation__issuer", _BOUND_OPERATION_ISSUER)
    object.__setattr__(operation, "_BoundOperation__state", state)
    return operation


def _host_get_bound_operation_state(operation: object) -> _BoundOperationState:
    if not isinstance(operation, BoundOperation):
        raise TypeError("plan must reference a host-owned BoundOperation")
    issuer = object.__getattribute__(operation, "_BoundOperation__issuer")
    state = object.__getattribute__(operation, "_BoundOperation__state")
    if issuer is not _BOUND_OPERATION_ISSUER or not isinstance(
        state, _BoundOperationState
    ):
        raise TypeError("plan contains a foreign BoundOperation")
    return state


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class ApprovalReceipt:
    """Evidence-safe proof of one host-approved bound operation.

    The identifiers and timestamp are copied from verified host fresh-approval
    evidence.  ``grant`` is an opaque, one-use capability.  Responder facts,
    conversation bindings, and replay nonces are deliberately absent.
    """

    approval_request_id: str
    approval_message_id: str
    approval_event_id: str
    approved_at: str
    evidence_hash: str
    grant: ApprovalGrant

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("ApprovalReceipt is host-issued and cannot be constructed")


@dataclass(frozen=True, slots=True)
class _ApprovalReceiptState:
    issuer: object
    bound_operation: BoundOperation
    snapshot: tuple[object, ...]


_APPROVAL_RECEIPT_STATES: WeakKeyDictionary[
    ApprovalReceipt, _ApprovalReceiptState
] = WeakKeyDictionary()
_APPROVAL_RECEIPT_LOCK = threading.RLock()


def _host_issue_approval_receipt(
    *,
    issuer: object,
    bound_operation: BoundOperation,
    approval_request_id: str,
    approval_message_id: str,
    approval_event_id: str,
    approved_at: str,
    evidence_hash: str,
    grant: ApprovalGrant,
) -> ApprovalReceipt:
    """Issue one receipt and seal its exact public values to host state."""

    if issuer is None:
        raise TypeError("approval receipt issuer is required")
    _host_get_bound_operation_state(bound_operation)
    values = (
        approval_request_id,
        approval_message_id,
        approval_event_id,
        approved_at,
        evidence_hash,
        grant,
    )
    for name, value in zip(
        (
            "approval_request_id",
            "approval_message_id",
            "approval_event_id",
            "approved_at",
            "evidence_hash",
        ),
        values[:-1],
        strict=True,
    ):
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
    if type(grant) is not ApprovalGrant:
        raise TypeError("approval receipt grant must be host-issued")

    receipt = object.__new__(ApprovalReceipt)
    for name, value in zip(
        (
            "approval_request_id",
            "approval_message_id",
            "approval_event_id",
            "approved_at",
            "evidence_hash",
            "grant",
        ),
        values,
        strict=True,
    ):
        object.__setattr__(receipt, name, value)
    with _APPROVAL_RECEIPT_LOCK:
        _APPROVAL_RECEIPT_STATES[receipt] = _ApprovalReceiptState(
            issuer=issuer,
            bound_operation=bound_operation,
            snapshot=values,
        )
    return receipt


def _host_validate_approval_receipt(
    receipt: object,
    *,
    issuer: object,
    bound_operation: BoundOperation,
) -> ApprovalGrant:
    """Validate issuer, operation, and every sealed public receipt value."""

    if type(receipt) is not ApprovalReceipt:
        raise TypeError("plan must carry a host-issued ApprovalReceipt")
    with _APPROVAL_RECEIPT_LOCK:
        state = _APPROVAL_RECEIPT_STATES.get(receipt)
    if state is None:
        raise TypeError("plan contains a fabricated ApprovalReceipt") from None
    current = (
        receipt.approval_request_id,
        receipt.approval_message_id,
        receipt.approval_event_id,
        receipt.approved_at,
        receipt.evidence_hash,
        receipt.grant,
    )
    if state.issuer is not issuer:
        raise TypeError("plan contains a foreign ApprovalReceipt")
    if state.bound_operation is not bound_operation:
        raise TypeError("approval receipt is bound to a different operation")
    if current != state.snapshot:
        raise TypeError("approval receipt was altered")
    if type(receipt.grant) is not ApprovalGrant:
        raise TypeError("approval receipt grant is malformed")
    return receipt.grant


@dataclass(frozen=True, slots=True)
class GovernedDispatchPlan:
    """A bound operation, its exact receipt, and a constrained final patch."""

    bound_operation: BoundOperation
    approval_receipt: ApprovalReceipt
    post_approval_patch: Mapping[str, Any]
    post_approval_meta_patch: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "post_approval_patch",
            _deep_freeze(self.post_approval_patch),
        )
        object.__setattr__(
            self,
            "post_approval_meta_patch",
            _deep_freeze(self.post_approval_meta_patch),
        )


@dataclass(frozen=True, slots=True)
class GovernedOutcome:
    """Sanitized policy classification of the single provider response."""

    status: GovernedStatus
    summary: str
    evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"verified", "blocked", "unknown"}:
            raise ValueError("invalid governed outcome status")
        if not isinstance(self.summary, str):
            raise TypeError("governed outcome summary must be a string")
        if self.evidence is not None:
            object.__setattr__(self, "evidence", _deep_freeze(self.evidence))


@runtime_checkable
class GovernedMCPHostServices(Protocol):
    """Stateful, single-use host capability available during policy prepare."""

    def preflight_call(
        self,
        raw_tool_name: str,
        arguments: Mapping[str, Any],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def bind_operation(
        self,
        raw_tool_name: str,
        base_arguments: Mapping[str, Any],
        *,
        proposal_hash: str,
        preview: str,
        display_text: str,
        base_meta: Mapping[str, Any] = _EMPTY_MAPPING,
        allowed_post_approval_patch_fields: Sequence[str] = (),
        allowed_post_approval_meta_fields: Sequence[str] = (),
    ) -> BoundOperation: ...

    def request_fresh_approval(
        self, operation: BoundOperation
    ) -> ApprovalReceipt: ...

    def hmac_sha256(
        self, secret_env_name: str, canonical_payload: bytes | str
    ) -> str: ...


@runtime_checkable
class MCPGovernorPolicy(Protocol):
    """Neutral two-phase policy implemented by a plugin."""

    def prepare(
        self,
        request: MCPGovernanceRequest,
        services: GovernedMCPHostServices,
    ) -> GovernedDispatchPlan | ReadOnlyPassThrough: ...

    def finalize(
        self,
        request: MCPGovernanceRequest,
        operation: BoundOperation,
        result: Any,
    ) -> GovernedOutcome: ...


__all__ = [
    "ApprovalReceipt",
    "BoundOperation",
    "GovernedDispatchPlan",
    "GovernedMCPHostServices",
    "GovernedOutcome",
    "GovernedStatus",
    "MCPGovernanceRequest",
    "MCPGovernorPolicy",
    "MCPToolDescriptor",
    "ReadOnlyPassThrough",
]
