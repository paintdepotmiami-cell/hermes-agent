"""Fresh, authenticated, invocation-bound approval primitives.

This module is intentionally independent of tools, transports, and products.
Hosts derive authenticated responder data at their trusted transport boundary;
future dispatch code can then request, await, and consume an exact one-use
grant without extending the model tool schema. The responder issuer is a
private host API seam: that separation keeps it out of a future plugin facade,
but does not claim memory isolation from trusted in-process Python plugins.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


Assurance = Literal["telegram_authenticated", "local_desktop_session"]
Decision = Literal["approve_once", "deny"]
Outcome = Literal["approved", "denied", "expired", "cancelled"]
_MAX_SAFE_INTEGER = (1 << 53) - 1


class ApprovalRejected(ValueError):
    """Raised when evidence or a grant cannot satisfy the exact request."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason.replace("_", " "))


@dataclass(frozen=True)
class ConversationBinding:
    """Exact host routing, conversation, session, and optional actor identity."""

    profile_id: str
    platform: str
    channel_id: str
    chat_id: str
    session_key: str
    session_id: str
    thread_id: Optional[str] = None
    actor_id: Optional[str] = None


@dataclass(frozen=True)
class InvocationSubject:
    """Immutable identity of the proposed invocation being approved."""

    invocation_id: str
    tool_call_id: str
    turn_id: str
    origin_message_id: str
    args_hash: str
    proposal_hash: str
    conversation: ConversationBinding


@dataclass(frozen=True)
class AuthenticatedResponder:
    """Responder facts derived by an authenticated host transport."""

    principal_id: Optional[str]
    assurance: Assurance
    binding: ConversationBinding
    inbound_id: str
    host_nonce: str
    authenticated_at: float
    _issuer: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class AuthenticatedResponderFacts:
    """Evidence-safe responder facts with no private issuer capability."""

    principal_id: Optional[str]
    assurance: Assurance
    binding: ConversationBinding
    inbound_id: str
    host_nonce: str
    authenticated_at: float


@dataclass(frozen=True)
class ApprovalRequest:
    """A random, exact-subject request that accepts only fresh evidence."""

    approval_id: str
    subject: InvocationSubject
    mode: Literal["fresh"]
    ttl_seconds: float
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class ApprovalEvidence:
    """Authenticated evidence submitted for one exact approval ID."""

    approval_id: str
    decision: Decision
    responder: AuthenticatedResponderFacts
    confirmed_at: float
    evidence_hash: str


class ApprovalGrant:
    """Opaque capability returned only for an accepted one-time approval."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<ApprovalGrant opaque>"


@dataclass(frozen=True)
class ApprovalResolution:
    """Terminal result of a fresh approval request."""

    approval_id: str
    subject: InvocationSubject
    outcome: Outcome
    evidence: Optional[ApprovalEvidence] = None
    grant: Optional[ApprovalGrant] = None


@dataclass
class _Record:
    request: ApprovalRequest
    resolution: Optional[ApprovalResolution] = None
    tombstone_expires_at: float = 0.0


def _require_safe_number(name: str, value: object) -> int | float:
    """Return a finite JSON-safe number, rejecting lossy integer values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a safe integer or finite float")
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError(f"{name} exceeds the safe integer range")
        return value
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _binding_payload(binding: ConversationBinding) -> dict[str, object]:
    return {
        "actor_id": binding.actor_id,
        "channel_id": binding.channel_id,
        "chat_id": binding.chat_id,
        "platform": binding.platform,
        "profile_id": binding.profile_id,
        "session_id": binding.session_id,
        "session_key": binding.session_key,
        "thread_id": binding.thread_id,
    }


def _evidence_payload(
    request: ApprovalRequest,
    decision: Decision,
    responder: AuthenticatedResponderFacts,
    confirmed_at: float,
) -> dict[str, object]:
    binding = _binding_payload(request.subject.conversation)
    return {
        "approval_id": request.approval_id,
        "confirmed_at": _require_safe_number("confirmed_at", confirmed_at),
        "decision": decision,
        "request": {
            "created_at": _require_safe_number(
                "request.created_at", request.created_at
            ),
            "expires_at": _require_safe_number(
                "request.expires_at", request.expires_at
            ),
        },
        "responder": {
            "assurance": responder.assurance,
            "authenticated_at": _require_safe_number(
                "responder.authenticated_at", responder.authenticated_at
            ),
            "binding": _binding_payload(responder.binding),
            "host_nonce": responder.host_nonce,
            "inbound_id": responder.inbound_id,
            "principal_id": responder.principal_id,
        },
        "subject": {
            "args_hash": request.subject.args_hash,
            "conversation": binding,
            "invocation_id": request.subject.invocation_id,
            "origin_message_id": request.subject.origin_message_id,
            "proposal_hash": request.subject.proposal_hash,
            "tool_call_id": request.subject.tool_call_id,
            "turn_id": request.subject.turn_id,
        },
    }


def _canonical_evidence_hash(
    request: ApprovalRequest,
    decision: Decision,
    responder: AuthenticatedResponderFacts,
    confirmed_at: float,
) -> str:
    encoded = json.dumps(
        _evidence_payload(request, decision, responder, confirmed_at),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FreshApprovalCoordinator:
    """Thread-safe coordinator for fresh, exact, one-use approvals."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        replay_ttl: float = 300.0,
    ) -> None:
        self._clock = clock
        self._replay_ttl = _require_safe_number("replay_ttl", replay_ttl)
        if self._replay_ttl <= 0:
            raise ValueError("replay_ttl must be positive")
        self._issuer = object()
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._records: dict[str, _Record] = {}
        self._grants: dict[ApprovalGrant, tuple[InvocationSubject, float]] = {}
        self._seen_responder_nonces: dict[str, float] = {}
        self._seen_inbound: dict[tuple[Assurance, ConversationBinding, str], float] = {}

    def _prune_locked(self, now: int | float) -> None:
        """Expire live state and prune elapsed replay tombstones under the lock."""

        for approval_id, record in list(self._records.items()):
            if record.resolution is None and now >= record.request.expires_at:
                record.resolution = ApprovalResolution(
                    approval_id=approval_id,
                    subject=record.request.subject,
                    outcome="expired",
                )
                self._changed.notify_all()
            if record.resolution is not None and now >= record.tombstone_expires_at:
                self._records.pop(approval_id, None)

        for grant, (_subject, expires_at) in list(self._grants.items()):
            if now >= expires_at:
                self._grants.pop(grant, None)

        for nonce, expires_at in list(self._seen_responder_nonces.items()):
            if now >= expires_at:
                self._seen_responder_nonces.pop(nonce, None)
        for inbound_key, expires_at in list(self._seen_inbound.items()):
            if now >= expires_at:
                self._seen_inbound.pop(inbound_key, None)

    def request(
        self,
        subject: InvocationSubject,
        *,
        ttl_seconds: float,
    ) -> ApprovalRequest:
        ttl_seconds = _require_safe_number("ttl_seconds", ttl_seconds)
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        for field_name in (
            "invocation_id",
            "tool_call_id",
            "turn_id",
            "origin_message_id",
            "args_hash",
            "proposal_hash",
        ):
            value = getattr(subject, field_name, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in (
            "profile_id",
            "platform",
            "channel_id",
            "chat_id",
            "session_key",
            "session_id",
        ):
            value = getattr(subject.conversation, field_name, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("thread_id", "actor_id"):
            value = getattr(subject.conversation, field_name, None)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be None or a non-empty string")
        now = _require_safe_number("created_at", self._clock())
        expires_at = _require_safe_number("expires_at", now + ttl_seconds)
        with self._changed:
            self._prune_locked(now)
            approval_id = f"fa_{secrets.token_urlsafe(18)}"
            while approval_id in self._records:
                approval_id = f"fa_{secrets.token_urlsafe(18)}"
            request = ApprovalRequest(
                approval_id=approval_id,
                subject=subject,
                mode="fresh",
                ttl_seconds=ttl_seconds,
                created_at=now,
                expires_at=expires_at,
            )
            self._records[request.approval_id] = _Record(
                request=request,
                tombstone_expires_at=_require_safe_number(
                    "tombstone_expires_at",
                    expires_at + self._replay_ttl,
                ),
            )
            self._changed.notify_all()
        return request

    def _mint_authenticated_responder_for_host(
        self,
        *,
        assurance: Assurance,
        binding: ConversationBinding,
        inbound_id: str,
        principal_id: Optional[str] = None,
    ) -> AuthenticatedResponder:
        """Mint responder facts at a trusted host boundary.

        This underscore-prefixed method is API separation, not an impossible
        memory-isolation boundary against trusted in-process Python plugins.
        """
        if assurance not in {"telegram_authenticated", "local_desktop_session"}:
            raise ApprovalRejected("unsupported_assurance")
        if not isinstance(inbound_id, str) or not inbound_id.strip():
            raise ApprovalRejected("missing_inbound_id")
        authenticated_at = _require_safe_number("authenticated_at", self._clock())
        with self._lock:
            self._prune_locked(authenticated_at)
            return AuthenticatedResponder(
                principal_id=principal_id,
                assurance=assurance,
                binding=binding,
                inbound_id=inbound_id,
                host_nonce=secrets.token_urlsafe(18),
                authenticated_at=authenticated_at,
                _issuer=self._issuer,
            )

    def resolve(
        self,
        approval_id: str,
        *,
        decision: Decision,
        responder: Optional[AuthenticatedResponder],
    ) -> ApprovalResolution:
        with self._changed:
            if decision not in {"approve_once", "deny"}:
                raise ApprovalRejected("invalid_decision")
            now = _require_safe_number("confirmed_at", self._clock())
            self._prune_locked(now)
            record = self._records.get(approval_id)
            if record is None:
                raise ApprovalRejected("unknown_approval_id")
            if record.resolution is not None:
                if record.resolution.outcome == "expired":
                    raise ApprovalRejected("approval_expired")
                raise ApprovalRejected("approval_already_resolved")

            if now >= record.request.expires_at:
                record.resolution = ApprovalResolution(
                    approval_id=approval_id,
                    subject=record.request.subject,
                    outcome="expired",
                )
                self._changed.notify_all()
                raise ApprovalRejected("approval_expired")
            if responder is None:
                raise ApprovalRejected("missing_authenticated_responder")
            if responder._issuer is not self._issuer:
                raise ApprovalRejected("untrusted_responder")

            subject = record.request.subject
            if responder.binding != subject.conversation:
                raise ApprovalRejected("conversation_binding_mismatch")
            if responder.assurance == "telegram_authenticated":
                if (
                    responder.binding.platform != "telegram"
                    or not responder.principal_id
                    or responder.principal_id != subject.conversation.actor_id
                ):
                    raise ApprovalRejected("actor_binding_mismatch")
            elif responder.assurance == "local_desktop_session":
                if (
                    responder.binding.platform not in {"local", "local_desktop"}
                    or responder.principal_id is not None
                ):
                    raise ApprovalRejected("local_responder_must_be_unnamed")
                # Legacy direct fresh approvals use an unnamed
                # ``local_desktop`` binding. Host-issued action provenance uses
                # ``local`` plus the opaque transport identity as its exact
                # actor/channel binding; the TUI gateway validates that value
                # against the live server-side transport before minting this
                # responder.
                if (
                    responder.binding.platform == "local_desktop"
                    and responder.binding.actor_id is not None
                ):
                    raise ApprovalRejected("local_responder_must_be_unnamed")
            else:
                raise ApprovalRejected("unsupported_assurance")
            if responder.inbound_id == subject.origin_message_id:
                raise ApprovalRejected("origin_message_cannot_approve")
            if responder.authenticated_at < record.request.created_at:
                raise ApprovalRejected("responder_not_fresh_for_request")
            if responder.authenticated_at > now:
                raise ApprovalRejected("responder_time_is_in_the_future")
            inbound_key = (
                responder.assurance,
                responder.binding,
                responder.inbound_id,
            )
            if (
                responder.host_nonce in self._seen_responder_nonces
                or inbound_key in self._seen_inbound
            ):
                raise ApprovalRejected("responder_replay")
            replay_expires_at = _require_safe_number(
                "replay_expires_at",
                max(
                    record.request.expires_at,
                    now + self._replay_ttl,
                ),
            )
            self._seen_responder_nonces[responder.host_nonce] = replay_expires_at
            self._seen_inbound[inbound_key] = replay_expires_at

            responder_facts = AuthenticatedResponderFacts(
                principal_id=responder.principal_id,
                assurance=responder.assurance,
                binding=responder.binding,
                inbound_id=responder.inbound_id,
                host_nonce=responder.host_nonce,
                authenticated_at=responder.authenticated_at,
            )
            evidence = ApprovalEvidence(
                approval_id=approval_id,
                decision=decision,
                responder=responder_facts,
                confirmed_at=now,
                evidence_hash=_canonical_evidence_hash(
                    record.request,
                    decision,
                    responder_facts,
                    now,
                ),
            )
            if decision == "deny":
                resolution = ApprovalResolution(
                    approval_id=approval_id,
                    subject=record.request.subject,
                    outcome="denied",
                    evidence=evidence,
                )
            else:
                grant = ApprovalGrant()
                self._grants[grant] = (
                    record.request.subject,
                    record.request.expires_at,
                )
                resolution = ApprovalResolution(
                    approval_id=approval_id,
                    subject=record.request.subject,
                    outcome="approved",
                    evidence=evidence,
                    grant=grant,
                )
            record.resolution = resolution
            record.tombstone_expires_at = replay_expires_at
            self._changed.notify_all()
            return resolution

    def verify_evidence(self, evidence: ApprovalEvidence) -> bool:
        """Verify canonical evidence against the retained exact request record."""

        if not isinstance(evidence, ApprovalEvidence):
            return False
        with self._lock:
            now = _require_safe_number("verification_time", self._clock())
            self._prune_locked(now)
            record = self._records.get(evidence.approval_id)
            if (
                record is None
                or record.resolution is None
                or record.resolution.evidence != evidence
            ):
                return False
            try:
                expected = _canonical_evidence_hash(
                    record.request,
                    evidence.decision,
                    evidence.responder,
                    evidence.confirmed_at,
                )
            except (TypeError, ValueError):
                return False
            return hmac.compare_digest(evidence.evidence_hash, expected)

    def cancel(self, approval_id: str) -> ApprovalResolution:
        """Cancel one exact pending request and wake every waiter."""

        with self._changed:
            now = _require_safe_number("cancelled_at", self._clock())
            self._prune_locked(now)
            record = self._records.get(approval_id)
            if record is None:
                raise ApprovalRejected("unknown_approval_id")
            if record.resolution is not None:
                raise ApprovalRejected("approval_already_resolved")
            resolution = ApprovalResolution(
                approval_id=approval_id,
                subject=record.request.subject,
                outcome="cancelled",
            )
            record.resolution = resolution
            record.tombstone_expires_at = _require_safe_number(
                "tombstone_expires_at",
                max(
                    record.request.expires_at,
                    now + self._replay_ttl,
                ),
            )
            self._changed.notify_all()
            return resolution

    def await_resolution(
        self,
        approval_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[ApprovalResolution]:
        """Block a worker thread until the exact request reaches a terminal state."""

        with self._changed:
            now = _require_safe_number("awaited_at", self._clock())
            self._prune_locked(now)
            record = self._records.get(approval_id)
            if record is None:
                raise ApprovalRejected("unknown_approval_id")
            wait_deadline = (
                None if timeout is None else time.monotonic() + max(0, timeout)
            )
            while record.resolution is None:
                now = _require_safe_number("awaited_at", self._clock())
                if now >= record.request.expires_at:
                    record.resolution = ApprovalResolution(
                        approval_id=approval_id,
                        subject=record.request.subject,
                        outcome="expired",
                    )
                    self._changed.notify_all()
                    break
                wait_for = max(0.0, record.request.expires_at - now)
                if wait_deadline is not None:
                    remaining = wait_deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    wait_for = min(wait_for, remaining)
                self._changed.wait(timeout=wait_for)
            return record.resolution

    def get_request(self, approval_id: str) -> ApprovalRequest:
        """Return the exact host-owned request used to derive responder binding."""

        with self._lock:
            now = _require_safe_number("lookup_time", self._clock())
            self._prune_locked(now)
            record = self._records.get(approval_id)
            if record is None:
                raise ApprovalRejected("unknown_approval_id")
            return record.request

    async def await_resolution_async(
        self,
        approval_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[ApprovalResolution]:
        """Await a request without blocking the caller's event-loop thread."""

        return await asyncio.to_thread(
            self.await_resolution,
            approval_id,
            timeout=timeout,
        )

    def consume(self, grant: ApprovalGrant, subject: InvocationSubject) -> bool:
        with self._lock:
            now = _require_safe_number("consume_time", self._clock())
            self._prune_locked(now)
            if not isinstance(grant, ApprovalGrant):
                raise ApprovalRejected("foreign_approval_grant")
            grant_record = self._grants.get(grant)
            if grant_record is None:
                raise ApprovalRejected("unknown_or_consumed_approval_grant")
            expected, expires_at = grant_record
            if now >= expires_at:
                self._grants.pop(grant)
                raise ApprovalRejected("approval_grant_expired")
            if expected != subject:
                raise ApprovalRejected("approval_grant_subject_mismatch")
            self._grants.pop(grant)
            return True


_DEFAULT_COORDINATOR = FreshApprovalCoordinator()


def get_fresh_approval_coordinator() -> FreshApprovalCoordinator:
    """Return the process-local coordinator shared by trusted host surfaces."""

    return _DEFAULT_COORDINATOR
