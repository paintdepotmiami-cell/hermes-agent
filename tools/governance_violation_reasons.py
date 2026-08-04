"""Sanitize host-owned governed MCP violation reasons for telemetry."""

from __future__ import annotations

from typing import Final


UNRECOGNIZED_VIOLATION_REASON: Final = "unrecognized_reason"

GOVERNANCE_VIOLATION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "governance capability cannot be used from a detached context",
        "governance capability is closed",
        "only one preflight call is permitted",
        "preflight raw tool name must be non-empty",
        "MCP target changed before preflight",
        "preflight must target a real raw tool on the same server",
        "preflight outcome is unknown",
        "preflight could not start",
        "only one operation may be bound",
        "bound operation target is not one exact registered raw tool on the current server",
        "approval must reference this service's bound operation",
        "only one fresh approval may be requested",
        "fresh approval notifier is unavailable",
        "fresh approval timed out",
        "fresh approval was not granted",
        "fresh approval evidence could not issue a receipt",
        "HMAC secret is not declared for this governor",
        "HMAC payload must be bytes or string",
        "HMAC payload is too large",
        "HMAC secret is unavailable",
    }
)


def sanitized_violation_reason(value: object) -> str:
    if isinstance(value, str) and value in GOVERNANCE_VIOLATION_REASONS:
        return value
    return UNRECOGNIZED_VIOLATION_REASON
