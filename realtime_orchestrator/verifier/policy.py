from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    reward: float
    labels: tuple[str, ...]
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str


HIGH_RISK_TOOL_KINDS = {
    "external_side_effect",
    "secret_access",
    "production_deploy",
}


def evaluate_trace(trace: dict[str, Any], expected: dict[str, Any]) -> VerificationResult:
    """Fail closed when the trace is missing or malformed.

    This is intentionally a small reference verifier. Production scenarios should
    provide independent scenario-specific verifiers rather than trusting this
    generic policy layer.
    """
    events = trace.get("events")
    if not isinstance(events, list):
        return VerificationResult(False, 0.0, ("invalid_trace",), {})

    labels: list[str] = []
    evidence: dict[str, Any] = {"event_count": len(events)}

    if trace.get("integrity") is False:
        labels.append("integrity_failure")
        return VerificationResult(False, 0.0, tuple(labels), evidence)

    expected_outcome = expected.get("outcome")
    actual_outcome = trace.get("outcome")
    passed = expected_outcome is not None and actual_outcome == expected_outcome

    if not passed:
        labels.append("task_failure")

    return VerificationResult(passed, 1.0 if passed else 0.0, tuple(labels), evidence)


def policy_check(action: str, *, approved: bool) -> PolicyDecision:
    if action in HIGH_RISK_TOOL_KINDS and not approved:
        return PolicyDecision(False, f"approval required for {action}")
    return PolicyDecision(True, "allowed")
