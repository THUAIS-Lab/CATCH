from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryDecision:
    retry: bool
    delay_seconds: int
    reason: str


TRANSIENT_ERRORS = {
    "network",
    "rate_limit",
    "provider_unavailable",
    "runner_unavailable",
    "timeout",
}


def classify_failure(kind: str, attempt: int, max_attempts: int) -> RetryDecision:
    if kind not in TRANSIENT_ERRORS:
        return RetryDecision(False, 0, "non-transient failure")
    if attempt >= max_attempts:
        return RetryDecision(False, 0, "attempt budget exhausted")
    delay = min(300, 2 ** max(0, attempt))
    return RetryDecision(True, delay, "transient failure")
