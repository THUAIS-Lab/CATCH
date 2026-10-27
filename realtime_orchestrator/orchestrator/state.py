from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import time


class RunState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    VERIFYING = "verifying"
    AUDITING = "auditing"
    BLOCKED = "blocked"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Run:
    run_id: str
    scenario_id: str
    agent_id: str
    state: RunState = RunState.QUEUED
    attempt: int = 0
    parent_run_id: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    labels: set[str] = field(default_factory=set)

    def transition(self, state: RunState) -> None:
        self.state = state
        self.updated_at = time()


ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.QUEUED: {RunState.PREPARING, RunState.CANCELLED},
    RunState.PREPARING: {RunState.RUNNING, RunState.BLOCKED, RunState.FAILED},
    RunState.RUNNING: {RunState.VERIFYING, RunState.RETRYING, RunState.CANCELLED, RunState.FAILED},
    RunState.VERIFYING: {RunState.AUDITING, RunState.SUCCEEDED, RunState.FAILED},
    RunState.AUDITING: {RunState.SUCCEEDED, RunState.FAILED, RunState.RETRYING},
    RunState.BLOCKED: {RunState.PREPARING, RunState.CANCELLED},
    RunState.RETRYING: {RunState.PREPARING, RunState.CANCELLED, RunState.FAILED},
    RunState.SUCCEEDED: set(),
    RunState.FAILED: set(),
    RunState.CANCELLED: set(),
}


def transition(run: Run, target: RunState) -> Run:
    if target not in ALLOWED_TRANSITIONS[run.state]:
        raise ValueError(f"invalid transition: {run.state} -> {target}")
    run.transition(target)
    return run
