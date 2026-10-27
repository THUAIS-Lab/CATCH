from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class Schedule:
    name: str
    cron: str
    timezone: str = "UTC"
    enabled: bool = True
    max_concurrent: int = 1


@dataclass(frozen=True, slots=True)
class ScheduleTick:
    schedule: str
    fired_at: datetime
    dedupe_key: str


def tick(schedule: Schedule, *, now: datetime | None = None) -> ScheduleTick:
    if not schedule.enabled:
        raise ValueError("schedule is disabled")
    current = now or datetime.now(timezone.utc)
    bucket = current.strftime("%Y%m%d%H%M")
    return ScheduleTick(schedule.name, current, f"{schedule.name}:{bucket}")
