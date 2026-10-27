from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EventKind = Literal[
    "session", "conversation", "audio", "response", "tool",
    "interrupt", "verification", "reward", "policy", "automation",
]


@dataclass(slots=True)
class Event:
    run_id: str
    sequence: int
    kind: EventKind
    type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()


@dataclass(slots=True)
class Trace:
    run_id: str
    events: list[Event] = field(default_factory=list)

    def append(self, event: Event) -> None:
        if event.run_id != self.run_id:
            raise ValueError("event run_id does not match trace")
        expected = len(self.events)
        if event.sequence != expected:
            raise ValueError(f"non-contiguous event sequence: expected {expected}")
        self.events.append(event)

    def verify_integrity(self) -> bool:
        return all(
            event.run_id == self.run_id and event.sequence == idx
            for idx, event in enumerate(self.events)
        )
