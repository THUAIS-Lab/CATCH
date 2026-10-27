from dataclasses import dataclass, field
from typing import Any

@dataclass
class RealtimeSession:
    session_id: str
    connected: bool = False
    interrupted: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    def connect(self) -> None:
        self.connected = True
        self.events.append({"type": "session.connected"})
    def disconnect(self) -> None:
        self.connected = False
        self.events.append({"type": "session.disconnected"})
    def interrupt(self) -> None:
        self.interrupted = True
        self.events.append({"type": "response.interrupted"})
    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)
