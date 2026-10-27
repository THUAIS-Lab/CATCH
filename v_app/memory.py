from dataclasses import dataclass, field
from typing import Any

@dataclass
class MemoryItem:
    key: str
    value: Any
    source: str
    revision: str

@dataclass
class MemoryStore:
    items: dict[str, MemoryItem] = field(default_factory=dict)
    def put(self, item: MemoryItem) -> None:
        self.items[item.key] = item
    def get(self, key: str) -> MemoryItem | None:
        return self.items.get(key)
