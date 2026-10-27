"""Cache is an evidence store, not a shortcut around verification."""
from dataclasses import asdict
from hashlib import sha256
import json
from .models import ExecutionClass, Result, Task, VerificationStatus

class VerifiedCache:
    """Deterministic cache that can only return verified, reusable results."""
    def __init__(self) -> None:
        self._items: dict[str, Result] = {}

    @staticmethod
    def key(task: Task) -> str:
        payload = asdict(task)
        payload["execution_class"] = task.execution_class.value
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(encoded.encode()).hexdigest()

    def get(self, task: Task) -> Result | None:
        if task.execution_class in {ExecutionClass.EXTERNAL_SIDE_EFFECT, ExecutionClass.SECRET_ACCESS, ExecutionClass.PRODUCTION_DEPLOY}:
            return None
        item = self._items.get(self.key(task))
        if item is None or item.status is not VerificationStatus.VERIFIED:
            return None
        return item

    def put_verified(self, task: Task, result: Result) -> str:
        if result.status is not VerificationStatus.VERIFIED:
            raise ValueError("Only VERIFIED results may enter the reusable cache")
        key = self.key(task)
        result.cache_key = key
        result.artifact_id = sha256((key + json.dumps(result.output, sort_keys=True, default=str)).encode()).hexdigest()
        self._items[key] = result
        return key
