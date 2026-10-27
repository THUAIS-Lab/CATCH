from typing import Any, Callable
from .cache import VerifiedCache
from .models import Result, Task, VerificationStatus
from .policy import Policy
from .verifier import ConservativeIndependentVerifier, Verifier

class AgentRuntime:
    """Control plane: policy -> cache -> worker -> visible verifier -> independent verifier -> cache."""
    def __init__(self, cache: VerifiedCache | None = None, policy: Policy | None = None) -> None:
        self.cache = cache or VerifiedCache()
        self.policy = policy or Policy()
        self.independent_verifier: Verifier = ConservativeIndependentVerifier()

    def execute(self, task: Task, worker: Callable[[Task], Any], verifier: Verifier) -> Result:
        self.policy.check(task)
        cached = self.cache.get(task)
        if cached is not None:
            cached.metadata["cache_hit"] = True
            return cached
        output = worker(task)
        visible = verifier.verify(task, output)
        independent = self.independent_verifier.verify(task, output)
        if visible.status is not independent.status:
            return Result(status=VerificationStatus.UNKNOWN, output=output, verifier_disagreement=True,
                          metadata={"visible": visible.status.value, "independent": independent.status.value})
        if visible.status is not VerificationStatus.VERIFIED:
            return visible
        self.cache.put_verified(task, visible)
        visible.metadata["cache_hit"] = False
        return visible
