from typing import Any, Protocol
from .models import Result, Task, VerificationStatus

class Verifier(Protocol):
    name: str
    def verify(self, task: Task, output: Any) -> Result: ...

class ExactVerifier:
    name = "exact"
    def __init__(self, expected: Any) -> None:
        self.expected = expected
    def verify(self, task: Task, output: Any) -> Result:
        status = VerificationStatus.VERIFIED if output == self.expected else VerificationStatus.FAILED
        return Result(status=status, output=output, metadata={"verifier": self.name})

class ConservativeIndependentVerifier:
    name = "independent-conservative"
    def verify(self, task: Task, output: Any) -> Result:
        return Result(status=VerificationStatus.UNKNOWN, output=output, metadata={"verifier": self.name})
