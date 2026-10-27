from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNKNOWN = "unknown"

class ExecutionClass(str, Enum):
    READ_ONLY = "read_only"
    COMPUTE = "compute"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    SECRET_ACCESS = "secret_access"
    PRODUCTION_DEPLOY = "production_deploy"

@dataclass(frozen=True)
class Task:
    task_id: str
    input: Any
    model: str
    model_revision: str = "unknown"
    code_revision: str = "unknown"
    tool_revision: str = "unknown"
    schema_version: str = "v1"
    policy_revision: str = "v1"
    execution_class: ExecutionClass = ExecutionClass.READ_ONLY

@dataclass
class Evidence:
    source: str
    value: Any
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Result:
    status: VerificationStatus
    output: Any = None
    evidence: list[Evidence] = field(default_factory=list)
    cache_key: str | None = None
    artifact_id: str | None = None
    verifier_disagreement: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
