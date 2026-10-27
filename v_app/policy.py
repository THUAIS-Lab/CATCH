from .models import ExecutionClass, Task

class PolicyDenied(PermissionError):
    pass

class Policy:
    """User-controlled execution boundary. Policy is checked on every execution."""
    def __init__(self, approvals: set[ExecutionClass] | None = None) -> None:
        self.approvals = approvals or set()

    def check(self, task: Task) -> None:
        high_risk = {ExecutionClass.EXTERNAL_SIDE_EFFECT, ExecutionClass.SECRET_ACCESS, ExecutionClass.PRODUCTION_DEPLOY}
        if task.execution_class in high_risk and task.execution_class not in self.approvals:
            raise PolicyDenied(f"Approval required for {task.execution_class.value}")
