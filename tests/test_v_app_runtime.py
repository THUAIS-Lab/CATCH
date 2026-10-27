from v_app.models import Result, Task, VerificationStatus
from v_app.runtime import AgentRuntime
from v_app.verifier import ExactVerifier

def test_conservative_verifier_never_publishes_false_cache():
    rt=AgentRuntime(); calls=[]; task=Task(task_id="1",input="x",model="demo",code_revision="a")
    def worker(_): calls.append(1); return 42
    first=rt.execute(task,worker,ExactVerifier(42)); second=rt.execute(task,worker,ExactVerifier(42))
    assert first.status is VerificationStatus.UNKNOWN
    assert second.status is VerificationStatus.UNKNOWN
    assert len(calls)==2
