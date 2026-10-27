from v_app.cache import VerifiedCache
from v_app.models import ExecutionClass, Result, Task, VerificationStatus

def task(**kwargs):
    base = dict(task_id="t1", input={"x": 1}, model="local", code_revision="abc")
    base.update(kwargs)
    return Task(**base)

def test_only_verified_results_are_reusable():
    cache = VerifiedCache()
    t = task()
    try:
        cache.put_verified(t, Result(status=VerificationStatus.UNKNOWN, output="x"))
        assert False
    except ValueError:
        pass
    assert cache.get(t) is None

def test_key_changes_with_provenance():
    cache = VerifiedCache()
    t1, t2 = task(code_revision="a"), task(code_revision="b")
    cache.put_verified(t1, Result(status=VerificationStatus.VERIFIED, output=1))
    assert cache.get(t1) is not None
    assert cache.get(t2) is None

def test_side_effects_never_replay():
    cache = VerifiedCache()
    t = task(execution_class=ExecutionClass.EXTERNAL_SIDE_EFFECT)
    cache.put_verified(t, Result(status=VerificationStatus.VERIFIED, output="done"))
    assert cache.get(t) is None
