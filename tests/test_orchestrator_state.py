from realtime_orchestrator.orchestrator.recovery import classify_failure
from realtime_orchestrator.orchestrator.state import Run, RunState, transition
from realtime_orchestrator.audit import compare_verifiers


def test_run_state_machine() -> None:
    run = Run("r1", "s1", "a1")
    transition(run, RunState.PREPARING)
    transition(run, RunState.RUNNING)
    transition(run, RunState.VERIFYING)
    transition(run, RunState.AUDITING)
    transition(run, RunState.SUCCEEDED)
    assert run.state == RunState.SUCCEEDED


def test_invalid_transition_fails_closed() -> None:
    run = Run("r1", "s1", "a1")
    try:
        transition(run, RunState.SUCCEEDED)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid transition was accepted")


def test_retry_is_bounded() -> None:
    assert classify_failure("network", 0, 3).retry
    assert not classify_failure("network", 3, 3).retry
    assert not classify_failure("policy_denied", 0, 3).retry


def test_visible_independent_disagreement_is_a_signal() -> None:
    result = compare_verifiers(True, False, hack_methods=("context_hack",))
    assert result.verifier_disagreement
    assert result.nontrivial_hack
    assert not result.genuine_success
