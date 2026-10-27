from realtime_orchestrator.schemas.events import Event, Trace
from realtime_orchestrator.verifier.policy import evaluate_trace, policy_check


def test_trace_sequence_and_integrity() -> None:
    trace = Trace("run_1")
    trace.append(Event("run_1", 0, "session", "session.created", {}))
    trace.append(Event("run_1", 1, "response", "response.done", {}))
    assert trace.verify_integrity()


def test_verifier_fails_closed_for_invalid_trace() -> None:
    result = evaluate_trace({}, {"outcome": "success"})
    assert not result.passed
    assert "invalid_trace" in result.labels


def test_policy_blocks_unapproved_side_effect() -> None:
    result = policy_check("external_side_effect", approved=False)
    assert not result.allowed


def test_policy_allows_approved_side_effect() -> None:
    result = policy_check("external_side_effect", approved=True)
    assert result.allowed
