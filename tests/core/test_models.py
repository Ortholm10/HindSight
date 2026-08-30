import pytest

from hindsight_core.models import (
    Budget,
    EventType,
    Finding,
    LeakCandidate,
    ProofResult,
    RunRecord,
    SandboxOutcome,
)

CANDIDATE = LeakCandidate(
    leak_type="L03",
    file="strategy.py",
    line=42,
    snippet="position = signal.astype(int)",
    reason="no shift between signal derivation and use",
)


def _finding(**kwargs):
    return Finding(candidate=CANDIDATE, diff="- a\n+ b", **kwargs)


def test_finding_requires_both_run_ids():
    ok = _finding(before_run_id="r1", after_run_id="r2")
    assert ok.before_run_id == "r1"


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("", "r2", "before_run_id"),
        ("r1", "", "after_run_id"),
        ("", "", "before_run_id, after_run_id"),
    ],
)
def test_finding_without_execution_record_raises(before, after, expected):
    with pytest.raises(ValueError, match=expected):
        _finding(before_run_id=before, after_run_id=after)


def test_sandbox_outcomes_are_exactly_four():
    assert [o.value for o in SandboxOutcome] == [
        "completed",
        "crashed",
        "timed_out",
        "zero_trades",
    ]


def test_zero_trades_is_not_a_completed_run():
    run = RunRecord(run_id="r3", outcome=SandboxOutcome.ZERO_TRADES)
    assert run.outcome is not SandboxOutcome.COMPLETED
    assert run.metrics == {}


def test_event_types_match_the_stream_schema():
    assert [e.value for e in EventType] == [
        "scan_complete",
        "triage",
        "baseline",
        "prove_start",
        "prove_result",
        "agent_decision",
        "final",
    ]


def test_budget_refuses_to_overspend_and_names_which_cap_it_hit():
    budget = Budget(max_llm_calls=2, max_sandbox_runs=1)
    assert budget.spend_llm() is True
    assert budget.spend_llm() is True
    assert budget.spend_llm() is False
    assert budget.exhausted is True
    assert "llm" in budget.reason


def test_budget_counts_the_two_resources_separately():
    budget = Budget(max_llm_calls=1, max_sandbox_runs=2)
    assert budget.spend_llm() is True
    assert budget.spend_run() is True
    assert budget.spend_run() is True
    assert budget.spend_run() is False
    assert "sandbox" in budget.reason


def test_proof_result_defaults_to_no_finding():
    result = ProofResult(candidate=CANDIDATE, status="no_effect", attempts=())
    assert result.finding is None
    assert result.attempts == ()
