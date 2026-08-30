"""The hook is the product rule in code: no finding without two real runs."""

from __future__ import annotations

import pytest

from hindsight_core.hooks.verification import VerificationError, verify_findings
from hindsight_core.models import Finding, LeakCandidate, RunRecord, SandboxOutcome

CANDIDATE = LeakCandidate(
    leak_type="L03", file="s.py", line=9, snippet="signal = a > b", reason="no lag"
)


def _run(run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id, outcome=SandboxOutcome.COMPLETED, metrics={"sharpe": 1.0}
    )


def test_a_finding_backed_by_two_stored_runs_passes():
    runs = {"before": _run("before"), "after": _run("after")}
    finding = Finding(
        candidate=CANDIDATE, before_run_id="before", after_run_id="after", diff="-x\n+y"
    )
    assert verify_findings([finding], runs) == [finding]


def test_a_fabricated_finding_is_blocked():
    """Both run IDs are well-formed and neither exists. This is the attack the
    hook exists for: a Finding whose __post_init__ passes but whose evidence
    does not."""
    finding = Finding(
        candidate=CANDIDATE,
        before_run_id="deadbeefdeadbeef",
        after_run_id="cafebabecafebabe",
        diff="-x\n+y",
    )
    with pytest.raises(VerificationError) as error:
        verify_findings([finding], {})
    assert "deadbeefdeadbeef" in str(error.value)


def test_half_fabricated_is_still_blocked():
    runs = {"before": _run("before")}
    finding = Finding(
        candidate=CANDIDATE, before_run_id="before", after_run_id="notreal", diff="d"
    )
    with pytest.raises(VerificationError) as error:
        verify_findings([finding], runs)
    assert "notreal" in str(error.value)


def test_a_run_that_did_not_complete_cannot_back_a_finding():
    """A crashed run is an execution record, but not evidence of a delta."""
    runs = {
        "before": _run("before"),
        "after": RunRecord(run_id="after", outcome=SandboxOutcome.CRASHED),
    }
    finding = Finding(
        candidate=CANDIDATE, before_run_id="before", after_run_id="after", diff="d"
    )
    with pytest.raises(VerificationError):
        verify_findings([finding], runs)


def test_empty_run_ids_are_refused_by_the_dataclass_itself():
    with pytest.raises(ValueError):
        Finding(candidate=CANDIDATE, before_run_id="", after_run_id="after", diff="d")


def test_nothing_to_verify_is_not_an_error():
    assert verify_findings([], {}) == []
