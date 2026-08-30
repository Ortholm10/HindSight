"""The prover's own retry loop, and the outcomes it must never collapse.

Every test here runs real code in the real sandbox against the committed SPY
cache. A prover that silently returns "nothing found" looks exactly like a
prover that works, so these assert on measured numbers rather than on shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.cache_data import DATA_DIR
from hindsight_core.models import Budget, RunRecord, SandboxOutcome
from hindsight_core.provers import differential
from hindsight_core.provers.differential import classify, prove_leak
from hindsight_core.tools.run_backtest import run_backtest
from hindsight_core.tools.scan_file import scan_file

DATA = DATA_DIR / "SPY.csv"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CASES = Path(__file__).resolve().parents[2] / "eval" / "cases"


@pytest.fixture
def mechanical(monkeypatch):
    """No network. The model rung returns nothing, so what is under test is the
    mechanical sweep — which is also the floor the audit must not fall through
    when a free-tier provider is unavailable mid-run."""
    monkeypatch.setattr(differential, "_next_operation", lambda *a, **k: "")


def _completed(run_id: str, sharpe: float) -> RunRecord:
    return RunRecord(
        run_id=run_id, outcome=SandboxOutcome.COMPLETED, metrics={"sharpe": sharpe}
    )


def _candidate(path: Path, leak_type: str = "", line: int = 0):
    return next(
        c
        for c in scan_file(path)
        if (not leak_type or c.leak_type == leak_type) and (not line or c.line == line)
    )


# --- the three failure modes are never collapsed ----------------------------


def test_classify_calls_a_crash_a_broken_patch_not_a_clean_run():
    after = RunRecord(run_id="a", outcome=SandboxOutcome.CRASHED, stderr="NameError")
    assert classify(_completed("b", 2.0), after, {}) == "patch_broken"


def test_classify_calls_a_timeout_a_broken_patch():
    after = RunRecord(run_id="a", outcome=SandboxOutcome.TIMED_OUT)
    assert classify(_completed("b", 2.0), after, {}) == "patch_broken"


def test_classify_calls_zero_trades_untestable_never_clean():
    after = RunRecord(run_id="a", outcome=SandboxOutcome.ZERO_TRADES)
    assert classify(_completed("b", 2.0), after, {}) == "untestable"


def test_classify_proves_only_a_negative_sharpe_delta():
    before, after = _completed("b", 2.0), _completed("a", 0.5)
    assert classify(before, after, {"sharpe": -1.5}) == "proven"
    assert classify(before, after, {"sharpe": 0.4}) == "no_effect"


# --- multi-step repair -------------------------------------------------------


def test_a_second_operation_is_tried_when_the_first_ran_but_proved_nothing(mechanical):
    """l05 is the guaranteed instance. expanding_stat is listed first in
    .claude/skills/l05.md and is taxonomically valid, so triage picks it — and
    it runs clean while moving Sharpe the wrong way. rolling_stat is one retry
    away. Nothing here is staged: the skill file is untouched."""
    path = CASES / "l05_full_sample_zscore" / "strategy.py"
    baseline = run_backtest(path, DATA)
    runs = {baseline.run_id: baseline}

    result = prove_leak(
        path,
        _candidate(path, "L05"),
        DATA,
        baseline,
        operation="expanding_stat",
        runs=runs,
    )

    operations = [a.operation for a in result.attempts]
    assert operations[0] == "expanding_stat"
    assert result.attempts[0].status == "no_effect"
    assert result.attempts[0].delta["sharpe"] > 0  # it moved the wrong way
    assert "rolling_stat" in operations
    assert result.status == "proven"
    assert result.finding is not None
    assert result.finding.delta["sharpe"] < 0


def test_a_crashing_patch_is_read_and_a_different_repair_is_tried(mechanical):
    """Seeded with `lag`, which is a defensible answer for L05 and which the
    sandbox rejects: quantile() returns a scalar, and a scalar has no .shift.
    The prover has to read that traceback and keep going — twice, as it turns
    out, because the operation after it also fails to deflate."""
    path = CASES / "l05_full_sample_zscore" / "strategy.py"
    baseline = run_backtest(path, DATA)

    result = prove_leak(
        path,
        _candidate(path, "L05"),
        DATA,
        baseline,
        operation="lag",
        runs={baseline.run_id: baseline},
    )

    first = result.attempts[0]
    assert first.operation == "lag"
    assert first.applied is True
    assert first.status == "patch_broken"
    assert "AttributeError" in first.stderr
    assert len(result.attempts) > 1  # it did not give up after one attempt
    assert result.status == "proven"


def test_a_crash_with_no_other_repair_available_is_reported_as_broken(mechanical):
    """The honest terminal case: fit_in_fold is the only operation that
    transforms this line, and the patched code cannot run. Not clean, not
    proven — broken, and said so."""
    path = FIXTURES / "crash_on_patch.py"
    baseline = run_backtest(path, DATA)

    result = prove_leak(
        path,
        _candidate(path, "L11"),
        DATA,
        baseline,
        operation="fit_in_fold",
        runs={baseline.run_id: baseline},
    )

    assert result.status == "patch_broken"
    assert result.finding is None
    assert "NameError" in result.attempts[0].stderr


# --- the vocabulary boundary -------------------------------------------------


def test_the_vocabulary_boundary_is_reported_not_forced(mechanical):
    """L08 line 9: lag runs and changes nothing, drop_column has no operand to
    remove. A correct repair needs a substitute column, which is a judgement.
    The answer is "detected, not mechanically patchable" — never an invented
    value to force a patch through."""
    path = CASES / "l08_forward_target" / "strategy.py"
    baseline = run_backtest(path, DATA)

    result = prove_leak(
        path,
        _candidate(path, line=9),
        DATA,
        baseline,
        operation="lag",
        runs={baseline.run_id: baseline},
    )

    assert result.status == "not_mechanically_patchable"
    assert result.finding is None


# --- budget and bookkeeping --------------------------------------------------


def test_the_budget_stops_the_retry_loop(mechanical):
    path = CASES / "l05_full_sample_zscore" / "strategy.py"
    baseline = run_backtest(path, DATA)
    budget = Budget(max_llm_calls=0, max_sandbox_runs=1)

    result = prove_leak(
        path,
        _candidate(path, "L05"),
        DATA,
        baseline,
        operation="expanding_stat",
        budget=budget,
        runs={baseline.run_id: baseline},
    )

    assert len(result.attempts) == 1
    assert budget.exhausted is True
    assert result.status != "proven"


def test_every_run_the_prover_makes_is_recorded_for_the_hook(mechanical):
    path = CASES / "l02_centered_window" / "strategy.py"
    baseline = run_backtest(path, DATA)
    runs = {baseline.run_id: baseline}

    result = prove_leak(
        path,
        _candidate(path, "L02"),
        DATA,
        baseline,
        operation="trailing_window",
        runs=runs,
    )

    assert result.status == "proven"
    assert result.finding.before_run_id in runs
    assert result.finding.after_run_id in runs


def test_an_operation_outside_the_vocabulary_never_reaches_the_sandbox(mechanical):
    """A repair the model invented is refused by name. The sweep still covers
    the line, so this rejects the operation, not the candidate."""
    path = CASES / "l02_centered_window" / "strategy.py"
    baseline = run_backtest(path, DATA)

    result = prove_leak(
        path,
        _candidate(path, "L02"),
        DATA,
        baseline,
        operation="rewrite_the_signal",
        runs={baseline.run_id: baseline},
    )

    assert "rewrite_the_signal" not in [a.operation for a in result.attempts]
    assert result.status == "proven"  # the sweep found trailing_window anyway


# --- the sweep is not allowed to manufacture a finding on its own ------------


def test_a_repair_nobody_proposed_must_be_confirmed_before_it_counts(monkeypatch):
    """A Sharpe that fell is not the same fact as a leak that was removed.

    The seed operation does not apply here, so the repair that eventually
    proves comes from the mechanical sweep — chosen by trying operations until
    one moved the number, which establishes only that the number moved. Until
    something answers for it, it is not a finding.

    The instance this gate was built for — the sweep flipping a supervised
    training label on l10 — is now handled deterministically in scan_file,
    which no longer offers the label as a candidate at all. The gate remains
    because the sweep can still reach an operation nobody reasoned about.
    """
    monkeypatch.setattr(differential, "_next_operation", lambda *a, **k: "")
    monkeypatch.setattr(
        differential,
        "_confirm_repair",
        lambda *a, **k: (False, "that corrupts a value the strategy is entitled to"),
    )

    path = CASES / "l02_centered_window" / "strategy.py"
    baseline = run_backtest(path, DATA)

    result = prove_leak(
        path,
        _candidate(path, "L02"),
        DATA,
        baseline,
        operation="resample_label",  # does not apply; the sweep takes over
        runs={baseline.run_id: baseline},
    )

    assert result.finding is None
    assert result.status == "repair_rejected"
    rejected = [a for a in result.attempts if a.status == "repair_rejected"]
    assert rejected and rejected[0].operation == "trailing_window"
    # The number really did fall. That it fell is exactly what is not enough.
    assert rejected[0].delta["sharpe"] < 0


def test_a_repair_the_triage_named_is_not_second_guessed(monkeypatch):
    """Confirmation applies to the sweep, not to the model's own answer. The
    seed operation was chosen with the code in front of it; re-asking would
    spend a call to relitigate a judgement already made."""
    asked = []
    monkeypatch.setattr(differential, "_next_operation", lambda *a, **k: "")
    monkeypatch.setattr(
        differential,
        "_confirm_repair",
        lambda *a, **k: (asked.append(1), (True, "fine"))[1],
    )

    path = CASES / "l02_centered_window" / "strategy.py"
    baseline = run_backtest(path, DATA)

    result = prove_leak(
        path,
        _candidate(path, "L02"),
        DATA,
        baseline,
        operation="trailing_window",
        runs={baseline.run_id: baseline},
    )

    assert result.status == "proven"
    assert asked == [], "the seed operation needs no confirmation"


def test_a_confirmed_sweep_repair_still_counts(monkeypatch):
    """l05's rolling_stat comes from the sweep and is a real repair. The gate
    must not cost the retry behaviour it was added alongside."""
    monkeypatch.setattr(differential, "_next_operation", lambda *a, **k: "")
    monkeypatch.setattr(
        differential, "_confirm_repair", lambda *a, **k: (True, "causal")
    )

    path = CASES / "l05_full_sample_zscore" / "strategy.py"
    baseline = run_backtest(path, DATA)

    result = prove_leak(
        path,
        _candidate(path, "L05"),
        DATA,
        baseline,
        operation="expanding_stat",
        runs={baseline.run_id: baseline},
    )

    assert result.status == "proven"
    assert result.finding is not None
    assert result.attempts[-1].operation == "rolling_stat"
