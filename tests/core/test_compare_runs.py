"""compare_runs is where every metric lands, and where a run with no metrics
must not be turned into a delta that looks like proof."""

import pytest

from hindsight_core.models import RunRecord, SandboxOutcome
from hindsight_core.tools.compare_runs import compare_runs
from hindsight_core.tools.run_backtest import save_run


def record(run_id, outcome=SandboxOutcome.COMPLETED, sharpe=1.0, equity=(("d", 1.0),)):
    metrics = (
        {"sharpe": sharpe, "total_return": 0.5, "max_drawdown": -0.1}
        if outcome is SandboxOutcome.COMPLETED
        else {}
    )
    return RunRecord(
        run_id=run_id,
        outcome=outcome,
        metrics=metrics,
        position_changes=10 if outcome is SandboxOutcome.COMPLETED else 0,
        equity=equity,
    )


@pytest.fixture
def store(tmp_path):
    save_run(record("before", sharpe=4.12, equity=(("2022-01-03", 1.0),)), tmp_path)
    save_run(record("after", sharpe=0.34, equity=(("2022-01-03", 1.0),)), tmp_path)
    save_run(record("empty", outcome=SandboxOutcome.ZERO_TRADES, equity=()), tmp_path)
    save_run(record("boom", outcome=SandboxOutcome.CRASHED, equity=()), tmp_path)
    return tmp_path


def test_delta_is_after_minus_before(store):
    comparison = compare_runs("before", "after", store=store)

    assert comparison.delta["sharpe"] == pytest.approx(0.34 - 4.12)
    assert comparison.before.run_id == "before"
    assert comparison.after.run_id == "after"


def test_both_equity_curves_come_back_as_plain_series(store):
    comparison = compare_runs("before", "after", store=store)

    assert comparison.before_equity == [("2022-01-03", 1.0)]
    assert comparison.after_equity == [("2022-01-03", 1.0)]


def test_a_zero_trade_run_produces_no_delta(store):
    comparison = compare_runs("before", "empty", store=store)

    assert comparison.delta == {}


def test_a_crashed_run_produces_no_delta(store):
    comparison = compare_runs("boom", "after", store=store)

    assert comparison.delta == {}


def test_an_unknown_run_id_is_a_readable_error(store):
    with pytest.raises(KeyError, match="ghost"):
        compare_runs("before", "ghost", store=store)
