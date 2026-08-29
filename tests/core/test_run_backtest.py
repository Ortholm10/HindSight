"""run_backtest is the sandbox plus persistence: every finding cites a run_id,
so a run_id that cannot be read back later is not evidence."""

import math
from pathlib import Path

import pytest

from hindsight_core.models import SandboxOutcome
from hindsight_core.tools.run_backtest import load_run, run_backtest

REPO = Path(__file__).resolve().parents[2]
CASE = REPO / "eval" / "cases" / "l01_future_index" / "strategy.py"
DATA = REPO / "eval" / "data" / "SPY.csv"


@pytest.mark.timeout(120)
def test_real_eval_case_produces_a_sharpe_and_a_run_id(tmp_path):
    record = run_backtest(CASE, DATA, store=tmp_path)

    assert record.outcome is SandboxOutcome.COMPLETED
    assert math.isfinite(record.metrics["sharpe"])
    assert len(record.run_id) == 16
    assert len(record.equity) > 500


@pytest.mark.timeout(120)
def test_the_run_record_is_readable_back_by_its_run_id(tmp_path):
    record = run_backtest(CASE, DATA, store=tmp_path)

    restored = load_run(record.run_id, store=tmp_path)

    assert restored == record


def test_loading_an_unknown_run_id_is_a_readable_error(tmp_path):
    with pytest.raises(KeyError, match="nosuchrun"):
        load_run("nosuchrun", store=tmp_path)
