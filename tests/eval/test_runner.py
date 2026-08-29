import pytest

from eval.runner import CaseMeta, load_case, run_case
from hindsight_core.models import SandboxOutcome


def test_load_case_parses_metadata(make_case):
    meta = load_case(make_case())
    assert isinstance(meta, CaseMeta)
    assert meta.case_id == "fixture_case"
    assert meta.symbols == ["SPY"]
    assert meta.start == "2022-01-03"


def test_run_case_produces_a_completed_record(make_case):
    record = run_case(load_case(make_case()))
    assert record.outcome is SandboxOutcome.COMPLETED
    assert record.position_changes > 10
    assert record.metrics["sharpe"] == pytest.approx(record.metrics["sharpe"])


def test_run_case_reports_zero_trades_rather_than_a_zero_sharpe(
    make_case, never_trades
):
    record = run_case(load_case(make_case(source=never_trades)))
    assert record.outcome is SandboxOutcome.ZERO_TRADES
    assert record.metrics == {}


def test_window_restricts_the_data_and_changes_the_result(make_case):
    meta = load_case(make_case())
    full = run_case(meta)
    first_year = run_case(meta, window=("2022-01-03", "2022-12-30"))
    assert first_year.metrics["sharpe"] != full.metrics["sharpe"]


def test_run_id_is_content_addressed_so_identical_inputs_reuse_it(make_case):
    meta = load_case(make_case())
    assert run_case(meta).run_id == run_case(meta).run_id


def test_run_id_differs_between_windows(make_case):
    meta = load_case(make_case())
    a = run_case(meta, window=("2022-01-03", "2022-12-30"))
    b = run_case(meta, window=("2023-01-03", "2023-12-29"))
    assert a.run_id != b.run_id


def test_multi_symbol_case_receives_one_frame_with_symbol_columns(make_case):
    source = """
import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    close = df.xs("close", axis=1, level=1)
    return (close["SPY"] > close["QQQ"] * 0.0).shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    close = df.xs("close", axis=1, level=1)
    returns = run_positions(df) * close["SPY"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
"""
    meta = load_case(make_case(source=source, symbols=["SPY", "QQQ"]))
    assert run_case(meta).outcome is SandboxOutcome.COMPLETED
