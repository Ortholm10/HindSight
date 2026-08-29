import math

import pandas as pd
import pytest

from hindsight_core.metrics import (
    count_position_changes,
    evaluate,
    max_drawdown,
    sharpe,
    total_return,
)
from hindsight_core.models import SandboxOutcome


def test_sharpe_of_zero_mean_returns_is_zero():
    assert sharpe(pd.Series([0.01, -0.01] * 10)) == 0.0


def test_sharpe_is_annualised_by_root_252():
    returns = pd.Series([0.02, 0.01, 0.01, 0.02])
    expected = (returns.mean() / returns.std(ddof=1)) * math.sqrt(252)
    assert sharpe(returns) == pytest.approx(expected)


def test_sharpe_ignores_leading_nan():
    assert sharpe(pd.Series([float("nan"), 0.01, -0.01])) == pytest.approx(
        sharpe(pd.Series([0.01, -0.01]))
    )


def test_max_drawdown_is_the_worst_peak_to_trough():
    assert max_drawdown(pd.Series([1.0, 2.0, 1.0, 4.0])) == pytest.approx(-0.5)


def test_total_return_is_last_over_first():
    assert total_return(pd.Series([1.0, 1.2, 1.5])) == pytest.approx(0.5)


def test_count_position_changes_counts_transitions():
    assert count_position_changes(pd.Series([0, 0, 1, 1, 0])) == 2


def test_count_position_changes_ignores_the_opening_nan():
    assert count_position_changes(pd.Series([float("nan"), 0, 1])) == 1


def test_evaluate_reports_completed_with_metrics():
    equity = pd.Series([1.0, 1.01, 1.03, 1.02, 1.05])
    positions = pd.Series([0, 1, 1, 0, 1])
    outcome, metrics, changes = evaluate(equity, positions)
    assert outcome is SandboxOutcome.COMPLETED
    assert set(metrics) == {"sharpe", "total_return", "max_drawdown"}
    assert changes == 3


def test_evaluate_reports_zero_trades_when_never_in_the_market():
    equity = pd.Series([1.0, 1.0, 1.0, 1.0])
    positions = pd.Series([0, 0, 0, 0])
    outcome, metrics, changes = evaluate(equity, positions)
    assert outcome is SandboxOutcome.ZERO_TRADES
    assert metrics == {}
    assert changes == 0


def test_evaluate_reports_zero_trades_on_zero_variance_returns():
    # A flat equity curve has an undefined Sharpe. Never report it as 0.0.
    equity = pd.Series([1.0, 1.0, 1.0, 1.0])
    positions = pd.Series([1, 1, 1, 1])
    outcome, metrics, _ = evaluate(equity, positions)
    assert outcome is SandboxOutcome.ZERO_TRADES
    assert metrics == {}
