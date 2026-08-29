"""Turning a strategy run into numbers, and into a verdict about those numbers.

Lives in core because the eval harness and the Phase-2 sandbox must agree on
what "Sharpe" means down to the last digit. One definition, one place.
"""

from __future__ import annotations

import math

import pandas as pd

from hindsight_core.models import SandboxOutcome

TRADING_DAYS = 252


def sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe at a zero risk-free rate."""
    r = returns.dropna()
    std = r.std(ddof=1)
    if len(r) < 2 or std == 0 or not math.isfinite(std):
        return float("nan")
    return float(r.mean() / std * math.sqrt(TRADING_DAYS))


def total_return(equity: pd.Series) -> float:
    e = equity.dropna()
    return float(e.iloc[-1] / e.iloc[0] - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    e = equity.dropna()
    return float((e / e.cummax() - 1.0).min())


def count_position_changes(positions: pd.Series) -> int:
    p = positions.dropna()
    return int((p != p.shift(1)).iloc[1:].sum())


def evaluate(
    equity: pd.Series, positions: pd.Series
) -> tuple[SandboxOutcome, dict[str, float], int]:
    """Classify a completed run and, only if it is testable, measure it.

    A strategy that never traded, or whose equity never moved, has no defined
    Sharpe. It is ZERO_TRADES — untestable, not a zero-scoring success — and
    carries no metrics at all so nothing downstream can read one by accident.
    """
    changes = count_position_changes(positions)
    returns = equity.pct_change()
    metrics = {
        "sharpe": sharpe(returns),
        "total_return": total_return(equity),
        "max_drawdown": max_drawdown(equity),
    }
    if changes == 0 or not math.isfinite(metrics["sharpe"]):
        return SandboxOutcome.ZERO_TRADES, {}, changes
    return SandboxOutcome.COMPLETED, metrics, changes
