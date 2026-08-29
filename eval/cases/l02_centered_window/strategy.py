"""Trend following on SPY: hold while the 60-day average is rising."""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(60, center=True).mean()
    # The average's DIRECTION rather than the price's distance from it.
    # Comparing price to a centered average crosses about twice as often as
    # comparing it to a trailing one, because a centered average has no phase
    # lag; a slope moves slowly in both versions and keeps the trade counts
    # comparable (case validity rule 2).
    signal = sma > sma.shift(5)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
