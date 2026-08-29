"""100-day trend on SPY. The full-sample min/max sizes a chart axis.

Traps L05. df["close"].max() is a full-sample aggregate, but it is never a
feature -- it only bounds a plot.
"""

import pandas as pd


def chart_limits(df: pd.DataFrame) -> tuple[float, float]:
    """Axis bounds for the equity plot. Not a feature."""
    return float(df["close"].min()), float(df["close"].max())


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    signal = closes > closes.rolling(100).mean()
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
