"""Two stacked leaks, the second invisible until the first one is fixed.

Measured on eval/data/SPY.csv. The full-sample `cutoff` is a real leak, but
while the centered `fast` average is present, removing the cutoff *raises*
Sharpe by 0.62 — so a system that judges each candidate once, against the
original baseline, files it as no-effect and moves on. Only after the centered
window is repaired and the baseline re-run does removing the cutoff deflate the
strategy.

The fixture exists to make one point: "no effect" is a statement about a
baseline, not about a line of code.
"""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    fast = close.rolling(9, center=True).mean()
    slow = close.rolling(30).mean()
    edge = fast / slow - 1
    cutoff = edge.quantile(0.60)
    signal = edge > cutoff
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
