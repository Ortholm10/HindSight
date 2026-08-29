"""Mean reversion on SPY scored against an expanding-window z-score.

Traps L05. expanding() is past-only by construction and must never be flagged.
"""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    spread = closes / closes.rolling(20).mean() - 1
    z = (spread - spread.expanding(60).mean()) / spread.expanding(60).std()
    signal = z < -1.0
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
