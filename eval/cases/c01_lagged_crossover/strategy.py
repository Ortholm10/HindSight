"""20/50 crossover on SPY with the execution lag correctly applied.

Traps L03. One token from the leaked version: the .shift(1) is present.
"""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()
    signal = fast > slow
    # Decided on yesterday's close, held through today. This is the lag.
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
