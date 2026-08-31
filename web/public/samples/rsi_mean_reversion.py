"""SPY trend against a sparse quarterly reference carried forward.

Traps L06. Forward fill is causal: each row repeats the last OBSERVED value.
"""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    quarterly = closes.resample("QE").last()
    macro = quarterly.reindex(closes.index).ffill()
    signal = (closes > macro) & (closes > closes.rolling(20).mean())
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
