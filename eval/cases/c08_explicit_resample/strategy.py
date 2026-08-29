"""Daily SPY trend gated by a monthly average, resampled explicitly.

Traps L07. label="right", closed="right" AND a shift: the label alone would
stamp the bin at the instant it closes, which is still simultaneous with the
decision.
"""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    monthly = closes.resample("MS", label="right", closed="right").mean().shift(1)
    filt = monthly.reindex(closes.index, method="ffill")
    signal = (closes > closes.rolling(10).mean()) & (closes > filt)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
