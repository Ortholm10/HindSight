"""SPY momentum measured against a sparsely sampled month-end reference."""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    monthly = closes.resample("ME").last()
    macro = monthly.reindex(closes.index)
    macro = macro.ffill()
    signal = closes < macro
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
