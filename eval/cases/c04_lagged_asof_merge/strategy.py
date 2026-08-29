"""Daily SPY trend gated by a weekly average, merged as-of with a prior lag.

Traps L04. direction="backward" alone would NOT be safe -- it lands on the
week still in progress. The shift(1) before the merge is what makes it clean.
"""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    weekly = closes.resample("W", label="left", closed="right").mean()
    weekly_sma = weekly.rolling(3).mean()
    htf_values = weekly_sma.shift(1)
    htf = pd.DataFrame({"date": weekly_sma.index, "htf": htf_values.to_numpy()})
    base = pd.DataFrame({"date": closes.index, "close": closes.to_numpy()})
    merged = pd.merge_asof(base, htf, on="date", direction="backward")
    filt = pd.Series(merged["htf"].to_numpy(), index=closes.index)
    signal = (closes > closes.rolling(30).mean()) & (closes > filt)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
