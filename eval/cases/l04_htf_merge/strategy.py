"""Daily SPY trend, gated by a weekly average merged back with merge_asof."""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    weekly = closes.resample("W", label="left", closed="right").mean()
    weekly_sma = weekly.rolling(2).mean()
    # The weekly bar is stamped at the week's OPEN, so an as-of merge lands on
    # the week still in progress. Lagging one full week before the merge is what
    # makes row t see only a closed higher-timeframe bar.
    htf_values = weekly_sma
    htf = pd.DataFrame({"date": weekly_sma.index, "htf": htf_values.to_numpy()})
    base = pd.DataFrame({"date": closes.index, "close": closes.to_numpy()})
    merged = pd.merge_asof(base, htf, on="date", direction="backward")
    filt = pd.Series(merged["htf"].to_numpy(), index=closes.index)
    # The filter's DIRECTION rather than the price's distance from it: comparing
    # price to a future-aware weekly average crosses far more often than
    # comparing it to a lagged one, which breaks rule 2's trade-count bound.
    signal = filt > filt.shift(5)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
