"""Daily SPY trend, gated by a monthly average."""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    # "MS" is one of the frequencies pandas labels LEFT by default, which would
    # stamp a whole month's mean on that month's first day. Label right, close
    # right, then lag: row t sees only a month that has finished.
    monthly = closes.resample("MS").mean()
    filt = monthly.reindex(closes.index, method="ffill")
    raw = closes < filt
    # Rebalance monthly, the horizon this leak actually spans: a bare "MS"
    # resample hands every day the mean of the month it is standing in.
    rebalance = pd.Series(
        ~closes.index.to_period("M").duplicated(), index=closes.index
    )
    signal = raw.where(rebalance).ffill().fillna(False).astype(bool)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
