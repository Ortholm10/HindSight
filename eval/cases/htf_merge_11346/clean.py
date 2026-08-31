"""Freqtrade issue #11346: a weekly trend merged onto daily rows by the weekly
bar's OPEN date, so Monday already knows how the week closes.

Not our design. Reported at github.com/freqtrade/freqtrade/issues/11346 against
a 15m base with a 1h informative; transcribed here to daily/weekly, which is the
data this eval caches. The reporter's own diagnosis of their line
`pair_dataframe[pair_dataframe['date'] <= row_time]` is the merge below.
See PROVENANCE.md.
"""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    weekly = closes.resample("W-MON", label="left", closed="left").last()
    weekly_sma = weekly.rolling(4).mean()
    # Lagged one full weekly period before the merge, which is exactly what
    # freqtrade's own merge_informative_pair does by stamping the informative
    # row at open + one interval. Row t now sees only a closed weekly bar.
    w_up = (weekly > weekly_sma).astype(float).shift(1)
    htf = pd.DataFrame({"date": w_up.index, "w_up": w_up.to_numpy()})
    base = pd.DataFrame({"date": closes.index})
    merged = pd.merge_asof(base, htf, on="date", direction="backward")
    signal = pd.Series(merged["w_up"].to_numpy(), index=closes.index).fillna(0.0)
    return (signal > 0).shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
