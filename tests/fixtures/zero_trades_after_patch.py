"""A strategy whose only repair leaves it with nothing to trade.

The backward fill is what makes the threshold reachable at all: the level is
observed on the last bar and carried backwards over the whole series, so every
earlier day is compared against a number from the end of the sample. Forward
fill instead leaves the series NaN until that final bar and never crosses, so
the patched strategy holds no position anywhere.

That is not a clean result. It is an untestable one, and keeping the two apart
is the whole of CLAUDE.md critical rule 3.
"""

import numpy as np
import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    sparse = pd.Series(np.nan, index=close.index)
    sparse.iloc[-1] = close.mean()
    level = sparse.bfill()
    signal = close > level
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
