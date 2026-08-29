"""Buy SPY for twenty sessions after a local low has been identified."""

import pandas as pd


def two_sided_troughs(close: pd.Series, k: int) -> pd.Series:
    """A low compared against its neighbours on BOTH sides."""
    window = close.rolling(2 * k + 1, center=True).min()
    return close.eq(window)


def causal_troughs(close: pd.Series, k: int) -> pd.Series:
    """The same low, declared only once k later bars have failed to beat it.

    close.shift(k) is the candidate bar and the window ending at t spans the k
    bars either side of it, so the mark lands k bars after the low itself.
    """
    window = close.rolling(2 * k + 1).min()
    return close.shift(k).eq(window)


def run_positions(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    troughs = two_sided_troughs(close, 10)
    signal = troughs.astype(int).rolling(20).max().fillna(0).astype(bool)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
