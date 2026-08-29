"""20-day momentum on SPY, with a forward-looking column kept for reporting."""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    forward_max = df["close"].rolling(10).max().shift(-10)
    momentum = df["close"] / df["close"].shift(20) - 1
    signal = momentum > 0
    return signal.shift(1).fillna(False).astype(int)


def report(df: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc diagnostics only. Nothing here reaches the decision."""
    return pd.DataFrame({"forward_max": df["close"].rolling(10).max().shift(-10)})


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
