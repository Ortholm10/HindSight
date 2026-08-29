"""20-day momentum on SPY. A forward return is computed, for reporting only.

Traps L01 and L08. The .shift(-5) column never reaches the decision, so a
scanner keyed on negative shifts flags this and a prover does not: shifting the
reporting column leaves the equity curve untouched.
"""

import pandas as pd


def forward_return(df: pd.DataFrame) -> pd.Series:
    """What the next five sessions did. Reporting only, never a feature."""
    return df["close"].shift(-5) / df["close"] - 1


def run_positions(df: pd.DataFrame) -> pd.Series:
    momentum = df["close"] / df["close"].shift(20) - 1
    signal = momentum > 0
    return signal.shift(1).fillna(False).astype(int)


def report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"forward_return": forward_return(df)})


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
