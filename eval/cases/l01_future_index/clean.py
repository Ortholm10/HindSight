"""Slow trend following on SPY against a 50-day average."""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(50).mean()
    signal = df["close"] > sma
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
