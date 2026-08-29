"""Equal-weight 50-day trend across a fixed multi-asset universe."""

import pandas as pd

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE"]


def _survived(prices: pd.Series) -> bool:
    """Whether the instrument gained 30% over the WHOLE period. End-of-period info."""
    return bool(prices.iloc[-1] / prices.iloc[0] - 1 > 0.30)


def _closes(df: pd.DataFrame) -> pd.DataFrame:
    return df.xs("close", axis=1, level=1)


def _weights(df: pd.DataFrame) -> pd.DataFrame:
    closes = _closes(df)
    universe = SYMBOLS
    signals = {}
    for symbol in universe:
        price = closes[symbol]
        held = price > price.rolling(50).mean()
        signals[symbol] = held.shift(1).fillna(False)
    return pd.DataFrame(signals).astype(float)


def run_positions(df: pd.DataFrame) -> pd.Series:
    return _weights(df).sum(axis=1).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    weights = _weights(df)
    returns = _closes(df)[weights.columns].pct_change()
    held = weights.sum(axis=1)
    portfolio = (weights * returns).sum(axis=1) / held.where(held > 0)
    return (1 + portfolio.fillna(0)).cumprod()
