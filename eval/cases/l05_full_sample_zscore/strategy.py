"""Mean reversion on SPY: buy the weakest days against the 20-day trend."""

import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    closes = df["close"]
    spread = closes / closes.rolling(20).mean() - 1
    cutoff = spread.quantile(0.15)
    raw = spread < cutoff
    # Rebalance weekly so both versions decide on identical days (rule 2).
    rebalance = pd.Series(range(len(closes)), index=closes.index) % 5 == 0
    signal = raw.where(rebalance).ffill().fillna(False).astype(bool)
    return signal.shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
