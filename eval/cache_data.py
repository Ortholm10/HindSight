"""Pre-cache the frozen eval data to disk. Run once; the cache is committed.

An eval run must never touch the network — a judge reproducing the results may
not have any. This module is the only place in the project that does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# Pinned, never derived from today's date. Three calendar years of daily bars:
# the standard error of an annualised Sharpe drops from ~1.0 (one year) to
# ~0.58, and there is room for the three non-overlapping robustness windows.
START = "2022-01-03"
END = "2024-12-31"

# SPY carries the single-instrument cases. The rest exist for L12, which needs a
# universe wide enough that survivorship filtering changes the answer.
SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE"]

COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch(symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    import yfinance

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yfinance.download(
        symbol,
        start=start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[COLUMNS]
    df.index.name = "date"
    return df.sort_index()


def cache(symbols: list[str] | None = None, out_dir: Path = DATA_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for symbol in symbols or SYMBOLS:
        df = fetch(symbol)
        path = out_dir / f"{symbol}.csv"
        # Fixed float format so the committed cache is byte-stable across runs.
        df.to_csv(path, float_format="%.6f", lineterminator="\n")
        written.append(path)
    return written


def load(symbol: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{symbol}.csv", index_col="date", parse_dates=["date"])
    return df.sort_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=None)
    args = parser.parse_args()
    for p in cache(args.symbols or None):
        print(p)
