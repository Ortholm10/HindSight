import json

import pytest

TRIVIAL_STRATEGY = """
import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    sma = df["close"].rolling(20).mean()
    return (df["close"] > sma).shift(1).fillna(False).astype(int)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
"""

NEVER_TRADES = """
import pandas as pd


def run_positions(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0, index=df.index)


def run_strategy(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)
"""

META = {
    "case_id": "fixture_case",
    "kind": "clean",
    "leak_type": None,
    "ground_truth_file": None,
    "ground_truth_line": None,
    "expected_inflation": "none",
    "freqtrade_expressible": "yes",
    "patchable": True,
    "symbols": ["SPY"],
    "start": "2022-01-03",
    "end": "2024-12-31",
    "seed": 20220103,
    "traps": ["L03"],
    "causal_check": True,
    "known_limitations": [],
    "limitation_reason": "",
    "description": "fixture",
}


@pytest.fixture
def make_case(tmp_path):
    def _make(source=TRIVIAL_STRATEGY, **meta_overrides):
        meta = {**META, **meta_overrides}
        case_dir = tmp_path / meta["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
        (case_dir / "strategy.py").write_text(source, "utf-8")
        return case_dir

    return _make


@pytest.fixture
def never_trades():
    return NEVER_TRADES
