"""Baseline #2: freqtrade lookahead-analysis, invoked as a subprocess.

Only cases marked `freqtrade_expressible: "yes"` in meta.json get a real run.
"partly" and "no" cases are recorded not_applicable, never as a miss — see
CLAUDE.md and eval/runner.CaseMeta.freqtrade_expressible.

The shim strategy below does not reimplement each case's signal logic in
freqtrade's indicator language. It imports the case's own `run_positions`
function and calls it from `populate_indicators`, converting the resulting
0/1 position series into enter/exit signals. freqtrade's lookahead-analysis
re-runs `populate_indicators` on truncated slices of the same data and diffs
early-row signal values against the full-history run — which is exactly a
causality test on `run_positions` itself, so the shim is a faithful adapter,
not a re-implementation.

Known, disclosed limitation of this wrapper (see eval/baselines/README or the
project report): freqtrade's raw `has_bias` verdict also fires on every clean
control case here, because converting a 0/1 position series into discrete
enter/exit *events* interacts with freqtrade's next-candle market-order fill
timing and reads as "biased" entry/exit timestamps even for causal strategies.
The `biased_indicators` list — which compares indicator *column values*
directly rather than trade timestamps — does not show this noise and is the
more trustworthy signal, but it also has a real blind spot: a single-candle
`.shift(-1)` leak (case L01) falls entirely inside the one-candle window that
freqtrade's own truncation boundary already grants for order-fill latency, so
it is invisible to this check too. Confirmed as a genuine, hardcoded tool
limitation, not a configuration gap on our side: `freqtrade lookahead-analysis
--help` exposes no flag that controls this window (checked every option;
--minimum-trade-amount/--targeted-trade-amount only bound how many trades are
analyzed, --timeframe-detail only affects sub-candle price-path simulation),
and freqtrade/optimize/analysis/lookahead.py (installed package, not ours),
lines 146-148 and 156-158, hardcodes the truncation boundary to exactly one
candle with no config.json key or multiplier:
    entry_varHolder.to_dt = result_row["open_date"] + timedelta(
        minutes=timeframe_to_minutes(self.full_varHolder.timeframe)
    )
    # comment in that file: "to_dt needs +1 candle since it won't buy on the
    # last candle" — a deliberate design choice by freqtrade's own authors.
Both facts are reported as-is, not patched around, per CLAUDE.md rule 11
("report the numbers that hurt").
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eval.cache_data import DATA_DIR

WORKSPACE = Path(__file__).resolve().parent / ".ft_workspace"
DATADIR = WORKSPACE / "data"
STRATEGY_DIR = WORKSPACE / "strategies"
CONFIG_PATH = WORKSPACE / "config.json"
EXCHANGE = "binance"  # never contacted: pairlist is static, data is local
PAIR = "BTC/USDT"  # a real market so freqtrade's pairlist validation passes;
# only our own local SPY OHLCV feather data drives the analysis (see
# _ensure_data) — no BTC price data is ever fetched or used.
TIMEFRAME = "1d"

FREQTRADE_BIN = str(Path(sys.executable).with_name("freqtrade"))

CONFIG = {
    "dry_run": True,
    "stake_currency": "USDT",
    "stake_amount": 1000,
    "dry_run_wallet": 100000,
    "max_open_trades": 1,
    "trading_mode": "spot",
    "timeframe": TIMEFRAME,
    "dataformat_ohlcv": "feather",
    "unfilledtimeout": {"entry": 10, "exit": 10, "unit": "minutes"},
    "entry_pricing": {
        "price_side": "other",
        "use_order_book": False,
        "check_depth_of_market": {"enabled": False},
    },
    "exit_pricing": {"price_side": "other", "use_order_book": False},
    "exchange": {
        "name": EXCHANGE,
        "pair_whitelist": [PAIR],
        "pair_blacklist": [],
        "ccxt_config": {},
        "ccxt_async_config": {},
    },
    "pairlists": [{"method": "StaticPairList"}],
}

STRATEGY_TEMPLATE = '''"""Generated shim, do not edit — imports the real case logic."""
import hashlib
import importlib.util
import sys

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

_CASE_MODULE_PATH = {module_path!r}


def _load_case_module():
    name = "_ft_case_" + hashlib.sha256(_CASE_MODULE_PATH.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, _CASE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class {class_name}(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "{timeframe}"
    minimal_roi = {{"0": 100}}
    stoploss = -0.99
    startup_candle_count = 250
    use_exit_signal = True
    can_short = False
    process_only_new_candles = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        case = _load_case_module()
        # freqtrade pads missing calendar candles (weekends/holidays) with
        # 0-volume synthetic rows before this hook runs. The case's own
        # run_positions expects the original trading-day-only index, so drop
        # the padding, run it on the real rows, then hold position over the
        # padded gaps like a real backtest would.
        real = dataframe[dataframe["volume"] > 0]
        frame = real.set_index("date")[["open", "high", "low", "close", "volume"]]
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
        position = case.run_positions(frame)
        position.index = real["date"].to_numpy()
        full_position = position.reindex(dataframe["date"]).ffill().fillna(0)
        dataframe["hs_position"] = full_position.to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prior = dataframe["hs_position"].shift(1).fillna(0)
        dataframe.loc[
            (dataframe["hs_position"] == 1) & (prior != 1), "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prior = dataframe["hs_position"].shift(1).fillna(0)
        dataframe.loc[
            (dataframe["hs_position"] == 0) & (prior == 1), "exit_long"
        ] = 1
        return dataframe
'''


def _class_name(case_id: str) -> str:
    return "HsCase_" + case_id


def _ensure_data() -> None:
    """Convert the committed SPY cache into freqtrade's feather format once."""
    from freqtrade.data.history.datahandlers.featherdatahandler import (
        FeatherDataHandler,
    )
    from freqtrade.enums import CandleType

    handler = FeatherDataHandler(DATADIR)
    if handler.ohlcv_data_min_max(PAIR, TIMEFRAME, CandleType.SPOT)[2] > 0:
        return  # already converted

    df = pd.read_csv(DATA_DIR / "SPY.csv", parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    DATADIR.mkdir(parents=True, exist_ok=True)
    handler.ohlcv_store(PAIR, TIMEFRAME, df, CandleType.SPOT)


def _ensure_config() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")


def _write_strategy(case: Path) -> str:
    case_id = case.name
    class_name = _class_name(case_id)
    module_path = str((case / "strategy.py").resolve())
    text = STRATEGY_TEMPLATE.format(
        module_path=module_path, class_name=class_name, timeframe=TIMEFRAME
    )
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    out = STRATEGY_DIR / f"{class_name}.py"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not out.exists() or hashlib.sha256(out.read_bytes()).hexdigest() != digest:
        out.write_text(text, encoding="utf-8")
    return class_name


_BIAS_COL_RE = re.compile(r"found look ahead bias in column (\S+)\.")
_TOO_FEW_RE = re.compile(r"less than minimum_trade_amount")

TOO_FEW_TRADES = "too_few_trades"


def _parse_output(text: str, class_name: str) -> dict[str, object] | str | None:
    """Parse freqtrade's log output for the bias verdict.

    Not the CSV exporter: freqtrade 2026.7's export_to_csv crashes under this
    pandas version on a "no bias" (empty biased_indicators) result — a
    freqtrade/pandas version-skew bug, not our shim. The same verdict is
    logged as plain text before the exporter runs, so we read that instead.
    """
    if _TOO_FEW_RE.search(text):
        return TOO_FEW_TRADES
    if " : bias detected!" in text:
        has_bias = True
    elif ": no bias detected" in text:
        has_bias = False
    else:
        return None
    biased = sorted(set(_BIAS_COL_RE.findall(text)))
    return {"has_bias": has_bias, "biased_indicators": biased}


def run(case: Path) -> dict[str, object]:
    """`case` is a case directory. Returns not_applicable for inexpressible cases."""
    meta = json.loads((case / "meta.json").read_text("utf-8"))
    expressible = meta.get("freqtrade_expressible", "no")
    if expressible != "yes":
        reason = (
            "meta.json marks this case freqtrade_expressible="
            f"{expressible!r}: its logic cannot be faithfully reduced to a "
            "single-pair OHLCV indicator/signal strategy without changing "
            "what is being tested."
        )
        return {
            "case_id": case.name,
            "applicable": False,
            "reason": reason,
            "has_bias": None,
            "biased_indicators": [],
        }

    _ensure_data()
    _ensure_config()
    class_name = _write_strategy(case)

    start = meta["start"].replace("-", "")
    end = meta["end"].replace("-", "")
    cmd = [
        FREQTRADE_BIN,
        "lookahead-analysis",
        "-c",
        str(CONFIG_PATH),
        "-d",
        str(DATADIR),
        "--userdir",
        str(WORKSPACE),
        "--strategy-path",
        str(STRATEGY_DIR),
        "-s",
        class_name,
        "-p",
        PAIR,
        "-i",
        TIMEFRAME,
        "--timerange",
        f"{start}-{end}",
        "--minimum-trade-amount",
        "5",
        "--targeted-trade-amount",
        "20",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = proc.stdout + "\n" + proc.stderr

    parsed = _parse_output(output, class_name)
    if parsed == TOO_FEW_TRADES:
        return {
            "case_id": case.name,
            "applicable": True,
            "reason": (
                "freqtrade found too few trades in this window to run "
                "lookahead-analysis (minimum 5) — inconclusive, not a miss."
            ),
            "has_bias": None,
            "biased_indicators": [],
            "error": None,
        }
    if proc.returncode != 0 or parsed is None:
        return {
            "case_id": case.name,
            "applicable": True,
            "reason": "",
            "has_bias": None,
            "biased_indicators": [],
            "error": output[-4000:],
        }

    return {
        "case_id": case.name,
        "applicable": True,
        "reason": "",
        "has_bias": parsed["has_bias"],
        "biased_indicators": parsed["biased_indicators"],
        "error": None,
    }
