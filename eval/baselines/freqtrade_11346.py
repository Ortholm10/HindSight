"""Baseline #2 for case 21 only: a NATIVE freqtrade strategy, not the shim.

eval/baselines/freqtrade.py cannot reproduce issue #11346 even in principle.
It resamples the higher timeframe itself, outside freqtrade, and hands
freqtrade one static pre-merged frame - so it never runs freqtrade's own
informative path, which is where #11346 lives. This module is built separately
for that one case: it stores a real weekly OHLCV feed alongside the daily one
and runs a strategy that reaches for it through informative_pairs() and
self.dp.get_pair_dataframe().

Why lookahead-analysis misses it, from freqtrade's own source rather than from
inference. lookahead.py:146-148 truncates the re-run to the entry candle plus
exactly one base candle, and dataprovider.py::historic_ohlcv loads the
informative series against that same truncated timerange - filtering on each
candle's OPEN date. A weekly candle stamped Monday therefore survives a
truncation to Wednesday, and it survives it WHOLE, because the stored candle
already contains Friday's close. Full run and truncated run read the identical
value, nothing differs, and the tool reports no bias. The blindness is
structural: no truncation on open-date boundaries can expose a leak that lives
inside a single higher-timeframe candle.

Also measured here, and it settles an open question from session 2: the
`has_bias`-fires-on-everything artifact documented in eval/baselines/
freqtrade.py is SHIM-SPECIFIC. On this native strategy freqtrade reports
has_bias=False for both the leaked and the repaired variant. That artifact came
from the shim converting a 0/1 position series into discrete enter/exit events,
not from freqtrade itself. biased_indicators is used as the cross-check
regardless, per CLAUDE.md rule 11.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from eval.baselines.freqtrade import (
    CONFIG,
    EXCHANGE,
    FREQTRADE_BIN,
    PAIR,
    TOO_FEW_TRADES,
    _parse_output,
)
from eval.cache_data import DATA_DIR

WORKSPACE = Path(__file__).resolve().parent / ".ft11346_workspace"
DATADIR = WORKSPACE / "data"
STRATEGY_DIR = WORKSPACE / "strategies"
CONFIG_PATH = WORKSPACE / "config.json"

TIMEFRAME = "1d"
INF_TIMEFRAME = "1w"

# The two variants of the case, and the freqtrade class each one defines.
VARIANTS = {"ft_strategy": "Leak11346", "ft_clean": "Clean11346"}

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "freqtrade_11346.json"


def _ensure_data() -> None:
    """Store the committed SPY cache as both a daily and a weekly feed.

    The weekly bars are stamped at the week's OPEN (Monday), which is the
    exchange convention freqtrade assumes and the stamp the whole bug turns on.
    """
    from freqtrade.data.history.datahandlers.featherdatahandler import (
        FeatherDataHandler,
    )
    from freqtrade.enums import CandleType

    handler = FeatherDataHandler(DATADIR)
    if handler.ohlcv_data_min_max(PAIR, INF_TIMEFRAME, CandleType.SPOT)[2] > 0:
        return

    daily = pd.read_csv(DATA_DIR / "SPY.csv", parse_dates=["date"])
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily = daily[["date", "open", "high", "low", "close", "volume"]]

    weekly = (
        daily.set_index("date")
        .resample("W-MON", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    DATADIR.mkdir(parents=True, exist_ok=True)
    handler.ohlcv_store(PAIR, TIMEFRAME, daily, CandleType.SPOT)
    handler.ohlcv_store(PAIR, INF_TIMEFRAME, weekly, CandleType.SPOT)


def _ensure_config() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        config = {**CONFIG, "timeframe": TIMEFRAME}
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _install(case: Path, variant: str) -> str:
    """Copy the case's own strategy file into the workspace, unmodified.

    Copied rather than templated: the point of this module is that freqtrade
    runs the case's real strategy, so nothing may be generated around it.
    """
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    class_name = VARIANTS[variant]
    source = (case / f"{variant}.py").read_text("utf-8")
    (STRATEGY_DIR / f"{class_name}.py").write_text(source, encoding="utf-8")
    return class_name


def run(case: Path, variant: str = "ft_strategy") -> dict[str, object]:
    """Run freqtrade lookahead-analysis over one variant of case 21."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {list(VARIANTS)}")

    meta = json.loads((case / "meta.json").read_text("utf-8"))
    _ensure_data()
    _ensure_config()
    class_name = _install(case, variant)

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

    base: dict[str, object] = {
        "case_id": case.name,
        "variant": variant,
        "strategy_class": class_name,
        "applicable": True,
        "exchange": EXCHANGE,
    }
    if parsed == TOO_FEW_TRADES:
        return {
            **base,
            "reason": (
                "freqtrade found too few trades to run lookahead-analysis "
                "(minimum 5) - inconclusive, not a miss."
            ),
            "has_bias": None,
            "biased_indicators": [],
            "error": None,
        }
    if proc.returncode != 0 or parsed is None:
        return {
            **base,
            "reason": "",
            "has_bias": None,
            "biased_indicators": [],
            "error": output[-4000:],
        }
    assert isinstance(parsed, dict)
    return {
        **base,
        "reason": "",
        "has_bias": parsed["has_bias"],
        "biased_indicators": parsed["biased_indicators"],
        "error": None,
    }


def run_both(case: Path) -> dict[str, object]:
    """Both variants, plus the verdict that matters: did the tool miss it?"""
    rows = {variant: run(case, variant) for variant in VARIANTS}
    leaked = rows["ft_strategy"]
    # "Detected" follows run_baseline's convention: biased_indicators, not the
    # raw has_bias flag. On this native path both are negative, so they agree.
    detected = (
        bool(leaked["biased_indicators"]) if leaked["has_bias"] is not None else None
    )
    return {
        "baseline": "freqtrade-native",
        "case_id": case.name,
        "issue": "https://github.com/freqtrade/freqtrade/issues/11346",
        "detected_on_leaked_variant": detected,
        "variants": rows,
    }


if __name__ == "__main__":
    from eval.runner import CASES_DIR

    result = run_both(CASES_DIR / "htf_merge_11346")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, row in result["variants"].items():
        print(
            f"{name:<14} has_bias={row['has_bias']!s:<6} "
            f"biased_indicators={row['biased_indicators']} "
            f"error={'yes' if row['error'] else 'no'}"
        )
    print(f"detected on leaked variant: {result['detected_on_leaked_variant']}")
    print(f"written to {RESULTS_PATH}")
