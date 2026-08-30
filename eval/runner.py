"""Load a frozen case, run it on the committed cache, and measure it.

Eval cases are our own frozen code, so they run in-process: no subprocess, no
timeout. Untrusted user code is Phase 2's sandbox problem, not this module's.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from eval.cache_data import DATA_DIR, load
from hindsight_core.metrics import evaluate
from hindsight_core.models import RunRecord

CASES_DIR = Path(__file__).resolve().parent / "cases"

Window = tuple[str, str]

# Three non-overlapping periods, one per calendar year of the pinned range.
# Case validity rule 4 asks whether an inflation survives a change of window;
# a single window cannot answer that, and a Sharpe standard error near 0.58
# means one window agreeing is not evidence.
WINDOWS: tuple[Window, ...] = (
    ("2022-01-03", "2022-12-30"),
    ("2023-01-03", "2023-12-29"),
    ("2024-01-02", "2024-12-31"),
)


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    kind: str  # "injected" | "clean"
    leak_type: str | None
    ground_truth_file: str | None
    ground_truth_line: int | None
    expected_inflation: str  # "up" | "none"
    freqtrade_expressible: str  # "yes" | "no" | "partly"
    patchable: bool
    symbols: list[str]
    start: str
    end: str
    seed: int
    traps: list[str]
    # False where the strategy refits a model over whatever history it is
    # given: truncating the frame legitimately changes its decisions, so the
    # truncation causality check cannot separate that from a leak.
    causal_check: bool
    # Validity rules this case provably cannot meet on this data. Recorded
    # rather than quietly weakening the rule for everyone.
    known_limitations: list[str]
    limitation_reason: str
    description: str
    path: Path
    # Deliberate edits to a frozen case, made after the freeze. Distinct from
    # known_limitations, which records a validity rule a case cannot meet: a
    # correction repairs the case's own construction, and the case must still
    # pass every rule afterwards. Empty, and intended to stay that way - the
    # schema exists so that if a case ever does need editing, the edit has to
    # be declared and reasoned rather than slipped into a case file.
    locked_corrections: list[str] = field(default_factory=list)
    correction_reason: str = ""

    @property
    def is_injected(self) -> bool:
        return self.kind == "injected"

    def source(self, variant: str = "strategy") -> Path:
        return self.path / f"{variant}.py"


def load_case(case_dir: Path) -> CaseMeta:
    raw = json.loads((case_dir / "meta.json").read_text("utf-8"))
    return CaseMeta(**raw, path=case_dir)


def discover_cases(cases_dir: Path = CASES_DIR) -> list[CaseMeta]:
    """Cases are found by globbing, never by a registry — adding case 21 later
    must not require touching the frozen 20."""
    return [load_case(meta.parent) for meta in sorted(cases_dir.glob("*/meta.json"))]


def load_frame(meta: CaseMeta, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = [load(symbol, data_dir) for symbol in meta.symbols]
    if len(frames) == 1:
        df = frames[0]
    else:
        # One frame, symbol-major MultiIndex columns: the contract stays
        # run_strategy(df) even for the universe cases.
        df = pd.concat(frames, axis=1, keys=meta.symbols)
    return df.loc[meta.start : meta.end]


def _import_source(path: Path) -> ModuleType:
    name = f"_hindsight_case_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import case source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_id(meta: CaseMeta, variant: str, window: Window | None, source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    key = "|".join([meta.case_id, variant, str(window), digest])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _execute(
    meta: CaseMeta,
    variant: str,
    window: Window | None,
    data_dir: Path,
    rows: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    df = load_frame(meta, data_dir)
    if window is not None:
        df = df.loc[window[0] : window[1]]
    if rows is not None:
        df = df.iloc[:rows]

    # Seeded per run, not per process: a case must give the same number no
    # matter what ran before it in the same interpreter.
    random.seed(meta.seed)
    np.random.seed(meta.seed)

    module = _import_source(meta.source(variant))
    return module.run_strategy(df), module.run_positions(df)


def run_case(
    meta: CaseMeta,
    variant: str = "strategy",
    window: Window | None = None,
    data_dir: Path = DATA_DIR,
) -> RunRecord:
    equity, positions = _execute(meta, variant, window, data_dir)
    outcome, metrics, changes = evaluate(equity, positions)
    return RunRecord(
        run_id=_run_id(meta, variant, window, meta.source(variant).read_text("utf-8")),
        outcome=outcome,
        metrics=metrics,
        position_changes=changes,
    )


def positions_of(
    meta: CaseMeta,
    variant: str = "strategy",
    window: Window | None = None,
    rows: int | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.Series:
    """The position series, optionally computed on only the first `rows` bars.

    Truncating the frame is how a leak is proven without reference to any
    Sharpe threshold: a causal strategy's decision for day t cannot change when
    the days after t are deleted.
    """
    return _execute(meta, variant, window, data_dir, rows)[1]
