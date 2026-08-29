"""Run a strategy file in the sandbox and return its execution record.

Persistence is the point as much as execution: every Finding cites a before and
an after run_id, and a run_id nobody can read back later proves nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from hindsight_core.models import RunRecord, SandboxOutcome
from hindsight_core.sandbox import run_sandboxed

RUNS_DIR = Path(__file__).resolve().parents[2] / ".hindsight" / "runs"


def run_backtest(
    path: Path,
    data_path: Path,
    timeout_s: float = 60.0,
    store: Path = RUNS_DIR,
) -> RunRecord:
    record = run_sandboxed(path, data_path, timeout_s)
    save_run(record, store)
    return record


def save_run(record: RunRecord, store: Path = RUNS_DIR) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{record.run_id}.json"
    path.write_text(json.dumps(asdict(record), indent=2), "utf-8")
    return path


def load_run(run_id: str, store: Path = RUNS_DIR) -> RunRecord:
    path = store / f"{run_id}.json"
    if not path.exists():
        raise KeyError(f"no run record for run_id {run_id!r} in {store}")
    raw = json.loads(path.read_text("utf-8"))
    return RunRecord(
        run_id=raw["run_id"],
        outcome=SandboxOutcome(raw["outcome"]),
        metrics=raw["metrics"],
        position_changes=raw["position_changes"],
        stderr=raw["stderr"],
        duration_s=raw["duration_s"],
        equity=tuple((str(d), float(v)) for d, v in raw["equity"]),
    )
