"""Run a strategy file in the sandbox and return its execution record."""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import RunRecord


def run_backtest(path: Path, data_path: Path, timeout_s: float = 60.0) -> RunRecord:
    raise NotImplementedError
