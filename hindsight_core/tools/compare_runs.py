"""Metric deltas between two execution records. All new metrics land here.

The two runs are addressed by run_id, not by object, so a comparison is always
reproducible from what is on disk — the same records the report will cite.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import RunComparison, RunRecord, SandboxOutcome
from hindsight_core.tools.run_backtest import RUNS_DIR, load_run


def compare_runs(
    before_run_id: str, after_run_id: str, store: Path = RUNS_DIR
) -> RunComparison:
    before, after = load_run(before_run_id, store), load_run(after_run_id, store)
    return RunComparison(
        before=before,
        after=after,
        delta=_delta(before, after),
        before_equity=list(before.equity),
        after_equity=list(after.equity),
    )


def _delta(before: RunRecord, after: RunRecord) -> dict[str, float]:
    """Empty unless both runs actually completed.

    A crashed or zero-trade run carries no metrics, and subtracting from nothing
    would manufacture a number that looks like a proven improvement. The three
    failure modes stay distinguishable precisely because this returns {}.
    """
    if SandboxOutcome.COMPLETED not in (before.outcome, after.outcome):
        return {}
    if before.outcome is not after.outcome:
        return {}
    return {
        name: after.metrics[name] - value
        for name, value in before.metrics.items()
        if name in after.metrics
    }
