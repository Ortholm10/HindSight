"""Metric deltas between two execution records. All new metrics land here."""

from __future__ import annotations

from hindsight_core.models import RunRecord


def compare_runs(before: RunRecord, after: RunRecord) -> dict[str, float]:
    raise NotImplementedError
