"""Differential execution prover: patch, re-run, compare, judge.

Owns its own retry loop. Returns a Finding only when the before/after runs
both COMPLETED and the delta is real; otherwise returns why not.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import Finding, LeakCandidate


def prove_leak(path: Path, candidate: LeakCandidate, data_path: Path) -> Finding | None:
    raise NotImplementedError
