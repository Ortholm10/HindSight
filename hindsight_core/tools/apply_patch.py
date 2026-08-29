"""libcst transforms from the closed fix vocabulary. Returns a unified diff."""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import LeakCandidate


def apply_patch(path: Path, candidate: LeakCandidate, operation: str) -> str:
    raise NotImplementedError
