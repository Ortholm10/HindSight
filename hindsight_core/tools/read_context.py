"""Pull the surrounding source for a candidate leak site."""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import LeakCandidate


def read_context(path: Path, candidate: LeakCandidate, radius: int = 15) -> str:
    raise NotImplementedError
