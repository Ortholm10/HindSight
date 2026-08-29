"""AST scan of a strategy file into candidate leak sites."""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import LeakCandidate


def scan_file(path: Path) -> list[LeakCandidate]:
    raise NotImplementedError
