"""JSON store of confirmed leak signatures."""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import Finding


def load(path: Path) -> list[dict[str, object]]:
    raise NotImplementedError


def record(path: Path, finding: Finding) -> None:
    raise NotImplementedError
