"""Injection recipes, implemented literally from docs/taxonomy.md section 4."""

from __future__ import annotations

from pathlib import Path


def inject(clean_source: Path, leak_type: str) -> tuple[str, int]:
    """Return (leaked source, ground-truth line number)."""
    raise NotImplementedError
