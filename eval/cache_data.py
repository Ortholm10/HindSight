"""Pre-cache three years of daily bars to disk. Run once; the cache is committed."""

from __future__ import annotations

from pathlib import Path


def cache(symbols: list[str], out_dir: Path) -> list[Path]:
    raise NotImplementedError
