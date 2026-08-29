"""Runs every frozen case and emits the results table."""

from __future__ import annotations

from pathlib import Path


def run_suite(suite: str, cases_dir: Path) -> list[dict[str, object]]:
    raise NotImplementedError
