"""Query the store of previously confirmed leak signatures.

A tool, so it returns data and decides nothing: the orchestrator reads the hit
and chooses what to do with it. What it can save is the triage LLM call. What
it cannot save, and must never be used to save, is the execution that turns a
candidate into a finding.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_core.memory import DEFAULT_PATH, lookup
from hindsight_core.models import LeakCandidate


def check_memory(
    candidate: LeakCandidate, path: Path = DEFAULT_PATH
) -> dict[str, object] | None:
    """The stored entry for this candidate's signature, or None."""
    return lookup(path, candidate)
