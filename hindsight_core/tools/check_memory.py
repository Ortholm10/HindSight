"""Query the store of previously confirmed leak signatures."""

from __future__ import annotations

from hindsight_core.models import LeakCandidate


def check_memory(candidate: LeakCandidate) -> dict[str, object] | None:
    raise NotImplementedError
