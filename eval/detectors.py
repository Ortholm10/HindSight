"""Reference detectors that bracket the range of possible behaviour.

Neither is useful for finding leaks. They exist to verify the harness itself:
one proves it does not invent detections, the other proves it can recognise a
detector that is merely highlighting everything. A harness checked against only
the first would score a line-flagging detector as perfect.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import LeakCandidate


def null_detector(path: Path) -> list[LeakCandidate]:
    """Reports nothing, ever."""
    return []


def everything_detector(path: Path) -> list[LeakCandidate]:
    """Reports every non-trivial line as a leak."""
    candidates = []
    for number, text in enumerate(path.read_text("utf-8").splitlines(), start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidates.append(
            LeakCandidate(
                leak_type="L03",
                file=path.name,
                line=number,
                snippet=stripped,
                reason="flagged unconditionally",
                confidence=1.0,
            )
        )
    return candidates


DETECTORS = {"null": null_detector, "everything": everything_detector}
