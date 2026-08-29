"""Every detector the harness can score, on one interface.

The first two bracket the range of possible behaviour. Neither is useful for
finding leaks; they exist to verify the harness itself: one proves it does not
invent detections, the other proves it can recognise a detector that is merely
highlighting everything. A harness checked against only the first would score a
line-flagging detector as perfect.

The third is Hindsight's own straight-line pipeline, scored on exactly the same
rules, which is what makes the pipeline-versus-agent delta a measurement rather
than a claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.cache_data import DATA_DIR
from hindsight_core.events import EventEmitter
from hindsight_core.models import LeakCandidate
from hindsight_core.pipeline import audit


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


def pipeline_detector(path: Path) -> list[LeakCandidate]:
    """The straight-line pipeline scored as a detector.

    It reports only what it *proved* - candidates whose repair measurably
    deflated the strategy - so a triage guess that survives no execution never
    reaches this list. That is the whole comparison against the one-shot
    baseline: the baseline reports what it believes, this reports what it ran.
    """
    findings = audit(path, _data_for(path), EventEmitter())
    return [f.candidate for f in findings]


def _data_for(path: Path) -> Path:
    """The CSV a case trades on, from its own meta.json.

    Falls back to SPY for a file that is not an eval case at all. The
    multi-symbol cases have no single CSV and will fail their baseline run -
    reported as an unprovable case rather than quietly scored as clean.
    """
    meta_file = path.parent / "meta.json"
    if not meta_file.exists():
        return DATA_DIR / "SPY.csv"
    symbols = json.loads(meta_file.read_text("utf-8")).get("symbols") or ["SPY"]
    return DATA_DIR / f"{symbols[0]}.csv"


DETECTORS = {
    "null": null_detector,
    "everything": everything_detector,
    "pipeline": pipeline_detector,
}
