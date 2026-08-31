"""Every detector the harness can score, on one interface.

The first two bracket the range of possible behaviour. Neither is useful for
finding leaks; they exist to verify the harness itself: one proves it does not
invent detections, the other proves it can recognise a detector that is merely
highlighting everything. A harness checked against only the first would score a
line-flagging detector as perfect.

The last three are Hindsight's own: the straight-line pipeline, the agent, and
the agent again with its unpatchable boundary reports included. All scored on
exactly the same rules, which is what makes the pipeline-versus-agent delta a
measurement rather than a claim.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

from eval.cache_data import DATA_DIR
from hindsight_core.events import EventEmitter
from hindsight_core.models import Event, EventType, LeakCandidate
from hindsight_core.orchestrator import audit as agent_audit
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


@contextmanager
def _cold_memory():
    """An empty leak-signature store, scoped to one scored case.

    The agent's memory is a real product feature and it persists between real
    audits. It must NOT persist inside the eval: a warm store makes case N's
    result depend on cases 1..N-1 and on every audit ever run on the machine,
    so a published score would stop reproducing from a clean clone. Each case
    is therefore scored cold, which is also the harder measurement.
    """
    with tempfile.TemporaryDirectory(prefix="hindsight-eval-memory-") as tmp:
        yield Path(tmp) / "memory.json"


def agent_detector(path: Path) -> list[LeakCandidate]:
    """The agent scored as a detector, on exactly the pipeline's rules.

    Proven leaks only. A candidate the agent detected but could not repair
    mechanically is real information, and it is deliberately NOT here: it has
    no execution record behind it, and the product refuses to report one of
    those as a finding. `agent-reported` is where it appears.
    """
    with _cold_memory() as memory_path:
        findings = agent_audit(
            path, _data_for(path), EventEmitter(), memory_path=memory_path
        )
    return [f.candidate for f in findings]


def agent_reported_detector(path: Path) -> list[LeakCandidate]:
    """Proven leaks plus the boundary cases the agent could not patch.

    Scored against the same eight controls as everything else, so a boundary
    report that fires on clean code lands in the false-positive column rather
    than passing as candour. Publishing the honest extra has to be able to cost
    something, or it is not a measurement.
    """
    emitter = EventEmitter()
    events: list[Event] = []
    emitter.subscribe(events.append)
    with _cold_memory() as memory_path:
        findings = agent_audit(path, _data_for(path), emitter, memory_path=memory_path)
    boundary = [
        LeakCandidate(**entry["candidate"])
        for event in events
        if event.type is EventType.FINAL
        for entry in event.payload.get("unproven", ())
        if entry["status"] == "not_mechanically_patchable"
    ]
    return [f.candidate for f in findings] + boundary


DETECTORS = {
    "null": null_detector,
    "everything": everything_detector,
    "pipeline": pipeline_detector,
    "agent": agent_detector,
    "agent-reported": agent_reported_detector,
}
