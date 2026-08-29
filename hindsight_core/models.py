"""Typed records that cross every module boundary in Hindsight."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class SandboxOutcome(StrEnum):
    """The four — and only four — ways a sandboxed strategy run can end.

    Defined once here because every downstream branch keys off it. A run that
    produced no trades is NOT a success with a zero metric: it is untestable,
    and collapsing it into COMPLETED would let an unprovable finding through.
    """

    COMPLETED = "completed"
    CRASHED = "crashed"
    TIMED_OUT = "timed_out"
    ZERO_TRADES = "zero_trades"


class EventType(StrEnum):
    SCAN_COMPLETE = "scan_complete"
    TRIAGE = "triage"
    BASELINE = "baseline"
    PROVE_START = "prove_start"
    PROVE_RESULT = "prove_result"
    AGENT_DECISION = "agent_decision"
    FINAL = "final"


@dataclass(frozen=True)
class LeakCandidate:
    """A suspected leak site. Suspected only — nothing here is proven."""

    leak_type: str  # taxonomy ID, e.g. "L03"
    file: str
    line: int
    snippet: str
    reason: str
    confidence: float = 0.0


@dataclass(frozen=True)
class RunRecord:
    """One sandboxed execution of a strategy."""

    run_id: str
    outcome: SandboxOutcome
    metrics: dict[str, float] = field(default_factory=dict)
    position_changes: int = 0
    stderr: str = ""
    duration_s: float = 0.0


@dataclass(frozen=True)
class Finding:
    """A leak proven by execution.

    Structurally impossible to construct without both run IDs — see CLAUDE.md
    critical rule 1. This constraint is the product.
    """

    candidate: LeakCandidate
    before_run_id: str
    after_run_id: str
    diff: str
    delta: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in (
                ("before_run_id", self.before_run_id),
                ("after_run_id", self.after_run_id),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Finding requires execution records; missing: {', '.join(missing)}"
            )


@dataclass(frozen=True)
class Event:
    type: EventType
    payload: dict[str, object] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
