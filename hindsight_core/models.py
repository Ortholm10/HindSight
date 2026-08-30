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
    # Carried on the record rather than recomputed at display time: the verdict
    # screen overlays before/after curves, and re-running a strategy to redraw a
    # chart would be a second execution with no proof value.
    equity: tuple[tuple[str, float], ...] = ()


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
class PatchResult:
    """The outcome of one attempted repair.

    A failed transform is data, not an exception: the agent has to read why the
    patch did not apply and choose a different operation. `patched_source` is
    the untouched original on failure, so a caller that writes it out
    unconditionally still cannot produce a half-edited file.
    """

    ok: bool
    patched_source: str
    diff: str = ""
    error: str = ""


@dataclass(frozen=True)
class RunComparison:
    """Before/after for one repair, including the curves the verdict screen plots.

    The curves travel with the comparison rather than being recomputed at render
    time — redrawing a chart must never cost another execution.
    """

    before: RunRecord
    after: RunRecord
    delta: dict[str, float]
    before_equity: list[tuple[str, float]]
    after_equity: list[tuple[str, float]]


@dataclass(frozen=True)
class Event:
    type: EventType
    payload: dict[str, object] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class Budget:
    """Hard ceilings on one audit's two spendable resources.

    Mutable on purpose: it is loop state, not a record crossing a boundary.
    Hitting a ceiling is a *verdict*, never a quiet stop — the orchestrator
    turns `exhausted` into "stopped on budget", which is a different answer
    from "clean" and must never be rendered as one.
    """

    max_llm_calls: int = 50
    max_sandbox_runs: int = 60
    llm_calls: int = 0
    sandbox_runs: int = 0
    # The first ceiling reached, kept rather than recomputed: a later cap being
    # hit too does not change which one actually stopped the audit.
    hit: str = ""

    def spend_llm(self) -> bool:
        if self.llm_calls >= self.max_llm_calls:
            self.hit = self.hit or f"llm call cap of {self.max_llm_calls} reached"
            return False
        self.llm_calls += 1
        return True

    def spend_run(self) -> bool:
        if self.sandbox_runs >= self.max_sandbox_runs:
            self.hit = self.hit or f"sandbox run cap of {self.max_sandbox_runs} reached"
            return False
        self.sandbox_runs += 1
        return True

    @property
    def exhausted(self) -> bool:
        return bool(self.hit)

    @property
    def reason(self) -> str:
        return self.hit


@dataclass(frozen=True)
class ProofAttempt:
    """One repair tried against one candidate, and what it cost to learn."""

    operation: str
    status: str
    applied: bool
    error: str = ""
    before_run_id: str = ""
    after_run_id: str = ""
    delta: dict[str, float] = field(default_factory=dict)
    diff: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class ProofResult:
    """A prover's report on one candidate after its own retry loop.

    `finding` is present only for "proven". Every other status is a distinct
    reason the candidate is unproven, and they are kept apart because they ask
    for different next moves — repair the patch, find data it trades on, or
    report a boundary the fix vocabulary cannot express.
    """

    candidate: LeakCandidate
    status: str
    attempts: tuple[ProofAttempt, ...]
    finding: Finding | None = None
    patched_source: str = ""
