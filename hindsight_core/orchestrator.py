"""The agent loop: state -> decide -> tool -> observe -> update -> decide again.

Tools return data. A prover returns a verdict on one candidate. This module is
the only place that decides what happens next, and it is written to be read end
to end, because the loop is the product's argument.

Three things here are what pipeline.py does not do, and they are the entire
difference between the two files:

* A repair that fails is not the end of a candidate. The prover retries.
* A repair that works is not the end of the audit. The fix is kept, the
  baseline is re-run, and every candidate previously filed as "no effect" goes
  back in the queue — because that verdict was a statement about a baseline
  which no longer exists, not about a line of code. This is the load-bearing
  idea; the rest of this module is bookkeeping around it.
* Running out of budget is not "clean". It is its own verdict and says so.

Loop length is not fixed and is not a pass count. It ends when the queue is
empty or when a ceiling is hit — and how long that takes depends on what the
sandbox says, which is not knowable in advance.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from hindsight_core.events import EventEmitter
from hindsight_core.hooks.verification import verify_findings
from hindsight_core.llm import LLMError, complete
from hindsight_core.models import (
    Budget,
    Event,
    EventType,
    Finding,
    LeakCandidate,
    ProofResult,
    RunRecord,
    SandboxOutcome,
)

# Triage is imported, not reimplemented. The pipeline is the thing this file has
# to beat, and it has to beat it on the loop rather than on a better prompt — so
# both ask the model exactly the same question about exactly the same candidate.
from hindsight_core.pipeline import _as_payload, finding_payload, triage
from hindsight_core.provers.differential import prove_leak
from hindsight_core.tools.run_backtest import RUNS_DIR, run_backtest
from hindsight_core.tools.scan_file import scan_file

TRAJECTORY_DIR = Path(__file__).resolve().parents[1] / "runs"

# The fallback for "is this believable?", used only when the model cannot be
# reached or does not answer in format. A daily, long-only rule on one liquid
# asset clearing this over three years is not impossible, just unusual enough
# to be worth another pass — which is all this number decides.
# ponytail: one scalar for every strategy shape. A per-class prior would be
# better, and is worth building when the eval holds more than one shape.
PLAUSIBLE_SHARPE = 1.5

JUDGE_SYSTEM = (
    "You judge whether a backtest result is believable for the strategy that "
    "produced it. You answer in the exact format asked for and nothing else."
)

JUDGE_PROMPT = """A backtest has been repaired and re-run.

Strategy source, with the repairs already applied:
{source}

Sharpe after those repairs: {sharpe:.3f}
Repairs applied so far: {applied}

Is that Sharpe believable for this strategy on three years of daily bars, or \
does it still look like something is reading data it should not have? Do not \
be generous: a simple daily rule on one liquid asset rarely clears 1.5.

Answer in exactly this format:
PLAUSIBLE: yes or no
WHY: <one line>
"""

_PLAUSIBLE_RE = re.compile(r"PLAUSIBLE:\s*(yes|no)", re.IGNORECASE)
_WHY_RE = re.compile(r"WHY:\s*(.+)")


@dataclass
class _Queued:
    """One candidate and what the agent currently believes about it.

    An empty verdict means "not yet examined", which is also what a re-opened
    candidate is returned to. That is the whole requeue mechanism.
    """

    candidate: LeakCandidate
    verdict: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.candidate.leak_type, self.candidate.line)


@dataclass
class _State:
    """Everything the loop reasons over, in one object, so that a trajectory is
    a snapshot of it rather than a reconstruction after the fact."""

    source_path: Path
    baseline: RunRecord
    queue: list[_Queued] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    unproven: list[ProofResult] = field(default_factory=list)
    runs: dict[str, RunRecord] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    step: int = 0

    def next_candidate(self) -> _Queued | None:
        return next((q for q in self.queue if not q.verdict), None)

    @property
    def sharpe(self) -> float | None:
        return self.baseline.metrics.get("sharpe")


def audit(
    path: Path,
    data_path: Path,
    emitter: EventEmitter,
    *,
    timeout_s: float = 60.0,
    store: Path = RUNS_DIR,
    budget: Budget | None = None,
    trajectory_dir: Path = TRAJECTORY_DIR,
) -> list[Finding]:
    """Audit one file. Returns only what execution proved.

    Same return type as pipeline.audit on purpose: the two are scored by the
    same harness through the same detector interface, and a wider return type
    here would quietly make that comparison a different measurement.
    """
    budget = budget if budget is not None else Budget()
    events: list[Event] = []
    emitter.subscribe(events.append)
    started = time.time()
    audit_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="hindsight-audit-") as tmp:
        working = Path(tmp) / path.name
        working.write_bytes(path.read_bytes())

        candidates = _rehome(scan_file(working), path)
        emitter.emit(
            EventType.SCAN_COMPLETE,
            file=str(path),
            candidates=[_as_payload(c) for c in candidates],
        )

        budget.spend_run()
        baseline = run_backtest(working, data_path, timeout_s, store)
        _emit_baseline(emitter, baseline)

        if baseline.outcome is not SandboxOutcome.COMPLETED:
            # No measurable starting point means no measurable delta. Triaging
            # candidates that could never be proven would only manufacture
            # findings with nothing behind them.
            reason = (
                f"the unmodified file {baseline.outcome}; with no measurable "
                "starting point, nothing here can be proven either way"
            )
            emitter.emit(
                EventType.FINAL,
                verdict="untestable",
                reason=reason,
                findings=[],
                unproven=[],
                baseline_outcome=baseline.outcome,
                baseline_metrics={},
                original_metrics={},
                budget=_budget_payload(budget),
                stderr=baseline.stderr[-2000:],
            )
            _save(
                trajectory_dir,
                audit_id,
                path,
                data_path,
                "untestable",
                reason,
                [],
                [],
                budget,
                events,
                started,
                baseline,
            )
            return []

        state = _State(
            source_path=working,
            baseline=baseline,
            queue=[_Queued(c) for c in candidates],
            runs={baseline.run_id: baseline},
        )
        verdict, reason = _loop(
            state, path, data_path, emitter, budget, timeout_s, store
        )

        # Nothing reaches a report without passing this. It raises rather than
        # filters, so there is no code path around it.
        findings = verify_findings(state.findings, state.runs)
        emitter.emit(
            EventType.FINAL,
            verdict=verdict,
            reason=reason,
            findings=[finding_payload(f) for f in findings],
            unproven=[_unproven_payload(u) for u in state.unproven],
            baseline_outcome=state.baseline.outcome,
            baseline_metrics=state.baseline.metrics,
            original_metrics=baseline.metrics,
            budget=_budget_payload(budget),
        )
        _save(
            trajectory_dir,
            audit_id,
            path,
            data_path,
            verdict,
            reason,
            findings,
            state.unproven,
            budget,
            events,
            started,
            baseline,
        )
        return findings


def _loop(
    state: _State,
    original: Path,
    data_path: Path,
    emitter: EventEmitter,
    budget: Budget,
    timeout_s: float,
    store: Path,
) -> tuple[str, str]:
    """One turn per unexamined candidate, for as many turns as it takes."""
    while True:
        if budget.exhausted:
            return "stopped_on_budget", (
                f"{budget.reason}; the audit stopped with work outstanding, "
                "which is not a clean result"
            )

        queued = state.next_candidate()
        if queued is None:
            return _verdict(state), _closing_reason(state)

        state.step += 1
        if not budget.spend_llm():
            continue  # the next turn reports the ceiling rather than guessing

        is_leak, operation, answer = triage(state.source_path, queued.candidate)
        emitter.emit(
            EventType.TRIAGE,
            candidate=_as_payload(queued.candidate),
            is_leak=is_leak,
            operation=operation,
            answer=answer,
        )
        if not is_leak:
            queued.verdict = "discarded"
            _decide(
                emitter,
                state,
                budget,
                "discard",
                queued.candidate,
                f"triage says this is not a leak: {_first_line(answer)}",
            )
            continue

        emitter.emit(
            EventType.PROVE_START,
            candidate=_as_payload(queued.candidate),
            operation=operation,
        )
        result = prove_leak(
            state.source_path,
            queued.candidate,
            data_path,
            state.baseline,
            operation=operation,
            budget=budget,
            timeout_s=timeout_s,
            store=store,
            runs=state.runs,
        )
        for attempt in result.attempts:
            after = state.runs.get(attempt.after_run_id)
            emitter.emit(
                EventType.PROVE_RESULT,
                candidate=_as_payload(queued.candidate),
                status=attempt.status,
                operation=attempt.operation,
                applied=attempt.applied,
                error=attempt.error,
                before_run_id=attempt.before_run_id,
                after_run_id=attempt.after_run_id,
                before_metrics=state.baseline.metrics,
                after_metrics=after.metrics if after is not None else {},
                delta=attempt.delta,
                diff=attempt.diff,
                stderr=attempt.stderr,
            )

        queued.verdict = result.status
        if result.status != "proven" or result.finding is None:
            state.unproven.append(result)
            _decide(
                emitter,
                state,
                budget,
                "unproven",
                queued.candidate,
                _unproven_reason(result),
            )
            continue

        state.findings.append(result.finding)
        _accept(state, result, original, data_path, emitter, budget, timeout_s, store)


def _accept(
    state: _State,
    result: ProofResult,
    original: Path,
    data_path: Path,
    emitter: EventEmitter,
    budget: Budget,
    timeout_s: float,
    store: Path,
) -> None:
    """Keep the repair, re-measure, and re-open what the old number decided.

    The requeue below is the load-bearing line of the whole module. "No effect"
    was measured against a baseline this repair has just replaced, so it is no
    longer evidence about anything — and a system that never revisits it is
    exactly the single-pass system this file exists to beat.
    """
    before = state.baseline
    operation = result.attempts[-1].operation
    state.applied.append(f"{operation} at line {result.candidate.line}")
    state.source_path.write_text(result.patched_source, "utf-8")

    budget.spend_run()
    state.baseline = run_backtest(state.source_path, data_path, timeout_s, store)
    state.runs[state.baseline.run_id] = state.baseline
    _emit_baseline(emitter, state.baseline)

    reopened = [q for q in state.queue if q.verdict == "no_effect"]
    for queued in reopened:
        queued.verdict = ""
    state.unproven = [u for u in state.unproven if u.status != "no_effect"]

    known = {q.key for q in state.queue}
    fresh = _rehome(scan_file(state.source_path), original)
    new = [_Queued(c) for c in fresh if (c.leak_type, c.line) not in known]
    state.queue.extend(new)

    plausible, why = _judge_plausible(state, budget)
    _decide(
        emitter,
        state,
        budget,
        "accept",
        result.candidate,
        (
            f"proven by {operation}: Sharpe {_fmt(before.metrics.get('sharpe'))} -> "
            f"{_fmt(state.sharpe)}. "
            f"{'Believable now' if plausible else 'Still not believable'} — {why}. "
            f"Re-opened {len(reopened)} candidate(s) judged against the old "
            f"baseline; {len(new)} new candidate(s) visible after the repair."
        ),
        plausible=plausible,
        reopened=[_as_payload(q.candidate) for q in reopened],
        new_candidates=[_as_payload(q.candidate) for q in new],
    )


def _judge_plausible(state: _State, budget: Budget) -> tuple[bool, str]:
    """Is the number the repairs left behind believable for this strategy?

    Consulted after every accepted fix, and reported whichever way it answers.
    It does not gate the queue — the loop works through every candidate anyway
    — so a wrong "yes" here costs a misleading sentence in the trace, never a
    missed leak. That asymmetry is why it is allowed to be a judgement at all.
    """
    if state.sharpe is None:
        return False, "the repaired strategy produced no measurable Sharpe"
    if not budget.spend_llm():
        return state.sharpe < PLAUSIBLE_SHARPE, (
            f"judged against the {PLAUSIBLE_SHARPE} ceiling; no budget left to ask"
        )
    prompt = JUDGE_PROMPT.format(
        source=state.source_path.read_text("utf-8"),
        sharpe=state.sharpe,
        applied=", ".join(state.applied) or "none",
    )
    try:
        answer = complete(prompt, system=JUDGE_SYSTEM, max_tokens=2048)
    except LLMError as error:
        return state.sharpe < PLAUSIBLE_SHARPE, (
            f"no provider answered ({error}); judged against the "
            f"{PLAUSIBLE_SHARPE} ceiling instead"
        )
    verdict = _PLAUSIBLE_RE.search(answer)
    why = _WHY_RE.search(answer)
    if verdict is None:
        return state.sharpe < PLAUSIBLE_SHARPE, (
            f"unreadable judgement; used the {PLAUSIBLE_SHARPE} ceiling"
        )
    return verdict.group(1).lower() == "yes", (
        why.group(1).strip()[:200] if why else "no reason given"
    )


# Candidates the agent examined and could not settle either way. A run that
# ends holding one of these has not shown the file to be clean — it has shown
# that part of the file could not be tested, which is CLAUDE.md rule 3 applied
# to the verdict rather than only to a single candidate.
_UNSETTLED = ("untestable", "patch_broken", "not_mechanically_patchable")


def _verdict(state: _State) -> str:
    if state.findings:
        return "leaks_proven"
    if any(u.status in _UNSETTLED for u in state.unproven):
        return "inconclusive"
    return "clean"


def _closing_reason(state: _State) -> str:
    unsettled = [u for u in state.unproven if u.status in _UNSETTLED]
    if state.findings:
        return (
            f"{len(state.findings)} leak(s) proven by execution; "
            f"{len(state.unproven)} candidate(s) examined and left unproven"
        )
    if unsettled:
        return (
            f"no leak was proven, but {len(unsettled)} candidate(s) could not be "
            "settled either way — that is inconclusive, not clean"
        )
    if state.unproven:
        return (
            f"nothing proven; all {len(state.unproven)} examined candidate(s) were "
            "measured and changed nothing"
        )
    return "every candidate was examined and none survived triage"


def _unproven_reason(result: ProofResult) -> str:
    """One sentence per status. These are the three failure modes CLAUDE.md
    rule 3 refuses to collapse, plus the vocabulary boundary."""
    tried = ", ".join(a.operation for a in result.attempts) or "no operation"
    if result.status == "not_mechanically_patchable":
        return (
            f"detected, not mechanically patchable: tried {tried}. A correct "
            "repair here needs a substitute column, which is a judgement rather "
            "than a removal — so the boundary is reported instead of a value "
            "being invented to force a patch through"
        )
    if result.status == "untestable":
        return (
            f"tried {tried}; the patched strategy made no trades at all, so it "
            "is untestable rather than clean"
        )
    if result.status == "patch_broken":
        return f"tried {tried}; every repair that applied crashed the strategy"
    if result.status == "patch_failed":
        return f"tried {tried}; no operation in the vocabulary transforms this line"
    return (
        f"tried {tried}; removing this changed nothing measurable, so on the "
        "current baseline it is not the leak"
    )


def _decide(
    emitter: EventEmitter,
    state: _State,
    budget: Budget,
    action: str,
    candidate: LeakCandidate,
    reason: str,
    **extra: object,
) -> None:
    """The event the live UI renders. Discards carry their reason too — a
    candidate the agent dropped is part of the reasoning, not noise."""
    emitter.emit(
        EventType.AGENT_DECISION,
        step=state.step,
        action=action,
        candidate=_as_payload(candidate),
        reason=reason,
        sharpe=state.sharpe,
        findings=len(state.findings),
        pending=sum(1 for q in state.queue if not q.verdict),
        llm_calls=budget.llm_calls,
        sandbox_runs=budget.sandbox_runs,
        **extra,
    )


def _emit_baseline(emitter: EventEmitter, record: RunRecord) -> None:
    emitter.emit(
        EventType.BASELINE,
        run_id=record.run_id,
        outcome=record.outcome,
        metrics=record.metrics,
        position_changes=record.position_changes,
    )


def _rehome(candidates: list[LeakCandidate], original: Path) -> list[LeakCandidate]:
    """Point every candidate at the file the user handed us.

    Scanning happens on a working copy so the audit never edits its own
    evidence, and every operation in the vocabulary rewrites an expression in
    place, so line numbers survive both the copy and the repairs. A report
    citing a temp directory would be unusable, and one citing a line number
    that had drifted would be worse.
    """
    return [replace(c, file=str(original)) for c in candidates]


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _budget_payload(budget: Budget) -> dict[str, object]:
    return {
        "llm_calls": budget.llm_calls,
        "sandbox_runs": budget.sandbox_runs,
        "max_llm_calls": budget.max_llm_calls,
        "max_sandbox_runs": budget.max_sandbox_runs,
        "exhausted": budget.exhausted,
    }


def _unproven_payload(result: ProofResult) -> dict[str, object]:
    return {
        "candidate": _as_payload(result.candidate),
        "status": result.status,
        "reason": _unproven_reason(result),
        "attempts": [asdict(a) for a in result.attempts],
    }


def _save(
    trajectory_dir: Path,
    audit_id: str,
    path: Path,
    data_path: Path,
    verdict: str,
    reason: str,
    findings: list[Finding],
    unproven: list[ProofResult],
    budget: Budget,
    events: list[Event],
    started: float,
    original_baseline: RunRecord,
) -> Path:
    """Every audit leaves a trajectory, including the ones that proved nothing.

    A record that keeps only the successes is a highlight reel, and the runs
    where the agent was wrong are the ones worth reading.
    """
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    out = trajectory_dir / f"{audit_id}.json"
    out.write_text(
        json.dumps(
            {
                "audit_id": audit_id,
                "file": str(path),
                "data": str(data_path),
                "mode": "agent",
                "started": started,
                "duration_s": time.time() - started,
                "verdict": verdict,
                "reason": reason,
                "original_baseline": asdict(original_baseline),
                "budget": _budget_payload(budget),
                "findings": [finding_payload(f) for f in findings],
                "unproven": [_unproven_payload(u) for u in unproven],
                "events": [
                    {"type": e.type.value, "ts": e.ts, "payload": e.payload}
                    for e in events
                ],
            },
            indent=2,
            default=str,
        )
        + "\n",
        "utf-8",
    )
    return out
