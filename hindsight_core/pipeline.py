"""The straight-line pipeline: scan -> triage -> patch -> run -> delta.

Deliberately not an agent, and kept in the repo for exactly that reason. It
walks the candidate list once, in scan order, asks one question per candidate,
and stops. It never re-scans after a repair, never asks whether the number it
just produced is believable, and never tries a second operation when the first
one fails to apply. Every one of those gaps is a place the agent loop earns its
keep, and the measured difference between this file and that one is the
headline entry in the Improvement Changelog - so this file keeps working.

What it does not give up: nothing here is reported without two execution
records behind it. The dumb version is still not allowed to guess.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from hindsight_core.events import EventEmitter
from hindsight_core.llm import complete
from hindsight_core.models import (
    EventType,
    Finding,
    LeakCandidate,
    RunRecord,
    SandboxOutcome,
)
from hindsight_core.skills import load_skill
from hindsight_core.tools.apply_patch import OPERATIONS, apply_patch
from hindsight_core.tools.compare_runs import compare_runs
from hindsight_core.tools.read_context import read_context
from hindsight_core.tools.run_backtest import RUNS_DIR, run_backtest
from hindsight_core.tools.scan_file import scan_file

SYSTEM = (
    "You audit Python backtests for look-ahead bias: code that reads data "
    "which would not have existed at decision time. You answer in the exact "
    "format asked for and nothing else."
)

PROMPT = """A static scan flagged a possible {leak_type} leak.

{skill}

Candidate at {file}:{line}
  reason: {reason}
  snippet: {snippet}

Surrounding code:
{context}

Is the flagged line a real look-ahead leak? If it is, choose the repair \
operation that removes the illegitimate information. The operations available \
are: {operations}. Never propose adding a value that is not already in the data.

Answer in exactly this format:
LEAK: yes or no
OPERATION: <one operation name, or none>
"""

_LEAK_RE = re.compile(r"LEAK:\s*(yes|no)", re.IGNORECASE)
_OP_RE = re.compile(r"OPERATION:\s*([a-z_]+)", re.IGNORECASE)


def triage(path: Path, candidate: LeakCandidate) -> tuple[bool, str, str]:
    """One LLM call. Returns (is_leak, operation, raw answer).

    An unreadable answer is "no". The alternative - treating a shrug as a leak
    - would spend a sandbox run on noise, and unlike a missed leak it would
    also end up in front of the user.
    """
    prompt = PROMPT.format(
        leak_type=candidate.leak_type,
        skill=_skill_for(candidate.leak_type),
        file=Path(candidate.file).name,
        line=candidate.line,
        reason=candidate.reason,
        snippet=candidate.snippet,
        context=read_context(path, candidate),
        operations=", ".join(OPERATIONS),
    )
    # Budgeted for a reasoning model: the visible answer is two lines, but
    # the thinking that precedes it is charged to the same allowance, and a
    # budget that runs out mid-thought comes back as a truncated fragment.
    answer = complete(prompt, system=SYSTEM, max_tokens=2048)

    leak = _LEAK_RE.search(answer)
    operation = _OP_RE.search(answer)
    is_leak = bool(leak) and leak.group(1).lower() == "yes"
    return is_leak, (operation.group(1).lower() if operation else ""), answer


def _skill_for(leak_type: str) -> str:
    try:
        return load_skill(leak_type)
    except OSError:
        return ""


def classify(before: RunRecord, after: RunRecord, delta: dict[str, float]) -> str:
    """Which of the four things just happened. Never collapsed into one path.

    "patch_broken" and "untestable" both mean the leak is unproven, but they
    ask for different next moves - repair the patch, or find data the strategy
    actually trades on - and a report that cannot tell them apart is telling
    the user nothing.
    """
    if after.outcome in (SandboxOutcome.CRASHED, SandboxOutcome.TIMED_OUT):
        return "patch_broken"
    if after.outcome is SandboxOutcome.ZERO_TRADES:
        return "untestable"
    if before.outcome is not SandboxOutcome.COMPLETED or "sharpe" not in delta:
        return "untestable"
    return "proven" if delta["sharpe"] < 0 else "no_effect"


def audit(
    path: Path,
    data_path: Path,
    emitter: EventEmitter,
    *,
    timeout_s: float = 60.0,
    store: Path = RUNS_DIR,
) -> list[Finding]:
    candidates = scan_file(path)
    emitter.emit(
        EventType.SCAN_COMPLETE,
        file=str(path),
        candidates=[_as_payload(c) for c in candidates],
    )

    baseline = run_backtest(path, data_path, timeout_s, store)
    emitter.emit(
        EventType.BASELINE,
        run_id=baseline.run_id,
        outcome=baseline.outcome,
        metrics=baseline.metrics,
        position_changes=baseline.position_changes,
    )
    if baseline.outcome is not SandboxOutcome.COMPLETED:
        # No measurable starting point means no measurable delta. Stopping here
        # is the honest outcome; triaging candidates we could never prove would
        # only produce findings with nothing behind them.
        emitter.emit(
            EventType.FINAL,
            findings=[],
            baseline_outcome=baseline.outcome,
            reason=f"baseline run was {baseline.outcome}; nothing can be proven",
            stderr=baseline.stderr[-2000:],
        )
        return []

    findings: list[Finding] = []
    for candidate in candidates:
        is_leak, operation, answer = triage(path, candidate)
        emitter.emit(
            EventType.TRIAGE,
            candidate=_as_payload(candidate),
            is_leak=is_leak,
            operation=operation,
            answer=answer,
        )
        if not is_leak:
            continue

        finding = _prove(
            path, data_path, candidate, operation, baseline, emitter, timeout_s, store
        )
        if finding is not None:
            findings.append(finding)

    emitter.emit(
        EventType.FINAL,
        findings=[finding_payload(f) for f in findings],
        baseline_outcome=baseline.outcome,
        baseline_metrics=baseline.metrics,
        reason="",
    )
    return findings


def _prove(
    path: Path,
    data_path: Path,
    candidate: LeakCandidate,
    operation: str,
    baseline: RunRecord,
    emitter: EventEmitter,
    timeout_s: float,
    store: Path,
) -> Finding | None:
    patch = apply_patch(path, candidate, operation)
    if not patch.ok:
        emitter.emit(
            EventType.PROVE_RESULT,
            candidate=_as_payload(candidate),
            status="patch_failed",
            operation=operation,
            error=patch.error,
        )
        return None

    emitter.emit(
        EventType.PROVE_START,
        candidate=_as_payload(candidate),
        operation=operation,
        diff=patch.diff,
    )

    # The patched source is written to a scratch copy, never over the user's
    # file: an audit that edits the thing it is auditing has changed the
    # evidence.
    with tempfile.TemporaryDirectory(prefix="hindsight-patched-") as tmp:
        patched_path = Path(tmp) / path.name
        patched_path.write_text(patch.patched_source, "utf-8")
        after = run_backtest(patched_path, data_path, timeout_s, store)

    comparison = compare_runs(baseline.run_id, after.run_id, store)
    status = classify(baseline, after, comparison.delta)
    emitter.emit(
        EventType.PROVE_RESULT,
        candidate=_as_payload(candidate),
        status=status,
        operation=operation,
        before_run_id=baseline.run_id,
        after_run_id=after.run_id,
        before_metrics=baseline.metrics,
        after_metrics=after.metrics,
        delta=comparison.delta,
        stderr=after.stderr[-2000:],
    )
    if status != "proven":
        return None
    return Finding(
        candidate=candidate,
        before_run_id=baseline.run_id,
        after_run_id=after.run_id,
        diff=patch.diff,
        delta=comparison.delta,
    )


def _as_payload(candidate: LeakCandidate) -> dict[str, object]:
    return {
        "leak_type": candidate.leak_type,
        "file": candidate.file,
        "line": candidate.line,
        "snippet": candidate.snippet,
        "reason": candidate.reason,
        "confidence": candidate.confidence,
    }


def finding_payload(finding: Finding) -> dict[str, object]:
    return {
        "candidate": _as_payload(finding.candidate),
        "before_run_id": finding.before_run_id,
        "after_run_id": finding.after_run_id,
        "diff": finding.diff,
        "delta": finding.delta,
    }
