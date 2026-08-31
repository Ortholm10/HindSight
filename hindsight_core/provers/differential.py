"""Differential execution prover: patch, re-run, compare, judge, retry.

One prover takes one candidate and owns the whole question "is this a leak?"
from the first repair to the last. It is a subagent in the only sense that
matters here: it holds its own state, decides its own next move from what the
sandbox just told it, and hands back a single typed answer. Provers never see
each other, which is what makes them independent by construction.

The retry ladder is deliberate and has two rungs:

1. **The model.** Shown the traceback, the null delta, or the refusal to apply,
   it picks a different operation. That is the interesting behaviour, so it
   goes first.
2. **The mechanical sweep.** Whichever operations the model did not name, in
   vocabulary order, skipping every one that does not transform this line.
   Finding that out costs no LLM call and no sandbox run, because apply_patch
   is pure — and the rung exists because an audit whose correctness depends on
   a free-tier model naming the right operation is not an audit.

Neither rung can invent a repair: both draw from the same ten operations, and
an operation outside them is refused by name before anything runs.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from hindsight_core.llm import LLMError, complete
from hindsight_core.models import (
    Budget,
    Finding,
    LeakCandidate,
    ProofAttempt,
    ProofResult,
    RunRecord,
    SandboxOutcome,
)
from hindsight_core.skills import load_skill
from hindsight_core.tools.apply_patch import BOUNDARY_MARKER, OPERATIONS, apply_patch
from hindsight_core.tools.compare_runs import compare_runs
from hindsight_core.tools.read_context import read_context
from hindsight_core.tools.run_backtest import RUNS_DIR, run_backtest

# The ceiling on one candidate. Not a pass count for the audit — the loop above
# is uncapped — but a stop on grinding a single line: the vocabulary is ten
# operations and rarely more than three of them touch any given one.
MAX_ATTEMPTS = 5

# Which unproven answer to report when several were seen. A crash outranks a
# null delta because it says the repair itself was wrong; a null delta outranks
# a non-applying operation because at least something ran and was measured.
_INFORMATIVENESS = (
    "patch_broken",
    "untestable",
    "no_effect",
    "patch_failed",
    "no_operation",
)

SYSTEM = (
    "You repair look-ahead bias in Python backtests. Every repair removes "
    "information the strategy was not entitled to. You never add a value the "
    "data did not contain. You answer in the exact format asked for."
)

RETRY_PROMPT = """A repair for a suspected {leak_type} leak did not work.

{skill}

Candidate at {file}:{line}
  snippet: {snippet}

Surrounding code:
{context}

What has already been tried, and what the sandbox said:
{history}

Choose a different repair from this closed vocabulary: {operations}. Do not \
repeat an operation already tried. If no operation in the vocabulary can \
express a correct repair here, answer none — reporting that boundary is better \
than forcing a patch that invents a value the data did not contain.

Answer in exactly this format:
OPERATION: <one operation name, or none>
WHY: <one line>
"""

_OP_RE = re.compile(r"OPERATION:\s*([a-z_]+)", re.IGNORECASE)


def classify(before: RunRecord, after: RunRecord, delta: dict[str, float]) -> str:
    """Which of the four things just happened. Never collapsed into one path.

    Identical to the pipeline's rule on purpose. The agent has to win on the
    loop around this function, not on a softer definition of proof.
    """
    if after.outcome in (SandboxOutcome.CRASHED, SandboxOutcome.TIMED_OUT):
        return "patch_broken"
    if after.outcome is SandboxOutcome.ZERO_TRADES:
        return "untestable"
    if before.outcome is not SandboxOutcome.COMPLETED or "sharpe" not in delta:
        return "untestable"
    return "proven" if delta["sharpe"] < 0 else "no_effect"


def prove_leak(
    path: Path,
    candidate: LeakCandidate,
    data_path: Path,
    baseline: RunRecord,
    *,
    operation: str = "",
    budget: Budget | None = None,
    timeout_s: float = 60.0,
    store: Path = RUNS_DIR,
    runs: dict[str, RunRecord] | None = None,
    max_llm_retries: int = 2,
) -> ProofResult:
    """Try repairs at this line until one deflates the strategy, or none can.

    `runs` is an out-parameter on purpose: the verification hook rejects any
    finding citing a run it cannot find, so every record this prover creates
    has to land somewhere the caller will look.
    """
    budget = budget if budget is not None else Budget()
    runs = runs if runs is not None else {}
    runs.setdefault(baseline.run_id, baseline)

    attempts: list[ProofAttempt] = []
    tried: set[str] = set()
    llm_retries = 0

    next_op = operation if operation in OPERATIONS else ""
    if not next_op:
        next_op, refused = _sweep(path, candidate, tried)
        attempts += refused

    while next_op and len(attempts) < MAX_ATTEMPTS:
        tried.add(next_op)
        outcome = _attempt(
            path,
            candidate,
            next_op,
            data_path,
            baseline,
            timeout_s,
            store,
            runs,
            budget,
        )
        if outcome is None:
            # The sandbox charge was refused. Nothing was learned, so nothing
            # is recorded — an empty attempt in the trajectory would read as a
            # repair that was tried and told us something.
            break
        attempt, patched = outcome
        attempts.append(attempt)

        if attempt.status == "proven":
            return ProofResult(
                candidate=candidate,
                status="proven",
                attempts=tuple(attempts),
                finding=Finding(
                    candidate=candidate,
                    before_run_id=attempt.before_run_id,
                    after_run_id=attempt.after_run_id,
                    diff=attempt.diff,
                    delta=attempt.delta,
                ),
                patched_source=patched,
            )

        next_op = ""
        if llm_retries < max_llm_retries and budget.spend_llm():
            llm_retries += 1
            next_op = _next_operation(path, candidate, attempts, tried)
        if not next_op:
            next_op, refused = _sweep(path, candidate, tried)
            attempts += refused

    return ProofResult(
        candidate=candidate,
        status=_verdict(attempts),
        attempts=tuple(attempts),
    )


def _verdict(attempts: list[ProofAttempt]) -> str:
    """The boundary outranks everything else that happened.

    An operation that ran and changed nothing is evidence that *that operation*
    was wrong. Learning that no operation can express the repair at all is a
    different and larger fact, and it is the one the report has to carry.
    """
    if any(a.error.startswith(BOUNDARY_MARKER) for a in attempts):
        return "not_mechanically_patchable"
    if not attempts:
        return "no_operation"
    seen = {attempt.status for attempt in attempts}
    return next((s for s in _INFORMATIVENESS if s in seen), "no_operation")


def _attempt(
    path: Path,
    candidate: LeakCandidate,
    operation: str,
    data_path: Path,
    baseline: RunRecord,
    timeout_s: float,
    store: Path,
    runs: dict[str, RunRecord],
    budget: Budget,
) -> tuple[ProofAttempt, str] | None:
    """One repair, run to a verdict — the attempt and the patched source.

    None means the sandbox charge was refused, which is not a fact about the
    repair and must not be recorded as one.
    """
    patch = apply_patch(path, candidate, operation)
    if not patch.ok:
        return (
            ProofAttempt(
                operation=operation,
                status="patch_failed",
                applied=False,
                error=patch.error,
            ),
            "",
        )

    if not budget.spend_run():
        return None

    # The patched source goes to a scratch copy, never over the audited file:
    # an audit that edits the thing it is auditing has changed the evidence.
    with tempfile.TemporaryDirectory(prefix="hindsight-patched-") as tmp:
        patched_path = Path(tmp) / path.name
        patched_path.write_text(patch.patched_source, "utf-8")
        after = run_backtest(patched_path, data_path, timeout_s, store)
    runs[after.run_id] = after

    delta = compare_runs(baseline.run_id, after.run_id, store).delta
    return (
        ProofAttempt(
            operation=operation,
            status=classify(baseline, after, delta),
            applied=True,
            before_run_id=baseline.run_id,
            after_run_id=after.run_id,
            delta=delta,
            diff=patch.diff,
            stderr=after.stderr[-2000:],
        ),
        patch.patched_source,
    )


def _sweep(
    path: Path, candidate: LeakCandidate, tried: set[str]
) -> tuple[str, list[ProofAttempt]]:
    """The next untried operation that transforms this line, plus any boundary.

    Free: apply_patch is pure, so asking costs no run and no call. This is the
    floor under the model's judgement rather than a replacement for it.

    The second return value is what stops the boundary from being invisible.
    An operation that refuses to apply is usually just the wrong tool and is
    not worth recording — except when it refuses *because* no removal exists,
    which is the one refusal that is itself the finding. Dropping that on the
    floor is how a detected-but-unpatchable leak silently becomes "no effect".
    """
    refused: list[ProofAttempt] = []
    for operation in OPERATIONS:
        if operation in tried:
            continue
        patch = apply_patch(path, candidate, operation)
        if patch.ok:
            return operation, refused
        if patch.error.startswith(BOUNDARY_MARKER):
            tried.add(operation)
            refused.append(
                ProofAttempt(
                    operation=operation,
                    status="patch_failed",
                    applied=False,
                    error=patch.error,
                )
            )
    return "", refused


def _next_operation(
    path: Path,
    candidate: LeakCandidate,
    attempts: list[ProofAttempt],
    tried: set[str],
) -> str:
    """Ask for a different repair, given what the sandbox just said.

    An unreadable answer, a repeat, or an invented operation all come back as
    "" and fall through to the sweep. A refusal to answer is never allowed to
    read as a verdict about the candidate.
    """
    prompt = RETRY_PROMPT.format(
        leak_type=candidate.leak_type,
        skill=_skill_for(candidate.leak_type),
        file=Path(candidate.file).name,
        line=candidate.line,
        snippet=candidate.snippet,
        context=read_context(path, candidate),
        history=_history(attempts),
        operations=", ".join(op for op in OPERATIONS if op not in tried),
    )
    try:
        answer = complete(prompt, system=SYSTEM, max_tokens=2048)
    except LLMError:
        return ""
    match = _OP_RE.search(answer)
    chosen = match.group(1).lower() if match else ""
    return chosen if chosen in OPERATIONS and chosen not in tried else ""


def _history(attempts: list[ProofAttempt]) -> str:
    """Each prior attempt as one line the model can actually act on."""
    lines = []
    for attempt in attempts:
        if not attempt.applied:
            lines.append(f"- {attempt.operation}: did not apply — {attempt.error}")
        elif attempt.status == "patch_broken":
            tail = (attempt.stderr.strip().splitlines() or [""])[-1]
            lines.append(f"- {attempt.operation}: the patched code crashed — {tail}")
        elif attempt.status == "untestable":
            lines.append(f"- {attempt.operation}: the patched code made no trades")
        elif "sharpe" in attempt.delta:
            lines.append(
                f"- {attempt.operation}: ran clean, Sharpe moved "
                f"{attempt.delta['sharpe']:+.3f} — the wrong direction, so this "
                "did not remove the leak"
            )
        else:
            lines.append(f"- {attempt.operation}: ran, but no delta was measurable")
    return "\n".join(lines)


def _skill_for(leak_type: str) -> str:
    try:
        return load_skill(leak_type)
    except OSError:
        return ""
