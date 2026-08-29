"""Runs a detector over the frozen cases and reports what it got right.

Scoring is deliberately two-sided. Recall over the 12 injected cases says
whether a detector finds leaks; false positives over the 8 controls and
localisation PRECISION say whether it is discriminating or just underlining
every line it sees. A detector that flags everything scores 12/12 on the first
and must be visibly terrible on the other two.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.runner import CASES_DIR, CaseMeta, discover_cases, run_case
from hindsight_core.models import LeakCandidate, SandboxOutcome

Detector = Callable[[Path], Sequence[LeakCandidate]]

SUITES = ("all", "injected", "clean")


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    kind: str
    leak_type: str | None
    candidates: int
    detected: bool
    localised: bool
    type_correct: bool
    on_ground_truth: int
    sharpe_delta: float | None
    known_limitations: list[str]

    @property
    def false_positive(self) -> bool:
        return self.kind == "clean" and self.candidates > 0


@dataclass(frozen=True)
class SuiteResult:
    suite: str
    results: list[CaseResult]

    @property
    def injected(self) -> list[CaseResult]:
        return [r for r in self.results if r.kind == "injected"]

    @property
    def controls(self) -> list[CaseResult]:
        return [r for r in self.results if r.kind == "clean"]

    @property
    def injected_total(self) -> int:
        return len(self.injected)

    @property
    def clean_total(self) -> int:
        return len(self.controls)

    @property
    def detected(self) -> int:
        return sum(r.detected for r in self.injected)

    @property
    def localised(self) -> int:
        return sum(r.localised for r in self.injected)

    @property
    def type_correct(self) -> int:
        return sum(r.type_correct for r in self.injected)

    @property
    def false_positives(self) -> int:
        return sum(r.false_positive for r in self.controls)

    @property
    def candidates_on_injected(self) -> int:
        return sum(r.candidates for r in self.injected)

    @property
    def localisation_precision(self) -> float:
        """Share of reported candidates sitting on a ground-truth line.

        Zero when nothing was reported: a detector that says nothing has not
        localised anything, and calling that undefined would let it vanish from
        the column rather than score badly in it.
        """
        total = self.candidates_on_injected
        if total == 0:
            return 0.0
        return sum(r.on_ground_truth for r in self.injected) / total


def _select(suite: str, case_id: str | None, cases_dir: Path) -> list[CaseMeta]:
    cases = discover_cases(cases_dir)
    if case_id is not None:
        chosen = [c for c in cases if c.case_id == case_id]
        if not chosen:
            raise ValueError(
                f"no case named {case_id!r}; known cases: "
                f"{', '.join(c.case_id for c in cases)}"
            )
        return chosen
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose one of {', '.join(SUITES)}")
    if suite == "injected":
        return [c for c in cases if c.is_injected]
    if suite == "clean":
        return [c for c in cases if not c.is_injected]
    return cases


def _sharpe_delta(meta: CaseMeta) -> float | None:
    if not meta.is_injected or not meta.patchable:
        return None
    clean, leaked = run_case(meta, "clean"), run_case(meta, "strategy")
    if clean.outcome is not SandboxOutcome.COMPLETED:
        return None
    if leaked.outcome is not SandboxOutcome.COMPLETED:
        return None
    return leaked.metrics["sharpe"] - clean.metrics["sharpe"]


def score_case(
    meta: CaseMeta, detector: Detector, with_metrics: bool = True
) -> CaseResult:
    candidates = list(detector(meta.source("strategy")))
    on_truth = [c for c in candidates if c.line == meta.ground_truth_line]
    return CaseResult(
        case_id=meta.case_id,
        kind=meta.kind,
        leak_type=meta.leak_type,
        candidates=len(candidates),
        detected=meta.is_injected and bool(candidates),
        localised=bool(on_truth),
        type_correct=any(c.leak_type == meta.leak_type for c in on_truth),
        on_ground_truth=len(on_truth),
        sharpe_delta=_sharpe_delta(meta) if with_metrics else None,
        known_limitations=list(meta.known_limitations),
    )


def run_suite(
    detector: Detector,
    suite: str = "all",
    case_id: str | None = None,
    cases_dir: Path = CASES_DIR,
    with_metrics: bool = True,
) -> SuiteResult:
    cases = _select(suite, case_id, cases_dir)
    return SuiteResult(
        suite=case_id or suite,
        results=[score_case(c, detector, with_metrics) for c in cases],
    )


def format_table(result: SuiteResult) -> str:
    header = (
        f"{'case':<32} {'type':<5} {'cands':>5} {'hit':>4} {'line':>5} "
        f"{'type':>5} {'dSharpe':>8}  notes"
    )
    lines = [header, "-" * len(header)]
    for row in result.results:
        delta = f"{row.sharpe_delta:+.2f}" if row.sharpe_delta is not None else ""
        notes = ",".join(row.known_limitations)
        if row.kind == "clean":
            verdict = "FP" if row.false_positive else "ok"
            lines.append(
                f"{row.case_id:<32} {'clean':<5} {row.candidates:>5} {verdict:>4} "
                f"{'':>5} {'':>5} {delta:>8}  {notes}"
            )
        else:
            lines.append(
                f"{row.case_id:<32} {row.leak_type or '':<5} {row.candidates:>5} "
                f"{'yes' if row.detected else 'no':>4} "
                f"{'yes' if row.localised else 'no':>5} "
                f"{'yes' if row.type_correct else 'no':>5} {delta:>8}  {notes}"
            )
    lines += [
        "-" * len(header),
        f"detected            {result.detected}/{result.injected_total}",
        f"line localised      {result.localised}/{result.injected_total}",
        f"leak type correct   {result.type_correct}/{result.injected_total}",
        f"false positives     {result.false_positives}/{result.clean_total}",
        f"localisation prec.  {result.localisation_precision:.1%} "
        f"({result.candidates_on_injected} candidates reported)",
    ]
    return "\n".join(lines)
