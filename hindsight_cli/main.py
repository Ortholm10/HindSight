"""Thin wrapper over the shared eval code. No scoring logic lives here."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from eval.baselines.run_baseline import build_results
from eval.baselines.run_baseline import format_table as format_baseline_table
from eval.cache_data import DATA_DIR
from eval.detectors import DETECTORS
from eval.harness import SuiteResult, format_table, run_suite
from hindsight_cli.printer import print_event
from hindsight_core import llm
from hindsight_core.events import EventEmitter
from hindsight_core.models import Event
from hindsight_core.orchestrator import audit as agent_audit
from hindsight_core.pipeline import audit as pipeline_audit
from hindsight_core.pipeline import finding_payload

# Two audit paths, both first-class. The pipeline keeps its own name rather than
# becoming "the old default", because the measured difference between the two is
# the headline entry in the changelog and it only stays measurable while both
# still run.
MODES = {"pipeline": pipeline_audit, "agent": agent_audit}

# The columns worth a spread. Localisation precision is derived from two of
# them, so reporting it here as well would be the same information twice.
_SPREAD_FIELDS = (
    "detected",
    "localised",
    "type_correct",
    "false_positives",
    "candidates_on_injected",
)

DEFAULT_DATA = DATA_DIR / "SPY.csv"


def _eval(args: argparse.Namespace) -> int:
    if args.baseline:
        result = build_results(args.baseline)
        text = json.dumps(result, indent=2)
        if args.json:
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(text + "\n", encoding="utf-8")
            else:
                print(text)
        else:
            print(format_baseline_table(result))
        return 0

    passes = [
        _one_pass(args, index, max(1, args.repeat))
        for index in range(1, max(1, args.repeat) + 1)
    ]
    result = passes[0]
    if args.json:
        payload = {
            "suite": result.suite,
            "detector": args.detector,
            "repeat": len(passes),
            **_suite_payload(result),
            "passes": [_suite_payload(p) for p in passes],
            "spread": _spread(passes),
        }
        text = json.dumps(payload, indent=2)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    else:
        print(format_table(result))
        if len(passes) > 1:
            print()
            print(format_spread(passes))
    return 0


def _one_pass(args: argparse.Namespace, index: int, total: int) -> SuiteResult:
    """One scoring pass, given its own prompt cache when there is more than one.

    Without this a repeat measures nothing. The cache is keyed by prompt, so a
    second pass over the same cache replays the first one byte for byte and
    reports a spread of zero no matter how unstable the agent actually is.
    Each pass therefore pays for its own answers, which is the real cost of
    finding out.
    """
    if total > 1:
        llm.CACHE_DIR = llm.pass_cache(index)
    return run_suite(
        DETECTORS[args.detector],
        suite=args.suite,
        case_id=args.case,
        with_metrics=not args.no_metrics,
    )


def _suite_payload(result: SuiteResult) -> dict[str, object]:
    return {
        "detected": result.detected,
        "injected_total": result.injected_total,
        "localised": result.localised,
        "type_correct": result.type_correct,
        "false_positives": result.false_positives,
        "clean_total": result.clean_total,
        "localisation_precision": result.localisation_precision,
        "candidates_on_injected": result.candidates_on_injected,
        "cases": [dataclasses.asdict(r) for r in result.results],
    }


def _spread(passes: list[SuiteResult]) -> dict[str, dict[str, object]]:
    """min, max, and every value. Deliberately not a mean.

    The agent is not deterministic, and a mean over three runs would hide the
    one that went badly — which is the only reason to run it three times.
    """
    spread: dict[str, dict[str, object]] = {}
    for field in _SPREAD_FIELDS:
        values = [getattr(p, field) for p in passes]
        spread[field] = {"min": min(values), "max": max(values), "values": values}
    return spread


def format_spread(passes: list[SuiteResult]) -> str:
    lines = [f"spread over {len(passes)} run(s)", "-" * 48]
    for field, stats in _spread(passes).items():
        values = " ".join(str(v) for v in stats["values"])
        lines.append(f"{field:<24} {stats['min']}-{stats['max']}   [{values}]")
    return "\n".join(lines)


def _audit(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        raise ValueError(f"no such file: {path}")

    emitter = EventEmitter()
    events: list[Event] = []
    emitter.subscribe(events.append)
    if not args.json or args.out:
        # --json to stdout would be corrupted by the stream; with --out the
        # file takes the document and the terminal still shows the run.
        emitter.subscribe(print_event)

    findings = MODES[args.mode](path, Path(args.data), emitter, timeout_s=args.timeout)

    if args.json or args.out:
        payload = {
            "file": str(path),
            "data": str(args.data),
            "mode": args.mode,
            "findings": [finding_payload(f) for f in findings],
            "events": [
                {"type": e.type.value, "ts": e.ts, "payload": e.payload} for e in events
            ],
        }
        _write_json(payload, args.out)
    return 0


def _write_json(payload: dict[str, object], out: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if out is None:
        print(text)
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hindsight")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="score a detector against the frozen cases")
    ev.add_argument("--suite", default="all", choices=["all", "injected", "clean"])
    ev.add_argument("--case", default=None, help="run a single case by id")
    ev.add_argument("--detector", default="null", choices=sorted(DETECTORS))
    ev.add_argument(
        "--baseline",
        default=None,
        choices=["oneshot", "freqtrade"],
        help="score a baseline, not a detector (ignores --detector/--suite/--case)",
    )
    ev.add_argument("--json", action="store_true", help="emit machine-readable results")
    ev.add_argument("--out", default=None, help="write JSON to this path")
    ev.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the suite N times and report the spread; the agent is not "
        "deterministic and one pass is not a result. Each pass gets its own "
        "prompt cache, so a repeat costs real provider calls - sharing one "
        "cache would replay pass 1 and report a spread of zero",
    )
    ev.add_argument(
        "--no-metrics",
        action="store_true",
        help="skip the before/after runs that measure each leak's Sharpe delta",
    )
    ev.set_defaults(handler=_eval)

    au = sub.add_parser("audit", help="audit a strategy file for look-ahead bias")
    au.add_argument("path")
    au.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="CSV of daily bars the strategy is run against",
    )
    au.add_argument(
        "--mode",
        default="pipeline",
        choices=sorted(MODES),
        help="pipeline: one straight-line pass, the documented baseline; "
        "agent: the loop that retries, re-baselines, and keeps hunting",
    )
    au.add_argument("--timeout", type=float, default=60.0, help="per-run seconds")
    au.add_argument("--json", action="store_true", help="emit machine-readable results")
    au.add_argument("--out", default=None, help="write JSON to this path")
    au.set_defaults(handler=_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    # The event stream carries em-dashes and arrows, and a Windows console
    # defaults to a codepage that cannot encode them. Without this the reasoning
    # the agent emits arrives on screen corrupted.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
