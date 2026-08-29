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
from eval.harness import format_table, run_suite
from hindsight_cli.printer import print_event
from hindsight_core.events import EventEmitter
from hindsight_core.models import Event
from hindsight_core.pipeline import audit as pipeline_audit
from hindsight_core.pipeline import finding_payload

# The agent loop joins this table in Session 5; --mode is here from the start so
# the pipeline keeps its own name rather than becoming "the old default".
MODES = {"pipeline": pipeline_audit}

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

    result = run_suite(
        DETECTORS[args.detector],
        suite=args.suite,
        case_id=args.case,
        with_metrics=not args.no_metrics,
    )
    if args.json:
        payload = {
            "suite": result.suite,
            "detector": args.detector,
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
        text = json.dumps(payload, indent=2)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    else:
        print(format_table(result))
    return 0


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
        help="pipeline: one straight-line pass, the documented baseline",
    )
    au.add_argument("--timeout", type=float, default=60.0, help="per-run seconds")
    au.add_argument("--json", action="store_true", help="emit machine-readable results")
    au.add_argument("--out", default=None, help="write JSON to this path")
    au.set_defaults(handler=_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
