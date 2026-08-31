"""Score a baseline (oneshot or freqtrade) across the 20 frozen cases.

Not eval/harness.py's Detector contract: that contract assumes every case is
applicable and scores localisation by source line, neither of which holds for
freqtrade (some cases aren't expressible as a freqtrade strategy at all, and
lookahead-analysis never reports a line number). Cases marked not-applicable
are excluded from the denominator, never counted as a miss — see CLAUDE.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from eval.baselines import freqtrade, oneshot
from eval.runner import CASES_DIR, discover_cases

RESULTS_DIR = Path(__file__).resolve().parent / "results"

BaselineName = Literal["oneshot", "freqtrade"]

FREQTRADE_FIELD_NOTES = {
    "has_bias": (
        "Raw verdict from freqtrade's own lookahead-analysis output. NOT the "
        "primary detection signal (see 'detected') and currently unreliable: "
        "it fires True on nearly every applicable case (leaked and clean "
        "alike) due to a truncated-run timing artifact in freqtrade's own "
        "backtest order-fill bookkeeping (exact open_date/close_date "
        "timestamp matching across differently-truncated re-runs), not "
        "because the underlying signal is non-causal. Kept for transparency, "
        "not deleted — see eval/baselines/freqtrade.py module docstring for "
        "the full mechanism."
    ),
    "biased_indicators": (
        "Primary detection signal: freqtrade's column-value diff between the "
        "full-history run and each truncated re-run, at the compared rows. "
        "Non-empty means an indicator/signal column actually differed under "
        "truncation — a real causality violation, not execution-timing noise. "
        "'detected' = bool(biased_indicators) for applicable cases."
    ),
}


def _score_oneshot(meta, result: dict) -> dict:
    leak_found = result["leak_found"]
    if leak_found is None:
        # The call errored (rate limit, malformed provider response, etc.):
        # inconclusive, not a scored miss — never fake a "no leak" verdict.
        return {
            "case_id": meta.case_id,
            "kind": meta.kind,
            "leak_type": meta.leak_type,
            "applicable": True,
            "reason": f"oneshot call failed: {result['error']}",
            "detected": None,
            "localised": None,
            "line_reported": None,
            "error": result["error"],
        }
    localised = None
    if meta.is_injected:
        localised = result["line_reported"] == meta.ground_truth_line
    return {
        "case_id": meta.case_id,
        "kind": meta.kind,
        "leak_type": meta.leak_type,
        "applicable": True,
        "reason": "",
        "detected": bool(leak_found),
        "localised": localised,
        "line_reported": result["line_reported"],
        "error": result["error"],
    }


def _score_freqtrade(meta, result: dict) -> dict:
    applicable = result["applicable"]
    has_bias = result["has_bias"]
    biased_indicators = result.get("biased_indicators", [])
    # Primary signal is biased_indicators non-emptiness, not raw has_bias.
    # has_bias fires on nearly every applicable case (see field_notes in
    # build_results) — an execution-timing artifact of freqtrade's own
    # backtest order-fill bookkeeping under truncation, not a signal-level
    # leak. biased_indicators compares indicator/signal *column values*
    # directly and does not carry that noise.
    detected = bool(biased_indicators) if applicable and has_bias is not None else None
    return {
        "case_id": meta.case_id,
        "kind": meta.kind,
        "leak_type": meta.leak_type,
        "applicable": applicable,
        "reason": result["reason"],
        "detected": detected,
        "localised": None,  # freqtrade never reports a source line
        "biased_indicators": biased_indicators,
        "has_bias": has_bias,
        "error": result.get("error"),
    }


def build_results(name: BaselineName, cases_dir: Path = CASES_DIR) -> dict:
    # Frozen set only - see eval/harness.py::_select for why.
    cases = [c for c in discover_cases(cases_dir) if c.frozen]
    rows = []
    for meta in cases:
        if name == "oneshot":
            raw = oneshot.run(meta.path)
            rows.append(_score_oneshot(meta, raw))
        elif name == "freqtrade":
            raw = freqtrade.run(meta.path)
            rows.append(_score_freqtrade(meta, raw))
        else:
            raise ValueError(f"unknown baseline {name!r}")

    injected_applicable = [
        r for r in rows if r["kind"] == "injected" and r["applicable"]
    ]
    clean_applicable = [r for r in rows if r["kind"] == "clean" and r["applicable"]]
    # A row whose "detected" is None (inconclusive: too few trades, or a run
    # error) is excluded from both the hit and miss counts — it answered
    # neither way, so it cannot be scored as either.
    injected_scored = [r for r in injected_applicable if r["detected"] is not None]
    clean_scored = [r for r in clean_applicable if r["detected"] is not None]
    localised_rows = [r for r in injected_applicable if r["localised"] is not None]

    summary = {
        "detected": sum(1 for r in injected_scored if r["detected"]),
        "injected_total": len(injected_applicable),
        "injected_scored": len(injected_scored),
        "false_positives": sum(1 for r in clean_scored if r["detected"]),
        "clean_total": len(clean_applicable),
        "clean_scored": len(clean_scored),
        "localised": sum(1 for r in localised_rows if r["localised"]),
        "localised_total": len(localised_rows),
        "not_applicable": [r["case_id"] for r in rows if not r["applicable"]],
        "inconclusive": [
            r["case_id"] for r in rows if r["applicable"] and r["detected"] is None
        ],
    }
    result = {"baseline": name, "summary": summary, "cases": rows}
    if name == "freqtrade":
        result["field_notes"] = FREQTRADE_FIELD_NOTES
    return result


def format_table(result: dict) -> str:
    has_raw_bias = any("has_bias" in row for row in result["cases"])
    if has_raw_bias:
        header = (
            f"{'case':<32} {'kind':<9} {'applicable':>10} {'detected':>9} "
            f"{'has_bias(raw)':>14} {'notes'}"
        )
    else:
        header = (
            f"{'case':<32} {'kind':<9} {'applicable':>10} {'detected':>9} {'notes'}"
        )
    lines = [header, "-" * len(header)]
    for row in result["cases"]:
        raw_bias = str(row.get("has_bias", "")) if has_raw_bias else None
        if not row["applicable"]:
            notes = "N/A: " + row["reason"]
            detected = ""
        elif row["detected"] is None:
            notes = "inconclusive: " + (row["reason"] or row.get("error") or "")[:60]
            detected = ""
        else:
            detected = "yes" if row["detected"] else "no"
            notes = ""
            if row.get("localised") is not None:
                notes = "localised" if row["localised"] else "not localised"
            if row.get("biased_indicators"):
                notes = "cols: " + ",".join(row["biased_indicators"])
        if has_raw_bias:
            lines.append(
                f"{row['case_id']:<32} {row['kind']:<9} {str(row['applicable']):>10} "
                f"{detected:>9} {raw_bias:>14}  {notes}"
            )
        else:
            lines.append(
                f"{row['case_id']:<32} {row['kind']:<9} {str(row['applicable']):>10} "
                f"{detected:>9}  {notes}"
            )
    s = result["summary"]
    lines += [
        "-" * len(header),
        f"detected            {s['detected']}/{s['injected_scored']} "
        f"(of {s['injected_total']} applicable injected)",
        f"false positives     {s['false_positives']}/{s['clean_scored']} "
        f"(of {s['clean_total']} applicable clean)",
        f"line localised      {s['localised']}/{s['localised_total']}",
        f"not applicable      {', '.join(s['not_applicable']) or '(none)'}",
        f"inconclusive        {', '.join(s['inconclusive']) or '(none)'}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for baseline_name in sys.argv[1:] or ["oneshot", "freqtrade"]:
        result = build_results(baseline_name)  # type: ignore[arg-type]
        out = RESULTS_DIR / f"{baseline_name}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"== {baseline_name} ==")
        print(format_table(result))
        print(f"written to {out}\n")
