"""Subscribes to core events and prints them. The only place print() belongs."""

from __future__ import annotations

import textwrap

from hindsight_core.models import Event, EventType

_STATUS_LABEL = {
    "proven": "PROVEN",
    "no_effect": "no effect",
    "patch_broken": "PATCH BROKEN",
    "untestable": "UNTESTABLE",
    "patch_failed": "PATCH DID NOT APPLY",
    "not_mechanically_patchable": "DETECTED, NOT MECHANICALLY PATCHABLE",
    "no_operation": "NO OPERATION AVAILABLE",
}


def print_event(event: Event) -> None:
    for line in _render(event):
        print(line)


def _render(event: Event) -> list[str]:
    payload = event.payload
    if event.type is EventType.SCAN_COMPLETE:
        candidates = payload["candidates"]
        lines = [f"scan      {len(candidates)} candidate(s)"]
        lines += [
            f"          {c['leak_type']} line {c['line']}: {c['snippet']}"
            for c in candidates
        ]
        return lines

    if event.type is EventType.BASELINE:
        head = f"baseline  {payload['outcome']}"
        if payload["metrics"]:
            head += f"  {_metrics(payload['metrics'])}"
        return [head, f"          run_id {payload['run_id']}"]

    if event.type is EventType.TRIAGE:
        candidate = payload["candidate"]
        verdict = "leak" if payload["is_leak"] else "not a leak"
        operation = f" -> {payload['operation']}" if payload["is_leak"] else ""
        return [
            f"triage    {candidate['leak_type']} line {candidate['line']}: "
            f"{verdict}{operation}"
        ]

    if event.type is EventType.PROVE_START:
        candidate = payload["candidate"]
        return [
            f"prove     line {candidate['line']} via {payload['operation']}",
            *(f"          {ln}" for ln in str(payload["diff"]).splitlines()),
        ]

    if event.type is EventType.PROVE_RESULT:
        return _prove_result(payload)

    if event.type is EventType.AGENT_DECISION:
        return _decision(payload)

    if event.type is EventType.FINAL:
        return _final(payload)

    return [f"{event.type:<9} {payload}"]


def _decision(payload: dict[str, object]) -> list[str]:
    """What the agent decided and why, including the candidates it dropped.

    A discard is reasoning, not noise: the reader needs to see the line the
    agent looked at and declined, or "0 proven leaks" is indistinguishable from
    "never looked".
    """
    candidate = payload["candidate"]
    lines = [
        f"decide    step {payload['step']} {payload['action']} "
        f"{candidate['leak_type']} line {candidate['line']}",
        *_wrap(str(payload["reason"])),
        f"          [{payload['findings']} proven, {payload['pending']} pending, "
        f"{payload['llm_calls']} llm, {payload['sandbox_runs']} runs]",
    ]
    return lines


def _final(payload: dict[str, object]) -> list[str]:
    lines = ["", f"final     {payload.get('verdict', '')}".rstrip()]
    if payload.get("reason"):
        lines += _wrap(str(payload["reason"]))
    lines.append(f"          {len(payload['findings'])} proven leak(s)")
    for entry in payload.get("unproven", ()):
        candidate = entry["candidate"]
        status = _STATUS_LABEL.get(entry["status"], entry["status"])
        lines.append(
            f"          unproven: {candidate['leak_type']} line "
            f"{candidate['line']} — {status}"
        )
    return lines


def _wrap(text: str, width: int = 76) -> list[str]:
    """Reasons are sentences, not labels, and a terminal is not infinitely wide."""
    return [f"          {line}" for line in textwrap.wrap(text, width) or [""]]


def _prove_result(payload: dict[str, object]) -> list[str]:
    candidate = payload["candidate"]
    status = _STATUS_LABEL.get(str(payload["status"]), str(payload["status"]))
    head = f"result    line {candidate['line']}: {status}"
    if payload["status"] == "patch_failed":
        return [head, f"          {payload['error']}"]

    lines = [
        head,
        f"          before {_metrics(payload['before_metrics'])} "
        f"({payload['before_run_id']})",
        f"          after  {_metrics(payload['after_metrics'])} "
        f"({payload['after_run_id']})",
    ]
    if payload["delta"]:
        lines.append(f"          delta  {_metrics(payload['delta'], signed=True)}")
    if payload["status"] == "patch_broken" and payload.get("stderr"):
        lines += [
            f"          {ln}" for ln in str(payload["stderr"]).strip().splitlines()[-1:]
        ]
    return lines


def _metrics(metrics: dict[str, float], signed: bool = False) -> str:
    """Empty is not zero. A run with no metrics says so in words, because a
    printed 0.000 would read as a measured result."""
    if not metrics:
        return "no metrics"
    fmt = "+.3f" if signed else ".3f"
    return "  ".join(f"{name} {value:{fmt}}" for name, value in metrics.items())
