"""Subscribes to core events and prints them. The only place print() belongs."""

from __future__ import annotations

from hindsight_core.models import Event, EventType

_STATUS_LABEL = {
    "proven": "PROVEN",
    "no_effect": "no effect",
    "patch_broken": "PATCH BROKEN",
    "untestable": "UNTESTABLE",
    "patch_failed": "PATCH DID NOT APPLY",
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

    if event.type is EventType.FINAL:
        if payload.get("reason"):
            return ["", f"final     {payload['reason']}"]
        return ["", f"final     {len(payload['findings'])} proven leak(s)"]

    return [f"{event.type:<9} {payload}"]


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
