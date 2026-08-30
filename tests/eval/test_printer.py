"""Every event the core can emit has to render without raising.

The printer subscribes to the live stream, so a KeyError here does not print a
broken line — it kills the audit mid-run. That is how a missing optional field
took down a working agent once already, which is why this exists despite the
CLAUDE.md waiver on display tests.
"""

from __future__ import annotations

from hindsight_cli.printer import print_event
from hindsight_core.models import Event, EventType

CANDIDATE = {
    "leak_type": "L02",
    "file": "s.py",
    "line": 7,
    "snippet": "x.rolling(20, center=True)",
    "reason": "centered",
    "confidence": 0.9,
}

# Deliberately the *minimum* each producer emits, not the maximum. The agent's
# prove_start carries no diff because it has not chosen an operation that
# applies yet; the pipeline's prove_result carries no diff at all.
MINIMAL = [
    Event(EventType.SCAN_COMPLETE, {"file": "s.py", "candidates": [CANDIDATE]}),
    Event(
        EventType.BASELINE,
        {
            "run_id": "r1",
            "outcome": "completed",
            "metrics": {"sharpe": 1.0},
            "position_changes": 12,
        },
    ),
    Event(
        EventType.TRIAGE,
        {
            "candidate": CANDIDATE,
            "is_leak": True,
            "operation": "trailing_window",
            "answer": "LEAK: yes",
        },
    ),
    Event(
        EventType.PROVE_START,
        {"candidate": CANDIDATE, "operation": "trailing_window"},
    ),
    Event(
        EventType.PROVE_RESULT,
        {
            "candidate": CANDIDATE,
            "status": "proven",
            "operation": "trailing_window",
            "before_run_id": "r1",
            "after_run_id": "r2",
            "before_metrics": {"sharpe": 1.0},
            "after_metrics": {"sharpe": 0.2},
            "delta": {"sharpe": -0.8},
        },
    ),
    Event(
        EventType.PROVE_RESULT,
        {
            "candidate": CANDIDATE,
            "status": "patch_failed",
            "operation": "lag",
            "error": "nothing to transform",
        },
    ),
    Event(
        EventType.AGENT_DECISION,
        {
            "step": 3,
            "action": "discard",
            "candidate": CANDIDATE,
            "reason": "triage says this is not a leak",
            "sharpe": 1.0,
            "findings": 0,
            "pending": 2,
            "llm_calls": 3,
            "sandbox_runs": 4,
        },
    ),
    Event(
        EventType.FINAL,
        {
            "verdict": "inconclusive",
            "reason": "one candidate could not be settled either way",
            "findings": [],
            "unproven": [{"candidate": CANDIDATE, "status": "untestable"}],
        },
    ),
]


def test_every_event_renders_without_raising(capsys):
    for event in MINIMAL:
        print_event(event)
    out = capsys.readouterr().out
    assert "scan" in out
    assert "decide    step 3 discard" in out
    assert "final     inconclusive" in out
    assert "UNTESTABLE" in out


def test_the_final_verdict_is_the_first_thing_on_the_final_line(capsys):
    print_event(MINIMAL[-1])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines[0].startswith("final     inconclusive")
