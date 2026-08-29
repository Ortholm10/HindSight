"""The straight-line pipeline, run against a real frozen case and a real sandbox.

Only the LLM is stubbed. Stubbing the sandbox too would leave the one claim
this module makes — that the delta was measured, not asserted — untested.
"""

from pathlib import Path

import pytest

from hindsight_core import pipeline
from hindsight_core.events import EventEmitter
from hindsight_core.models import Event, EventType, RunRecord, SandboxOutcome

REPO = Path(__file__).resolve().parents[2]
LEAKED_CASE = REPO / "eval" / "cases" / "l01_future_index" / "strategy.py"
SPY = REPO / "eval" / "data" / "SPY.csv"


@pytest.fixture
def recorder():
    emitter = EventEmitter()
    events: list[Event] = []
    emitter.subscribe(events.append)
    return emitter, events


def _answering(text: str):
    def stub(prompt: str, **kwargs: object) -> str:
        return text

    return stub


def types(events: list[Event]) -> list[EventType]:
    return [e.type for e in events]


def test_proven_leak_carries_both_run_ids_and_a_measured_delta(
    tmp_path, monkeypatch, recorder
):
    emitter, events = recorder
    monkeypatch.setattr(
        pipeline, "complete", _answering("LEAK: yes\nOPERATION: future_shift")
    )

    findings = pipeline.audit(LEAKED_CASE, SPY, emitter, store=tmp_path)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.candidate.line == 8
    assert finding.before_run_id and finding.after_run_id
    assert finding.before_run_id != finding.after_run_id
    # The leak inflated Sharpe, so removing it must lower it. This is measured
    # by two executions, not asserted by the model that flagged the line.
    assert finding.delta["sharpe"] < 0
    assert "shift" in finding.diff
    assert types(events) == [
        EventType.SCAN_COMPLETE,
        EventType.BASELINE,
        EventType.TRIAGE,
        EventType.PROVE_START,
        EventType.PROVE_RESULT,
        EventType.FINAL,
    ]


def test_triage_rejection_never_runs_a_backtest(tmp_path, monkeypatch, recorder):
    emitter, events = recorder
    monkeypatch.setattr(pipeline, "complete", _answering("LEAK: no\nOPERATION: none"))

    findings = pipeline.audit(LEAKED_CASE, SPY, emitter, store=tmp_path)

    assert findings == []
    assert EventType.PROVE_START not in types(events)


def test_a_patch_that_cannot_apply_is_reported_not_raised(
    tmp_path, monkeypatch, recorder
):
    emitter, events = recorder
    # forward_fill has nothing to transform on a shift(-1) line.
    monkeypatch.setattr(
        pipeline, "complete", _answering("LEAK: yes\nOPERATION: forward_fill")
    )

    findings = pipeline.audit(LEAKED_CASE, SPY, emitter, store=tmp_path)

    assert findings == []
    results = [e for e in events if e.type is EventType.PROVE_RESULT]
    assert [e.payload["status"] for e in results] == ["patch_failed"]


def test_an_unparseable_answer_is_treated_as_no_leak(tmp_path, monkeypatch, recorder):
    emitter, events = recorder
    monkeypatch.setattr(pipeline, "complete", _answering("I am not sure, sorry."))

    assert pipeline.audit(LEAKED_CASE, SPY, emitter, store=tmp_path) == []
    assert EventType.PROVE_START not in types(events)


def test_an_untestable_baseline_stops_the_audit(tmp_path, monkeypatch, recorder):
    """A strategy that never trades has not been proven clean — it has been
    proven unmeasurable, and nothing downstream may report a delta."""
    emitter, events = recorder
    never_trades = tmp_path / "flat.py"
    never_trades.write_text(
        "import pandas as pd\n\n\n"
        "def run_positions(df):\n"
        '    signal = df["close"].shift(-1) > 1e12\n'
        "    return signal.fillna(False).astype(int)\n\n\n"
        "def run_strategy(df):\n"
        '    return (1 + (run_positions(df) * df["close"].pct_change())'
        ".fillna(0)).cumprod()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline, "complete", _answering("LEAK: yes\nOPERATION: future_shift")
    )

    findings = pipeline.audit(never_trades, SPY, emitter, store=tmp_path)

    assert findings == []
    final = [e for e in events if e.type is EventType.FINAL][-1]
    assert final.payload["baseline_outcome"] == SandboxOutcome.ZERO_TRADES
    assert EventType.TRIAGE not in types(events)


def _record(outcome: SandboxOutcome, sharpe: float | None = None) -> RunRecord:
    metrics = {} if sharpe is None else {"sharpe": sharpe}
    return RunRecord(run_id="r" + outcome.value[:4], outcome=outcome, metrics=metrics)


def test_classify_keeps_the_three_failure_modes_apart():
    completed = _record(SandboxOutcome.COMPLETED, 4.0)

    assert (
        pipeline.classify(
            completed, _record(SandboxOutcome.COMPLETED, 0.3), {"sharpe": -3.7}
        )
        == "proven"
    )
    assert (
        pipeline.classify(completed, _record(SandboxOutcome.CRASHED), {})
        == "patch_broken"
    )
    assert (
        pipeline.classify(completed, _record(SandboxOutcome.TIMED_OUT), {})
        == "patch_broken"
    )
    assert (
        pipeline.classify(completed, _record(SandboxOutcome.ZERO_TRADES), {})
        == "untestable"
    )
    assert (
        pipeline.classify(
            completed, _record(SandboxOutcome.COMPLETED, 4.1), {"sharpe": 0.1}
        )
        == "no_effect"
    )
