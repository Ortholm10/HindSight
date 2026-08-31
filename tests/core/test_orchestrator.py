"""The loop.

Four behaviours are the point of this file and each one has its own test:
multi-step repair, continued suspicion after a fix lands, inconclusive outcomes
that are never rounded up to "clean", and a ceiling that is its own verdict.

The LLM is stubbed throughout. That is deliberate: what is under test is the
loop's own decisions, and a test that depends on a free-tier model answering
today would prove nothing tomorrow. Everything below the model — the patches,
the sandbox, the numbers — is real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.cache_data import DATA_DIR
from hindsight_core import orchestrator, pipeline
from hindsight_core.events import EventEmitter
from hindsight_core.models import (
    Budget,
    Event,
    EventType,
    ProofAttempt,
    ProofResult,
)
from hindsight_core.provers import differential
from hindsight_core.tools.run_backtest import RUNS_DIR

DATA = DATA_DIR / "SPY.csv"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CASES = Path(__file__).resolve().parents[2] / "eval" / "cases"

# The operation the taxonomy names first for each type — what a competent
# triage answer looks like, and for L05 deliberately the one that does not
# work, because that is the case the retry ladder exists for.
OBVIOUS = {
    "L01": "future_shift",
    "L02": "trailing_window",
    "L03": "lag",
    "L04": "lag",
    "L05": "expanding_stat",
    "L06": "forward_fill",
    "L07": "resample_label",
    "L09": "trailing_window",
    "L10": "chronological_split",
    "L11": "fit_in_fold",
}


def _stub_triage(path, candidate):
    return True, OBVIOUS.get(candidate.leak_type, ""), f"stub: {candidate.leak_type}"


@pytest.fixture
def offline(monkeypatch):
    """Triage calls every candidate a leak, retries fall through to the
    mechanical sweep, and the judge always says keep looking. No network."""
    monkeypatch.setattr(orchestrator, "triage", _stub_triage)
    monkeypatch.setattr(differential, "_next_operation", lambda *a, **k: "")
    monkeypatch.setattr(
        orchestrator, "_judge_plausible", lambda *a, **k: (False, "stubbed judge")
    )


def _collect(emitter: EventEmitter) -> list[Event]:
    events: list[Event] = []
    emitter.subscribe(events.append)
    return events


def _of(events: list[Event], kind: EventType) -> list[Event]:
    return [e for e in events if e.type is kind]


# --- continued suspicion -----------------------------------------------------


def test_a_leak_the_single_pass_discards_is_proven_after_the_first_fix(
    offline, monkeypatch, tmp_path
):
    """The L05 cutoff is a real leak whose removal *raises* Sharpe by 0.62
    while the centered window is still there. Judged once against the original
    baseline it is no-effect; judged again against the repaired one it deflates.
    Same file, same triage, same prover — the loop is the whole difference."""
    monkeypatch.setattr(pipeline, "triage", _stub_triage)
    path = FIXTURES / "stacked_leaks.py"

    agent = orchestrator.audit(path, DATA, EventEmitter(), trajectory_dir=tmp_path)
    straight_line = pipeline.audit(path, DATA, EventEmitter())

    assert "L05" in [f.candidate.leak_type for f in agent]
    assert "L05" not in [f.candidate.leak_type for f in straight_line]


def test_the_baseline_falls_step_by_step_as_fixes_are_kept(offline, tmp_path):
    """The chain the pipeline cannot produce. Each repair is measured against
    the strategy as the previous repair left it, so the numbers descend instead
    of each finding quoting the same untouched starting point."""
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        FIXTURES / "stacked_leaks.py", DATA, emitter, trajectory_dir=tmp_path
    )

    sharpes = [e.payload["metrics"]["sharpe"] for e in _of(events, EventType.BASELINE)]
    assert len(sharpes) >= 3
    assert sharpes == sorted(sharpes, reverse=True), sharpes
    assert sharpes[0] > 1.5 and sharpes[-1] < 0.6


def test_the_reopened_finding_is_measured_against_the_repaired_baseline(
    offline, tmp_path
):
    """The mechanical proof that this is not a single pass: the re-opened
    finding cites a before-run that did not exist when the audit started."""
    emitter = EventEmitter()
    events = _collect(emitter)

    findings = orchestrator.audit(
        FIXTURES / "stacked_leaks.py", DATA, emitter, trajectory_dir=tmp_path
    )

    first_baseline = _of(events, EventType.BASELINE)[0].payload["run_id"]
    reopened = next(f for f in findings if f.candidate.leak_type == "L05")
    assert reopened.before_run_id != first_baseline
    assert len(_of(events, EventType.BASELINE)) >= 2


def test_accepting_a_fix_reports_the_numbers_it_moved(offline, tmp_path):
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        FIXTURES / "stacked_leaks.py", DATA, emitter, trajectory_dir=tmp_path
    )

    accepts = [
        e
        for e in _of(events, EventType.AGENT_DECISION)
        if e.payload["action"] == "accept"
    ]
    assert accepts
    assert "->" in accepts[0].payload["reason"]
    assert "Re-opened" in accepts[0].payload["reason"]
    assert accepts[0].payload["sharpe"] < 1.886


def test_a_no_effect_verdict_is_reopened_when_the_baseline_it_was_measured_on_goes(
    offline, tmp_path
):
    """The requeue, driven directly.

    On the SPY cache the masked candidate happens to sit on a later line than
    the leak masking it, so the loop reaches it after the repair anyway and the
    re-opening branch never fires. It is still the branch that makes the rule
    general — "no effect" is a claim about a baseline, and this asserts the
    claim is withdrawn when that baseline is replaced.
    """
    # _accept writes the repaired source over its working copy, so it gets a
    # scratch copy here for the same reason the audit gives itself one.
    working = tmp_path / "stacked_leaks.py"
    working.write_bytes((FIXTURES / "stacked_leaks.py").read_bytes())
    emitter = EventEmitter()
    events = _collect(emitter)

    findings = orchestrator.audit(working, DATA, emitter, trajectory_dir=tmp_path)
    proven = findings[0]

    state = orchestrator._State(
        source_path=working,
        baseline=orchestrator.run_backtest(working, DATA),
        queue=[
            orchestrator._Queued(proven.candidate, verdict="no_effect"),
            orchestrator._Queued(findings[-1].candidate, verdict="discarded"),
        ],
    )
    stale = ProofResult(candidate=proven.candidate, status="no_effect", attempts=())
    state.unproven.append(stale)

    orchestrator._accept(
        state,
        ProofResult(
            candidate=proven.candidate,
            status="proven",
            attempts=(
                ProofAttempt(
                    operation="trailing_window", status="proven", applied=True
                ),
            ),
            finding=proven,
            patched_source=working.read_text("utf-8"),
        ),
        working,
        DATA,
        emitter,
        Budget(),
        60.0,
        RUNS_DIR,
    )

    assert state.queue[0].verdict == "", "the stale no-effect verdict must be dropped"
    assert state.queue[1].verdict == "discarded", (
        "a triage discard is a judgement about the code, not about a baseline"
    )
    assert stale not in state.unproven
    accept = [
        e
        for e in _of(events, EventType.AGENT_DECISION)
        if e.payload["action"] == "accept"
    ][-1]
    assert accept.payload["reopened"], accept.payload["reason"]


# --- inconclusive outcomes ---------------------------------------------------


def test_zero_trades_after_patching_is_untestable_never_clean(offline, tmp_path):
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        FIXTURES / "zero_trades_after_patch.py", DATA, emitter, trajectory_dir=tmp_path
    )

    statuses = {e.payload["status"] for e in _of(events, EventType.PROVE_RESULT)}
    assert "untestable" in statuses
    final = _of(events, EventType.FINAL)[-1].payload
    assert final["verdict"] != "clean"
    assert any(u["status"] == "untestable" for u in final["unproven"])


def test_a_crashing_patch_is_reported_as_broken_not_as_a_clean_run(offline, tmp_path):
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        FIXTURES / "crash_on_patch.py", DATA, emitter, trajectory_dir=tmp_path
    )

    statuses = [e.payload["status"] for e in _of(events, EventType.PROVE_RESULT)]
    assert "patch_broken" in statuses
    final = _of(events, EventType.FINAL)[-1].payload
    assert any(u["status"] == "patch_broken" for u in final["unproven"])


def test_a_detected_but_unpatchable_leak_is_reported_as_a_boundary(
    offline, monkeypatch, tmp_path
):
    """l08 line 9, with triage reading the file the way a person would: the
    forward column on line 7 is a diagnostic, and the decision on line 9 is the
    leak. No operation in the vocabulary expresses that repair — it needs a
    substitute column, which is a judgement — so the boundary is reported
    rather than forced through with an invented value."""
    monkeypatch.setattr(
        orchestrator,
        "triage",
        lambda path, candidate: (
            (True, "lag", "stub: the decision reads a forward column")
            if candidate.line == 9
            else (False, "", "stub: diagnostic only, never reaches the decision")
        ),
    )
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        CASES / "l08_forward_target" / "strategy.py",
        DATA,
        emitter,
        trajectory_dir=tmp_path,
    )

    final = _of(events, EventType.FINAL)[-1].payload
    boundary = [
        u for u in final["unproven"] if u["status"] == "not_mechanically_patchable"
    ]
    assert boundary, final["unproven"]
    assert boundary[0]["candidate"]["line"] == 9
    assert "substitute" in boundary[0]["reason"]


def test_a_baseline_that_cannot_run_is_untestable_and_still_leaves_a_trajectory(
    offline, tmp_path
):
    broken = tmp_path / "broken.py"
    broken.write_text(
        "def run_positions(df):\n"
        "    raise ValueError('nope')\n"
        "\n"
        "\n"
        "def run_strategy(df):\n"
        "    raise ValueError('nope')\n",
        "utf-8",
    )
    emitter = EventEmitter()
    events = _collect(emitter)

    findings = orchestrator.audit(broken, DATA, emitter, trajectory_dir=tmp_path)

    assert findings == []
    final = _of(events, EventType.FINAL)[-1].payload
    assert final["verdict"] == "untestable"
    assert list(tmp_path.glob("*.json"))


# --- the budget ceiling ------------------------------------------------------


def test_hitting_the_cap_is_its_own_verdict_and_is_never_clean(offline, tmp_path):
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        CASES / "l04_htf_merge" / "strategy.py",
        DATA,
        emitter,
        budget=Budget(max_llm_calls=1, max_sandbox_runs=2),
        trajectory_dir=tmp_path,
    )

    final = _of(events, EventType.FINAL)[-1].payload
    assert final["verdict"] == "stopped_on_budget"
    assert "cap" in final["reason"]


def test_a_full_budget_leaves_a_normal_verdict(offline, tmp_path):
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        CASES / "l02_centered_window" / "strategy.py",
        DATA,
        emitter,
        trajectory_dir=tmp_path,
    )

    assert _of(events, EventType.FINAL)[-1].payload["verdict"] == "leaks_proven"


# --- the event stream --------------------------------------------------------


def test_the_agent_emits_its_reasoning_including_what_it_discarded(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        orchestrator,
        "triage",
        lambda path, candidate: (False, "", "stub: this line is already lagged"),
    )
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        CASES / "l02_centered_window" / "strategy.py",
        DATA,
        emitter,
        trajectory_dir=tmp_path,
    )

    discards = [
        e
        for e in _of(events, EventType.AGENT_DECISION)
        if e.payload["action"] == "discard"
    ]
    assert discards
    assert "already lagged" in discards[0].payload["reason"]
    assert discards[0].payload["candidate"]["line"]
    assert _of(events, EventType.FINAL)[-1].payload["verdict"] == "clean"


def test_the_full_event_schema_is_emitted_in_order(offline, tmp_path):
    emitter = EventEmitter()
    events = _collect(emitter)

    orchestrator.audit(
        CASES / "l02_centered_window" / "strategy.py",
        DATA,
        emitter,
        trajectory_dir=tmp_path,
    )

    kinds = [e.type for e in events]
    assert kinds[0] is EventType.SCAN_COMPLETE
    assert kinds[1] is EventType.BASELINE
    assert kinds[-1] is EventType.FINAL
    for required in (
        EventType.TRIAGE,
        EventType.PROVE_START,
        EventType.PROVE_RESULT,
        EventType.AGENT_DECISION,
    ):
        assert required in kinds


# --- the trajectory ----------------------------------------------------------


def test_every_audit_writes_a_replayable_trajectory(offline, tmp_path):
    orchestrator.audit(
        CASES / "l02_centered_window" / "strategy.py",
        DATA,
        EventEmitter(),
        trajectory_dir=tmp_path,
    )

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    trajectory = json.loads(written[0].read_text("utf-8"))
    assert trajectory["file"].endswith("strategy.py")
    assert trajectory["verdict"] == "leaks_proven"
    assert trajectory["events"][0]["type"] == "scan_complete"
    assert trajectory["budget"]["sandbox_runs"] >= 2
    assert trajectory["findings"][0]["before_run_id"]


def test_candidates_are_reported_against_the_users_file_not_the_working_copy(
    offline, tmp_path
):
    """The audit works on a copy so it never edits its own evidence. A report
    citing that temp directory would be unusable."""
    path = CASES / "l02_centered_window" / "strategy.py"

    findings = orchestrator.audit(path, DATA, EventEmitter(), trajectory_dir=tmp_path)

    assert findings
    assert Path(findings[0].candidate.file) == path


def test_the_audited_file_is_never_modified(offline, tmp_path):
    path = FIXTURES / "stacked_leaks.py"
    before = path.read_bytes()

    orchestrator.audit(path, DATA, EventEmitter(), trajectory_dir=tmp_path)

    assert path.read_bytes() == before
