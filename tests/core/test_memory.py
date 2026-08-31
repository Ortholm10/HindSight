"""The leak-signature store, and the one property that matters about it.

Memory is a shortcut through TRIAGE and nothing else. It exists so the agent
does not spend an LLM call re-deciding whether a shape it has already put
through the sandbox is worth investigating. It must never become a shortcut
through PROOF: a signature the store recognises is still a suspicion, and the
only thing that has ever turned a suspicion into a Finding in this codebase is
a before/after pair of completed runs.

The last two tests are the ones that enforce that, and they are why this file
exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.cache_data import DATA_DIR
from hindsight_core import memory, orchestrator
from hindsight_core.events import EventEmitter
from hindsight_core.hooks.verification import VerificationError, verify_findings
from hindsight_core.models import Event, EventType, Finding, LeakCandidate
from hindsight_core.tools.check_memory import check_memory

DATA = DATA_DIR / "SPY.csv"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

SNIPPET = "w_up = (weekly > weekly_sma).astype(float)"


def candidate(leak_type: str = "L04", file: str = "a.py", line: int = 10):
    return LeakCandidate(leak_type, file, line, SNIPPET, "carries a resampled series")


def finding_for(cand: LeakCandidate, before: str = "b1", after: str = "a1") -> Finding:
    return Finding(cand, before, after, "--- a\n+++ b", {"sharpe": -1.2})


# --- the store ---------------------------------------------------------------


def test_signature_ignores_file_and_line_but_not_leak_type():
    """A signature is about a SHAPE. The same bug copied into another file, or
    drifting down a few lines, is the same bug; the same text flagged under a
    different taxonomy entry is not."""
    here = candidate()
    elsewhere = candidate(file="b.py", line=99)
    other_type = candidate(leak_type="L03")

    assert memory.signature(here) == memory.signature(elsewhere)
    assert memory.signature(here) != memory.signature(other_type)


def test_signature_ignores_whitespace_but_not_the_code():
    assert memory.signature(candidate()) == memory.signature(
        LeakCandidate("L04", "a.py", 10, f"   {SNIPPET}  ", "r")
    )
    assert memory.signature(candidate()) != memory.signature(
        LeakCandidate("L04", "a.py", 10, SNIPPET.replace("weekly", "monthly"), "r")
    )


def test_load_returns_empty_for_a_missing_store(tmp_path):
    """A first run has no memory, and that is not an error."""
    assert memory.load(tmp_path / "nope.json") == []


def test_load_returns_empty_for_a_corrupt_store(tmp_path):
    """A half-written store must not take an audit down with it. Memory is an
    optimisation; losing it costs an LLM call, not a result."""
    store = tmp_path / "memory.json"
    store.write_text("{not json", encoding="utf-8")
    assert memory.load(store) == []


def test_record_then_lookup_finds_the_signature(tmp_path):
    store = tmp_path / "memory.json"
    cand = candidate()

    memory.record(store, finding_for(cand), "lag")
    hit = memory.lookup(store, cand)

    assert hit is not None
    assert hit["leak_type"] == "L04"
    assert hit["operation"] == "lag"
    assert hit["confirmations"] == 1
    assert hit["proven_by"] == [["b1", "a1"]]


def test_lookup_misses_an_unseen_signature(tmp_path):
    store = tmp_path / "memory.json"
    memory.record(store, finding_for(candidate()), "lag")
    assert memory.lookup(store, candidate(leak_type="L06")) is None


def test_recording_the_same_signature_twice_updates_one_entry(tmp_path):
    """Confirmations accumulate; entries do not. Otherwise the store grows
    without bound and every re-audit of the same file doubles it."""
    store = tmp_path / "memory.json"
    cand = candidate()

    memory.record(store, finding_for(cand, "b1", "a1"), "lag")
    memory.record(store, finding_for(cand, "b2", "a2"), "lag")

    assert len(memory.load(store)) == 1
    hit = memory.lookup(store, cand)
    assert hit["confirmations"] == 2
    assert hit["proven_by"] == [["b1", "a1"], ["b2", "a2"]]


def test_a_recorded_entry_is_readable_json_on_disk(tmp_path):
    """The store is meant to be opened and edited by a human - CLAUDE.md keeps
    leak knowledge in editable text, not in a trained artifact."""
    store = tmp_path / "memory.json"
    memory.record(store, finding_for(candidate()), "lag")

    entries = json.loads(store.read_text("utf-8"))
    assert isinstance(entries, list)
    assert entries[0]["snippet"] == SNIPPET


def test_check_memory_reads_the_store_it_is_given(tmp_path):
    store = tmp_path / "memory.json"
    cand = candidate()
    assert check_memory(cand, store) is None

    memory.record(store, finding_for(cand), "lag")
    assert check_memory(cand, store)["operation"] == "lag"


# --- the property this store must never break --------------------------------


def test_a_memory_hit_still_has_to_be_proven_by_execution(offline, tmp_path):
    """The load-bearing test. A recognised signature skips the triage CALL and
    nothing else: the candidate still goes to the prover, still costs sandbox
    runs, and still reaches the report only with a completed before/after pair
    behind it.

    The failure mode this catches: a "fast path" that returns a Finding
    straight from a stored entry. That would be the exact fabrication
    hooks/verification.py exists to stop, and it would be invisible - a
    plausible finding carrying plausible run IDs that point at nothing.
    """
    store = tmp_path / "memory.json"
    path = FIXTURES / "stacked_leaks.py"

    # Seed the store from a real audit, then re-audit the same file so its
    # candidates arrive as memory hits rather than as triage calls.
    orchestrator.audit(
        path, DATA, EventEmitter(), trajectory_dir=tmp_path, memory_path=store
    )
    assert memory.load(store), "the first audit recorded nothing to remember"

    emitter = EventEmitter()
    events: list[Event] = []
    emitter.subscribe(events.append)
    findings = orchestrator.audit(
        path, DATA, emitter, trajectory_dir=tmp_path, memory_path=store
    )

    remembered = [
        e for e in events if e.type is EventType.TRIAGE and e.payload.get("from_memory")
    ]
    assert remembered, "no candidate was recognised from memory on the second pass"

    # 1. Recognition did not skip the sandbox: every remembered candidate was
    #    still handed to the prover.
    assert len(_of(events, EventType.PROVE_START)) >= len(remembered)
    assert _of(events, EventType.PROVE_RESULT)

    # 2. Everything reported still cites two runs that actually happened. The
    #    audit itself routes through verify_findings, which raises rather than
    #    filters, so reaching this line at all is half the assertion.
    assert findings
    for found in findings:
        assert found.before_run_id and found.after_run_id


def test_a_finding_built_straight_from_a_memory_hit_is_refused(tmp_path):
    """The other half, at the gate itself. Even holding a store entry whose run
    IDs were genuinely valid once, a finding assembled from memory alone does
    not pass - those runs are not in THIS audit's run store, so nothing here
    was measured."""
    store = tmp_path / "memory.json"
    cand = candidate()
    memory.record(store, finding_for(cand, "old-before", "old-after"), "lag")

    hit = check_memory(cand, store)
    before, after = hit["proven_by"][0]
    fabricated = Finding(cand, before, after, "diff", {"sharpe": -1.2})

    with pytest.raises(VerificationError, match="not in the run store"):
        verify_findings([fabricated], {})


def _of(events: list[Event], kind: EventType) -> list[Event]:
    return [e for e in events if e.type is kind]
