"""Shared stubs for the core tests.

The `offline` fixture lives here rather than in one test module because two
files need it: the loop tests and the memory tests. Both are about Hindsight's
own decisions, and neither should depend on a free-tier model answering today.
"""

from __future__ import annotations

import pytest

from hindsight_core import orchestrator
from hindsight_core.provers import differential

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


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Every core test gets an empty leak-signature store.

    Without this the suite is order-dependent and, worse, self-confirming: an
    audit run by one test teaches the shared store a signature, and the next
    test's audit then recognises it and skips the triage it was written to
    exercise. Tests that want a warm store build one under tmp_path and pass
    memory_path explicitly.
    """
    # In its own subdirectory: several tests count the files an audit wrote
    # into tmp_path, and a store dropped beside them would be miscounted.
    monkeypatch.setattr(
        orchestrator, "MEMORY_PATH", tmp_path / "_memory" / "memory.json"
    )


@pytest.fixture
def stub_triage():
    """The same stub `offline` installs, for tests that also need to patch the
    pipeline's copy of triage so the two paths answer identically."""
    return _stub_triage
