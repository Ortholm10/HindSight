"""JSON store of confirmed leak signatures. Recognition, never proof.

What this buys: on a file whose shape the agent has put through the sandbox
before, triage costs nothing. `check_memory` answers "I have seen this exact
shape, and last time the repair that worked was X" without an LLM call.

What it must never buy, and does not: a finding. A hit skips exactly one
thing - the triage question - and the candidate goes to the prover on the same
path a fresh one takes. `record` is called only from the branch that has
already been through hooks/verification.py, so nothing enters this store that
was not itself execution-proven, and nothing leaves it as proof of anything.

Deliberately a flat JSON list, not a database: CLAUDE.md keeps leak knowledge
in editable text a human can open, diff, and delete a wrong row from.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from hindsight_core.models import Finding, LeakCandidate

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "runs" / "memory.json"


def signature(candidate: LeakCandidate) -> str:
    """A shape, not a location.

    File and line are excluded on purpose - the same bug pasted into another
    module is the same bug, and a signature that moved when a line moved would
    never hit twice. Whitespace is collapsed for the same reason; the leak type
    is included because identical text under a different taxonomy entry is a
    different claim about what is wrong with it.
    """
    normalised = " ".join(candidate.snippet.split())
    key = f"{candidate.leak_type}|{normalised}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def load(path: Path = DEFAULT_PATH) -> list[dict[str, object]]:
    """Every entry, or none. A missing or unreadable store is not an error.

    Memory is an optimisation. Losing it costs one LLM call per candidate,
    which is a far better outcome than taking down an audit because a previous
    process was killed mid-write.
    """
    try:
        raw = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw if isinstance(raw, list) else []


def lookup(path: Path, candidate: LeakCandidate) -> dict[str, object] | None:
    wanted = signature(candidate)
    return next((e for e in load(path) if e.get("signature") == wanted), None)


def record(path: Path, finding: Finding, operation: str) -> None:
    """Remember a leak that execution already proved.

    Called only after a finding has passed the verification gate, so the run
    IDs stored here are real. They are kept for provenance - so a human can ask
    which audit taught the agent this - and NOT as evidence: they belong to a
    finished audit and are meaningless to the next one, which is exactly what
    tests/core/test_memory.py asserts.
    """
    entries = load(path)
    key = signature(finding.candidate)
    now = datetime.now(UTC).isoformat()
    pair = [finding.before_run_id, finding.after_run_id]

    existing = next((e for e in entries if e.get("signature") == key), None)
    if existing is not None:
        existing["confirmations"] = int(existing.get("confirmations", 0)) + 1
        existing["last_seen"] = now
        existing["operation"] = operation
        proven = existing.setdefault("proven_by", [])
        if isinstance(proven, list) and pair not in proven:
            proven.append(pair)
    else:
        entries.append(
            {
                "signature": key,
                "leak_type": finding.candidate.leak_type,
                "operation": operation,
                "snippet": " ".join(finding.candidate.snippet.split()),
                "confirmations": 1,
                "proven_by": [pair],
                "first_seen": now,
                "last_seen": now,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
