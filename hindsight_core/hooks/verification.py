"""Blocks any finding that lacks an execution record. Not a convention — a gate.

This is CLAUDE.md critical rule 1 written as code. `Finding.__post_init__`
already refuses an empty run ID; this hook refuses the subtler fabrication —
run IDs that are well-formed, plausible, and point at nothing.

It raises rather than filters. Silently dropping an unbacked finding would
leave a report that is indistinguishable from a clean file, which is the exact
failure mode this project exists to catch; and raising means every caller has
to route through it rather than around it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hindsight_core.models import Finding, RunRecord, SandboxOutcome


class VerificationError(RuntimeError):
    """A finding reached the gate without the execution behind it."""


def verify_findings(
    findings: Sequence[Finding], runs: Mapping[str, RunRecord]
) -> list[Finding]:
    for finding in findings:
        where = f"{finding.candidate.file}:{finding.candidate.line}"
        for role, run_id in (
            ("before", finding.before_run_id),
            ("after", finding.after_run_id),
        ):
            record = runs.get(run_id)
            if record is None:
                raise VerificationError(
                    f"finding at {where} cites {role} run {run_id!r}, which is not "
                    "in the run store"
                )
            # A crashed or zero-trade run is an execution record but not
            # evidence: there is no metric on it to have moved.
            if record.outcome is not SandboxOutcome.COMPLETED:
                raise VerificationError(
                    f"finding at {where} cites {role} run {run_id!r}, which "
                    f"{record.outcome} — a run that did not complete measures no "
                    "delta"
                )
    return list(findings)
