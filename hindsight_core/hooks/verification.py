"""Blocks any finding that lacks an execution record. Not a convention — a gate."""

from __future__ import annotations

from collections.abc import Sequence

from hindsight_core.models import Finding, RunRecord


def verify_findings(
    findings: Sequence[Finding], runs: dict[str, RunRecord]
) -> list[Finding]:
    raise NotImplementedError
