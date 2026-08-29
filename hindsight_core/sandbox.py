"""Subprocess execution with a hard timeout and resource caps.

Audited code is untrusted and often broken. Every distinct failure maps to a
distinct SandboxOutcome — never one collapsed error path.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_core.models import RunRecord


def run_sandboxed(script: Path, data_path: Path, timeout_s: float = 60.0) -> RunRecord:
    raise NotImplementedError
