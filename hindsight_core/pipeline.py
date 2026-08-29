"""The straight-line pipeline: scan -> triage -> prove. No agent judgement."""

from __future__ import annotations

from pathlib import Path

from hindsight_core.events import EventEmitter
from hindsight_core.models import Finding


def audit(path: Path, data_path: Path, emitter: EventEmitter) -> list[Finding]:
    raise NotImplementedError
