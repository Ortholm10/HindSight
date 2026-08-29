"""The agent loop: state -> decide -> tool -> observe -> update.

Tools return data; this module is the only place that decides whether to
keep going.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_core.events import EventEmitter
from hindsight_core.models import Finding


def audit(
    path: Path, data_path: Path, emitter: EventEmitter, max_steps: int = 25
) -> list[Finding]:
    raise NotImplementedError
