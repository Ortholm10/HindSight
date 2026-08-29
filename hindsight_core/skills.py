"""Read the leak-type knowledge files in .claude/skills/ as plain text.

These are Hindsight's own runtime knowledge, not Claude Code plugin skills.
"""

from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "skills"


def load_skill(leak_type: str, skills_dir: Path = SKILLS_DIR) -> str:
    return (skills_dir / f"{leak_type.lower()}.md").read_text(encoding="utf-8")


def available_types(skills_dir: Path = SKILLS_DIR) -> list[str]:
    return sorted(p.stem.upper() for p in skills_dir.glob("l??.md"))
