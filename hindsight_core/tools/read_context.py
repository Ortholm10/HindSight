"""Pull the surrounding source for a candidate leak site.

This is the main token cost per candidate, so the window is the *tighter* of the
radius and the enclosing function. A leak is an alignment error between adjacent
lines; the rest of the file rarely helps and always costs.
"""

from __future__ import annotations

import ast
from pathlib import Path

from hindsight_core.models import LeakCandidate


def read_context(path: Path, candidate: LeakCandidate, radius: int = 15) -> str:
    lines = path.read_text("utf-8").splitlines()
    if not 1 <= candidate.line <= len(lines):
        return ""

    start, end = candidate.line - radius, candidate.line + radius
    enclosing = _enclosing_block(lines, candidate.line)
    if enclosing is not None:
        start, end = max(start, enclosing[0]), min(end, enclosing[1])

    start, end = max(1, start), min(len(lines), end)
    width = len(str(end))
    return "\n".join(f"{n:>{width}} | {lines[n - 1]}" for n in range(start, end + 1))


def _enclosing_block(lines: list[str], line: int) -> tuple[int, int] | None:
    """The innermost def/class containing `line`, or None if the file is broken
    or the line sits at module level."""
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return None
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= line <= end:
            if best is None or (end - node.lineno) < (best[1] - best[0]):
                best = (node.lineno, end)
    return best
