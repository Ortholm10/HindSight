"""AST scan of a strategy file into candidate leak sites.

A candidate generator, not a detector. It is tuned for recall and deliberately
over-produces: a false candidate costs one prover run, a missed one costs the
whole finding. Confidence is an ordering hint for triage, never a filter.
"""

from __future__ import annotations

import ast
from pathlib import Path

from hindsight_core.models import LeakCandidate

# Statistics that consume every row they are given. Legitimate when the caller
# already restricted the window; a leak when applied to the full sample.
_FULL_SAMPLE_STATS = {
    "mean",
    "std",
    "var",
    "min",
    "max",
    "sum",
    "median",
    "quantile",
    "corr",
    "cov",
}
# ...which is why a stat reached through one of these is not a candidate.
_WINDOWED = {"rolling", "expanding", "ewm", "groupby", "resample", "shift"}
_EXTREMA = {"idxmax", "idxmin", "argmax", "argmin"}
_FIT = {"fit", "fit_transform"}
# Names whose value is the decision itself: if it was not lagged, the decision
# used the bar it is acting on. This is L03, the canonical case.
_DECISION_NAMES = ("position", "signal", "entry", "exit", "trade")


class _Scan(ast.NodeVisitor):
    def __init__(self, source: str, path: Path) -> None:
        self.lines = source.splitlines()
        self.path = str(path)
        self.found: dict[tuple[str, int], LeakCandidate] = {}

    def _add(self, leak_type: str, line: int, reason: str, confidence: float) -> None:
        key = (leak_type, line)
        if key not in self.found:
            self.found[key] = LeakCandidate(
                leak_type=leak_type,
                file=self.path,
                line=line,
                snippet=self.lines[line - 1].strip() if line <= len(self.lines) else "",
                reason=reason,
                confidence=confidence,
            )

    def visit_Call(self, node: ast.Call) -> None:
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else (node.func.id if isinstance(node.func, ast.Name) else "")
        )
        keywords = {kw.arg: kw.value for kw in node.keywords}
        line = node.lineno

        if name == "shift" and _is_negative(node.args[0] if node.args else None):
            self._add(
                "L01", line, "shift() with a negative period reads the future", 0.9
            )

        if name == "rolling" and _is_true(keywords.get("center")):
            self._add("L02", line, "rolling(center=True) spans bars after the row", 0.9)

        if name in _FULL_SAMPLE_STATS and not _reached_through_window(node.func):
            self._add(
                "L05",
                line,
                f"{name}() over the full sample includes rows after each decision",
                0.5,
            )

        if name == "bfill" or _is_str(keywords.get("method"), "bfill", "backfill"):
            self._add("L06", line, "backward fill carries a later value backwards", 0.8)

        if name == "resample" and not ({"label", "closed"} & keywords.keys()):
            self._add(
                "L07",
                line,
                "resample() without label=/closed= leaves the bar stamp ambiguous",
                0.6,
            )

        if name in _EXTREMA:
            self._add(
                "L09", line, f"{name}() is two-sided — it sees both directions", 0.6
            )

        if name == "train_test_split" and not _is_false(keywords.get("shuffle")):
            self._add("L10", line, "a shuffled split mixes later rows into train", 0.8)

        if name in _FIT:
            self._add("L11", line, f"{name}() fitted here may see the test fold", 0.4)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr in ("iloc", "iat")
            and _has_forward_offset(node.slice)
        ):
            self._add(
                "L01", node.lineno, "positional index offset into the future", 0.8
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [t.id.lower() for t in node.targets if isinstance(t, ast.Name)]
        if any(k in name for name in targets for k in _DECISION_NAMES):
            if not _contains_shift(node.value):
                self._add(
                    "L03",
                    node.lineno,
                    "decision assigned without a lag — it may act on its own bar",
                    0.5,
                )
        self.generic_visit(node)


def _is_negative(node: ast.expr | None) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return True
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and node.value < 0
    )


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_false(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _is_str(node: ast.expr | None, *values: str) -> bool:
    return isinstance(node, ast.Constant) and node.value in values


def _reached_through_window(func: ast.expr) -> bool:
    """True for `x.rolling(20).mean()` — the window already bounds the stat."""
    if not isinstance(func, ast.Attribute):
        return False
    inner = func.value
    return (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr in _WINDOWED
    )


def _has_forward_offset(node: ast.expr) -> bool:
    return any(
        isinstance(child, ast.BinOp) and isinstance(child.op, ast.Add)
        for child in ast.walk(node)
    )


def _contains_shift(node: ast.expr) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr in ("shift", "lag")
        for child in ast.walk(node)
    )


def scan_file(path: Path) -> list[LeakCandidate]:
    """Candidates ordered by line. An unparseable file yields none rather than
    raising: the agent loop must survive input that does not compile."""
    source = path.read_text("utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    scan = _Scan(source, path)
    scan.visit(tree)
    return sorted(scan.found.values(), key=lambda c: (c.line, c.leak_type))
