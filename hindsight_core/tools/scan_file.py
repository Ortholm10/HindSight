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
                "L01",
                _at(node.args[0], line),
                "shift() with a negative period reads the future",
                0.9,
            )

        if name == "rolling" and _is_true(keywords.get("center")):
            self._add(
                "L02",
                _at(keywords.get("center"), line),
                "rolling(center=True) spans bars after the row",
                0.9,
            )

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
            self._add(
                "L10",
                _at(keywords.get("shuffle"), line),
                "a shuffled split mixes later rows into train",
                0.8,
            )

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


def _at(node: ast.expr | None, fallback: int) -> int:
    """The line of the argument that carries the leak, not the call's first line.

    A multi-line call spans lines that are almost all innocent. Pointing a
    reader — or a patch — at `train_test_split(` says nothing; `shuffle=True`
    four lines below is the whole finding. Falls back to the call when no single
    argument is responsible, which is the case for leaks of ABSENCE: a bare
    `resample()` has no offending argument to name.
    """
    return fallback if node is None else node.lineno


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


def _reads_forward(node: ast.expr) -> bool:
    """True for an expression that reads a later row — the same two shapes the
    L01 rule fires on, asked of a whole assigned value rather than one call."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = child.func.attr if isinstance(child.func, ast.Attribute) else ""
            if name == "shift" and _is_negative(child.args[0] if child.args else None):
                return True
        if (
            isinstance(child, ast.Subscript)
            and isinstance(child.value, ast.Attribute)
            and child.value.attr in ("iloc", "iat")
            and _has_forward_offset(child.slice)
        ):
            return True
    return False


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _assigned_names(node: ast.Assign) -> set[str]:
    return {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}


def _absorbed_by_fit(statement: ast.stmt) -> set[str]:
    """Names this statement hands to a `.fit(...)` call as training data."""
    absorbed: set[str] = set()
    for node in ast.walk(statement):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in _FIT:
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                absorbed |= _loaded_names(argument)
    return absorbed


def _training_labels(tree: ast.Module) -> set[str]:
    """Forward-looking names whose every use ends up inside a `.fit()` call.

    A supervised label is forward-looking by definition — `y` for row t is what
    happened at t+1 — and that is not a leak. The scanner cannot tell a label
    from a leaky feature by looking at the expression, because the two are built
    identically; the whole difference is where the value goes afterwards. So
    this follows it: a value absorbed by `.fit()` is training data, and a value
    that reaches the decision by any other route is a feature.

    Reachability rather than a list of blessed function names, because the label
    rarely goes straight to `fit`. In l10 it passes through `train_test_split`
    first, and hard-coding that call would make this a rule about scikit-learn's
    API instead of a rule about where information flows.

    Why it is worth the machinery: without it the mechanical sweep reaches
    `future_shift` on the label, the fitted model collapses, and the falling
    Sharpe is indistinguishable from a removed leak. Measured on l10 that cost a
    false positive AND buried the real leak — a model trained on a corrupted
    label stops responding to `shuffle=True` at all.
    """
    statements = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.Expr | ast.Return)
    ]
    forward = {
        name
        for node in statements
        if isinstance(node, ast.Assign) and _reads_forward(node.value)
        for name in _assigned_names(node)
    }
    if not forward:
        return set()

    uses: dict[str, list[ast.stmt]] = {}
    for statement in statements:
        source = statement.value if isinstance(statement, ast.Assign) else statement
        if source is None:
            continue
        for name in _loaded_names(source):
            uses.setdefault(name, []).append(statement)

    # Least fixed point outward from the fit calls: a name is training data if
    # every use of it is, and forwarding through an assignment only counts once
    # the names it assigns are known to be training data themselves.
    labels: set[str] = set()
    while True:
        grown = {n for n in uses if n not in labels and _is_label(n, uses[n], labels)}
        if not grown:
            return forward & labels
        labels |= grown


def _is_label(name: str, statements: list[ast.stmt], labels: set[str]) -> bool:
    """Every use of `name` is training data, and at least one path reaches fit.

    A use forwards when the statement assigns it onward and the names it assigns
    are themselves labels. Targets never read again — the discarded halves of a
    `train_test_split` unpacking — carry nothing onward, so they cannot make the
    name a feature; but they cannot make it a label either, which is why at
    least one assigned name still has to be known training data.
    """
    reaches_fit = False
    for statement in statements:
        if name in _absorbed_by_fit(statement):
            reaches_fit = True
            continue
        if not isinstance(statement, ast.Assign):
            return False
        if not _assigned_names(statement) & labels:
            return False
        reaches_fit = True
    return reaches_fit


def _label_lines(tree: ast.Module, labels: set[str]) -> set[int]:
    """The lines that build a training label, and only those.

    Every line of the assignment, because a multi-line label expression puts the
    `shift(-1)` below the line the assignment opens on and the candidate is
    recorded against the argument.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _assigned_names(node) & labels:
            lines |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def _htf_chain(tree: ast.Module) -> tuple[dict[str, int], bool]:
    """Assignments carrying a resampled series, and whether any of them lags it.

    L04 cannot be judged one line at a time. The leak is that a higher-timeframe
    series reaches an as-of merge while its current bar is still open, and the
    repair is a single `.shift(1)` anywhere along the chain that builds it. So
    the chain is followed transitively from the `resample()` that starts it, and
    one lag found anywhere clears the whole thing - which is exactly the one
    token separating case l04 from its control.
    """
    assigns = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Assign)),
        key=lambda n: n.lineno,
    )
    carriers: dict[str, int] = {}
    lagged = False
    for node in assigns:
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        from_resample = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "resample"
            for c in ast.walk(node.value)
        )
        if not from_resample and not (names & carriers.keys()):
            continue
        # The walk stops at the join. Only a lag applied BEFORE the merge can
        # repair this leak; a shift further down the signal path acts on an
        # already-contaminated series, and counting it would clear the very
        # case this rule exists to catch.
        if _merges_as_of(node.value):
            break
        if _contains_shift(node.value):
            lagged = True
        for target in targets:
            carriers[target] = node.lineno
    return carriers, lagged


def _merges_as_of(tree: ast.AST) -> bool:
    """The join that makes an open higher-timeframe bar reachable at all.

    Gating on it keeps this rule off every file that merely resamples: without
    a merge back to the base frame there is no row t to contaminate.
    """
    return any(
        isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == "merge_asof")
            or (isinstance(n.func, ast.Name) and n.func.id == "merge_asof")
        )
        for n in ast.walk(tree)
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

    # Dropped after the visit rather than guarded inside it: the rule is about
    # a whole name's reachability, which is not knowable while walking one node.
    for line in _label_lines(tree, _training_labels(tree)):
        for key in [k for k in scan.found if k[1] == line]:
            del scan.found[key]

    carriers, lagged = _htf_chain(tree)
    if carriers and not lagged and _merges_as_of(tree):
        for name, line in carriers.items():
            scan._add(
                "L04",
                line,
                f"{name} carries a resampled series into an as-of merge with no "
                "lag, so row t can read a higher-timeframe bar still open",
                0.5,
            )

    return sorted(scan.found.values(), key=lambda c: (c.line, c.leak_type))
