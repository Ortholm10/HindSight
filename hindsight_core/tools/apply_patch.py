"""libcst transforms from the closed fix vocabulary. Returns a unified diff.

Every operation here *removes* information the strategy was not entitled to —
a lag, a narrowed window, a dropped direction. None of them invents a value,
which is CLAUDE.md rule 2 expressed as a closed vocabulary rather than a
review comment.

The function is pure. It never writes: the caller applies the patched source to
a sandbox copy, so a failed transform cannot leave a half-edited file behind.
"""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from hindsight_core.models import LeakCandidate, PatchResult

# The closed vocabulary, one entry per patchable row of docs/taxonomy.md 7.
# L12 has no entry on purpose: the taxonomy marks a hindsight universe
# detectable but not patchable, and inventing a repair for it would manufacture
# a proof the taxonomy says cannot exist.
OPERATIONS = (
    "future_shift",
    "lag",
    "trailing_window",
    "forward_fill",
    "expanding_stat",
    "rolling_stat",
    "resample_label",
    "drop_column",
    "fit_in_fold",
    "chronological_split",
)

_BFILL = ("bfill", "backfill")

# Statistics that consume every row handed to them. Legitimate once a window
# already bounds the input, a leak over the full sample.
_FULL_SAMPLE_STATS = (
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
)
_WINDOWED = ("rolling", "expanding", "ewm", "groupby", "resample", "shift")

# Taxonomy 7 offers two causal replacements for a full-sample statistic, and
# they are not interchangeable: expanding() keeps every row it has ever seen,
# rolling(n) forgets. Which one restores a given strategy is a question for the
# prover, so the vocabulary carries both rather than picking a favourite.
_STAT_WINDOWS = {
    "expanding_stat": "expanding(min_periods=20)",
    "rolling_stat": "rolling(40, min_periods=20)",
}

# The training fold is named, never computed here: `split` already exists in
# the audited code. Deriving our own boundary would be inventing a value, and
# a missing name surfaces honestly as a crashed run rather than a silent one.
_FOLD = ".iloc[:split]"

# Prefix on the one failure that is not "wrong tool" but "no tool exists".
# drop_column returns the same PatchResult shape either way, and the prover has
# to tell them apart: one says keep trying operations, the other says the
# closed vocabulary provably cannot express this repair. CLAUDE.md rule 2 is
# why the second is reported rather than forced.
BOUNDARY_MARKER = "vocabulary-boundary:"


class _Repair(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        line: int,
        operation: str,
        forward_names: frozenset[str] = frozenset(),
    ) -> None:
        self.line = line
        self.operation = operation
        self.forward_names = forward_names
        self.applied = False
        self.boundary = ""

    def _covers_line(self, node: cst.CSTNode) -> bool:
        """True when the candidate's line falls anywhere inside this node.

        Not just its first line. scan_file names the line of the argument that
        carries the leak, which for a multi-line call sits below the line the
        call opens on — so keying the transform to the opening line makes the
        repair unreachable for exactly the calls big enough to need one.
        """
        position = self.get_metadata(PositionProvider, node)
        return position.start.line <= self.line <= position.end.line

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.BaseExpression:
        if self.applied or not self._covers_line(original):
            return updated
        # Both call shapes matter: pandas repairs are methods (`x.resample`),
        # but train_test_split is imported and called bare.
        if isinstance(updated.func, cst.Attribute):
            name = updated.func.attr.value
        elif isinstance(updated.func, cst.Name):
            name = updated.func.value
        else:
            name = ""

        if self.operation == "future_shift" and name == "shift":
            args = list(updated.args)
            if args and isinstance(args[0].value, cst.UnaryOperation):
                if isinstance(args[0].value.operator, cst.Minus):
                    self.applied = True
                    args[0] = args[0].with_changes(value=args[0].value.expression)
                    return updated.with_changes(args=args)

        if self.operation == "trailing_window" and name == "rolling":
            kept = [a for a in updated.args if _keyword(a) != "center"]
            if len(kept) != len(updated.args):
                self.applied = True
                kept[-1] = kept[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
                return updated.with_changes(args=kept)

        if self.operation in _STAT_WINDOWS and name in _FULL_SAMPLE_STATS:
            if isinstance(updated.func, cst.Attribute) and not _windowed(updated.func):
                self.applied = True
                receiver = _code(updated.func.value)
                window = _STAT_WINDOWS[self.operation]
                return updated.with_changes(
                    func=cst.parse_expression(f"{receiver}.{window}.{name}")
                )

        if self.operation == "resample_label" and name == "resample":
            if not {_keyword(a) for a in updated.args} & {"label", "closed"}:
                self.applied = True
                kept = [
                    a.with_changes(comma=cst.MaybeSentinel.DEFAULT)
                    for a in updated.args
                ]
                return updated.with_changes(
                    args=[*kept, _kwarg("label", "right"), _kwarg("closed", "right")]
                )

        if self.operation == "fit_in_fold" and name == "fit":
            if any(a.keyword is None for a in updated.args):
                self.applied = True
                return updated.with_changes(
                    args=[
                        a.with_changes(
                            value=cst.parse_expression(_code(a.value) + _FOLD),
                            comma=cst.MaybeSentinel.DEFAULT,
                        )
                        if a.keyword is None
                        else a.with_changes(comma=cst.MaybeSentinel.DEFAULT)
                        for a in updated.args
                    ]
                )

        if self.operation == "chronological_split" and name == "train_test_split":
            args = [
                a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in updated.args
            ]
            for i, arg in enumerate(args):
                if _keyword(arg) == "shuffle":
                    self.applied = True
                    args[i] = arg.with_changes(value=cst.Name("False"))
                    return updated.with_changes(args=args)
            self.applied = True
            return updated.with_changes(args=[*args, _kwarg("shuffle", None)])

        if self.operation == "forward_fill":
            if name in _BFILL:
                self.applied = True
                return updated.with_changes(
                    func=updated.func.with_changes(attr=cst.Name("ffill"))
                )
            args = list(updated.args)
            for i, arg in enumerate(args):
                if _keyword(arg) == "method" and _string_of(arg.value) in _BFILL:
                    self.applied = True
                    args[i] = arg.with_changes(value=cst.SimpleString('"ffill"'))
                    return updated.with_changes(args=args)

        return updated

    def leave_Assign(self, original: cst.Assign, updated: cst.Assign) -> cst.Assign:
        if self.applied or not self._covers_line(original):
            return updated

        if self.operation == "lag":
            self.applied = True
            code = _code(updated.value)
            return updated.with_changes(
                value=cst.parse_expression(f"({code}).shift(1)")
            )

        if self.operation == "drop_column":
            kept = _without_forward_operands(updated.value, self.forward_names)
            if kept is not None:
                self.applied = True
                return updated.with_changes(value=kept)
            # It read a forward-looking name and still nothing could be taken
            # away: the assigned value IS the leak, not a conjunct alongside it.
            if _names_in(updated.value) & self.forward_names:
                self.boundary = (
                    f"{BOUNDARY_MARKER} the value assigned here is itself the "
                    "leaking comparison; no operand can be dropped, and rebuilding "
                    "it needs a substitute column — a judgement, not a removal"
                )

        return updated


def _code(node: cst.CSTNode) -> str:
    return cst.Module(body=()).code_for_node(node)


def _kwarg(name: str, text: str | None) -> cst.Arg:
    """A keyword argument rendered as `name=value`, no spaces around the `=`.

    `text` is a plain string for a string literal and None for `False`; nothing
    here accepts a caller-supplied expression, so no operation can smuggle in a
    computed value.
    """
    value = cst.Name("False") if text is None else cst.SimpleString(f'"{text}"')
    return cst.Arg(
        keyword=cst.Name(name),
        value=value,
        equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
        ),
        comma=cst.MaybeSentinel.DEFAULT,
    )


def _windowed(func: cst.Attribute) -> bool:
    """True for `x.rolling(20).mean()` - a window already bounds the stat."""
    inner = func.value
    return (
        isinstance(inner, cst.Call)
        and isinstance(inner.func, cst.Attribute)
        and inner.func.attr.value in _WINDOWED
    )


def _operands(node: cst.BaseExpression) -> list[cst.BaseExpression]:
    """Flatten `a & b & c` into its leaves.

    Pandas masks combine with & and |, which parse as BinaryOperation. `and`
    and `or` never appear on a Series mask - they would raise - so the boolean
    operators are deliberately not handled here.
    """
    if isinstance(node, cst.BinaryOperation) and isinstance(
        node.operator, cst.BitAnd | cst.BitOr
    ):
        return _operands(node.left) + _operands(node.right)
    return [node]


def _without_forward_operands(
    value: cst.BaseExpression, forward_names: frozenset[str]
) -> cst.BaseExpression | None:
    """Drop the conjuncts that read a forward-looking column, keep the rest.

    None when the removal cannot be made honestly: nothing to drop, or
    everything would go. A signal that IS the leaking comparison cannot be
    repaired by removal - rebuilding it needs a substitute column, and choosing
    one is a judgement rather than a deletion.
    """
    parts = _operands(value)
    if len(parts) < 2:
        return None
    kept = [p for p in parts if not (_names_in(p) & forward_names)]
    if not kept or len(kept) == len(parts):
        return None
    joined = kept[0]
    for part in kept[1:]:
        joined = cst.BinaryOperation(left=joined, operator=cst.BitAnd(), right=part)
    return joined


def _names_in(node: cst.CSTNode) -> set[str]:
    found: set[str] = set()

    class _Walk(cst.CSTVisitor):
        def visit_Name(self, node: cst.Name) -> None:
            found.add(node.value)

    node.visit(_Walk())
    return found


def _forward_names(source: str) -> frozenset[str]:
    """Names assigned from an expression that reads a later row.

    Read with `ast`, not libcst: nothing here is rewritten. It only decides
    which operand drop_column is entitled to remove, and picking the wrong one
    would delete a legitimate condition instead of the leak.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _reads_forward(node.value):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return frozenset(names)


def _reads_forward(node: ast.expr) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Attribute) or child.func.attr != "shift":
            continue
        first = child.args[0] if child.args else None
        if isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub):
            return True
        if (
            isinstance(first, ast.Constant)
            and isinstance(first.value, int)
            and first.value < 0
        ):
            return True
    return False


def _keyword(arg: cst.Arg) -> str:
    return arg.keyword.value if arg.keyword is not None else ""


def _string_of(node: cst.BaseExpression) -> str:
    return node.raw_value if isinstance(node, cst.SimpleString) else ""


def apply_patch(path: Path, candidate: LeakCandidate, operation: str) -> PatchResult:
    source = path.read_text("utf-8")

    if operation not in OPERATIONS:
        return PatchResult(
            ok=False,
            patched_source=source,
            error=f"unknown operation {operation!r}; vocabulary is {OPERATIONS}",
        )

    try:
        wrapper = MetadataWrapper(cst.parse_module(source))
    except cst.ParserSyntaxError as error:
        return PatchResult(
            ok=False, patched_source=source, error=f"{path} does not parse: {error}"
        )

    repair = _Repair(candidate.line, operation, _forward_names(source))
    patched = wrapper.visit(repair).code

    if not repair.applied:
        return PatchResult(
            ok=False,
            patched_source=source,
            error=repair.boundary
            or (
                f"{operation} found nothing to transform at {path.name}:"
                f"{candidate.line}"
            ),
        )

    diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
    )
    return PatchResult(ok=True, patched_source=patched, diff=diff)
