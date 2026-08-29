"""libcst transforms from the closed fix vocabulary. Returns a unified diff.

Every operation here *removes* information the strategy was not entitled to —
a lag, a narrowed window, a dropped direction. None of them invents a value,
which is CLAUDE.md rule 2 expressed as a closed vocabulary rather than a
review comment.

The function is pure. It never writes: the caller applies the patched source to
a sandbox copy, so a failed transform cannot leave a half-edited file behind.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from hindsight_core.models import LeakCandidate, PatchResult

OPERATIONS = ("future_shift", "lag", "trailing_window", "forward_fill")

_BFILL = ("bfill", "backfill")


class _Repair(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, line: int, operation: str) -> None:
        self.line = line
        self.operation = operation
        self.applied = False

    def _starts_here(self, node: cst.CSTNode) -> bool:
        return self.get_metadata(PositionProvider, node).start.line == self.line

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.BaseExpression:
        if self.applied or not self._starts_here(original):
            return updated
        name = (
            updated.func.attr.value if isinstance(updated.func, cst.Attribute) else ""
        )

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
        if self.applied or self.operation != "lag" or not self._starts_here(original):
            return updated
        code = cst.Module(body=()).code_for_node(updated.value)
        self.applied = True
        return updated.with_changes(value=cst.parse_expression(f"({code}).shift(1)"))


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

    repair = _Repair(candidate.line, operation)
    patched = wrapper.visit(repair).code

    if not repair.applied:
        return PatchResult(
            ok=False,
            patched_source=source,
            error=(
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
