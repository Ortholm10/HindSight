"""Runs inside the sandbox subprocess. Untrusted code executes here, not upstream.

This is the one module in `hindsight_core` allowed to print: its stdout *is* the
return channel back to the parent. The audited strategy may print too, so the
result is emitted after a sentinel and the parent reads only what follows the
last one.
"""

from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import random
import socket
import sys
import traceback
from pathlib import Path

SENTINEL = "<<<HINDSIGHT-RESULT>>>"

# Fixed, not derived from anything: the before and after runs of one audit
# must share a draw, and a judge re-running the audit must get our numbers.
_SEED = 20220103


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise PermissionError("sandbox: network access is blocked")


def _assert_inside(root: Path, target: object) -> None:
    try:
        resolved = Path(os.fspath(target)).resolve()
    except TypeError:
        # An already-open descriptor. It could only have been obtained through
        # a call that already passed this check.
        return
    if root != resolved and root not in resolved.parents:
        raise PermissionError(f"sandbox: write outside {root} blocked: {resolved}")


_WRITE_FLAGS = ("w", "a", "x", "+")


def _contain(root: Path) -> None:
    """Deny the two things audited code has no business doing: reaching the
    network, and writing anywhere but its own scratch copy.

    Reads stay open - the strategy has to import its libraries and read the
    data file, which lives outside `root` by design.

    Three separate open() doors have to be shut, not one: `builtins.open`,
    `io.open` (what pathlib actually calls), and `os.open` (what bypasses both).
    Patching only the first leaves `Path(...).write_text(...)` wide open.
    """
    # socket.socket stays a class. Replacing it with a function breaks every
    # later `class X(socket.socket)` - ssl does exactly that, and sklearn
    # reaches ssl through joblib and asyncio, so a swapped-out class turns
    # every scikit-learn strategy into an unexplained crash indistinguishable
    # from an untestable one. Blocking the methods that actually reach the
    # network keeps the type intact.
    for method in (
        "__init__",
        "connect",
        "connect_ex",
        "bind",
        "sendto",
        "sendall",
        "send",
    ):
        setattr(socket.socket, method, _blocked)
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.socketpair = _blocked  # type: ignore[assignment]

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        if any(flag in mode for flag in _WRITE_FLAGS):
            _assert_inside(root, file)
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open  # type: ignore[assignment]
    io.open = guarded_open  # type: ignore[assignment]

    real_os_open = os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def guarded_os_open(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if flags & write_flags:
            _assert_inside(root, path)
        return real_os_open(path, flags, *args, **kwargs)

    os.open = guarded_os_open  # type: ignore[assignment]

    for name in ("remove", "unlink", "rmdir", "mkdir", "truncate"):
        _guard_first_arg(name, root)

    # ponytail: address-space cap is POSIX-only; on Windows the parent's
    # wall-clock timeout is the only cap. Upgrade path is a job object.
    try:
        import resource

        limit = 4 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass


def _guard_first_arg(name: str, root: Path) -> None:
    real = getattr(os, name, None)
    if real is None:
        return

    def guard(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        _assert_inside(root, path)
        return real(path, *args, **kwargs)

    setattr(os, name, guard)


def _import_strategy(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("hindsight_audited", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import audited file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    script = Path(sys.argv[1]).resolve()
    data_path = Path(sys.argv[2]).resolve()

    # Imported before containment so the libraries' own start-up file access is
    # not mistaken for an escape attempt.
    import numpy as np
    import pandas as pd

    from hindsight_core.metrics import evaluate

    df = pd.read_csv(data_path, index_col=0, parse_dates=[0]).sort_index()

    # Seeded before the strategy is imported, the same way eval/runner.py seeds
    # each case. Without this, any strategy that draws from the global RNG
    # disagrees with itself: l10 calls train_test_split(shuffle=True) with no
    # random_state, which moved Sharpe across 3.39 / 3.58 / 3.77 on three runs
    # of one unchanged file. A before/after delta then stops being evidence of
    # anything. It makes the two runs *comparable*; it does not make either one
    # right, and a strategy that seeds its own estimators (l11) was never
    # affected either way.
    random.seed(_SEED)
    np.random.seed(_SEED)

    _contain(script.parent)

    module = _import_strategy(script)
    equity = module.run_strategy(df)
    positions = module.run_positions(df)
    outcome, metrics, changes = evaluate(equity, positions)

    print(SENTINEL)
    print(
        json.dumps(
            {
                "outcome": outcome.value,
                "metrics": metrics,
                "position_changes": changes,
                "equity": [
                    [str(stamp)[:10], float(value)]
                    for stamp, value in zip(equity.index, equity, strict=True)
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    # sys.exit stays outside the try: SystemExit is a BaseException, and
    # catching the clean exit would report every success as a crash.
    try:
        code = main()
    except BaseException:  # noqa: B036 - the child reports failures, never raises
        traceback.print_exc()
        code = 1
    sys.exit(code)
