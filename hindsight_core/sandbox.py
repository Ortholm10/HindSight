"""Subprocess execution with a hard timeout, resource caps, and a fixed seed.

Audited code is untrusted and often broken. Every distinct failure maps to a
distinct SandboxOutcome — never one collapsed error path. Broken input is the
normal case here, so nothing the child does may raise into the agent loop.

Determinism is a guarantee of this module, not an implementation detail
=======================================================================

Every audited strategy runs under a fixed global RNG seed (`_child._SEED`,
applied to both `random` and `numpy.random` before the strategy is imported).
This is load-bearing, not tidiness.

Hindsight's entire claim is differential: a leak is proven when removing the
illegitimate information measurably deflates the strategy. That subtraction is
only evidence if the two runs differ *by the patch and nothing else*. Under an
unseeded RNG they also differ by the draw, and the difference between those two
explanations is invisible in the number — which is exactly the class of silent,
plausible-looking failure this project exists to catch. Measured on
l10_random_split: forty runs of one unchanged file spanned Sharpe 3.03 to 4.75.
A patch worth 0.4 is unrecoverable inside that.

So a strategy that depends on unseeded randomness does not get a free pass to
be irreproducible, and its variance is not silently absorbed as measurement
noise. The sandbox fixes the seed and says so. Two consequences follow, and
both are intended:

* Re-running an audit reproduces its numbers exactly — for us, and for anyone
  checking our work from a clean clone.
* A strategy whose result depends on the draw is measured at one draw. That is
  a real limitation, and an honest one: it is stated here and belongs in the
  changelog rather than hidden behind a delta that looks decisive.
  Characterising such a strategy across seeds is a different tool than this
  one, and reporting a single seeded run as if it settled the matter would be
  the same overclaim Hindsight was built to refuse.

Seeding makes two runs *comparable*. It does not make either one *right*.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from hindsight_core._child import SENTINEL
from hindsight_core.models import RunRecord, SandboxOutcome

_CHILD = Path(__file__).resolve().parent / "_child.py"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{existing}" if existing else str(_REPO_ROOT)
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_sandboxed(script: Path, data_path: Path, timeout_s: float = 60.0) -> RunRecord:
    """Execute one strategy file against one CSV, in a copy, under a timeout.

    The audited file is copied into a fresh temp directory and run from there:
    patching a user's file on disk during an audit is not acceptable, and the
    copy is also what makes the write guard's "inside my own directory" rule
    meaningful.
    """
    run_id = uuid4().hex[:16]
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="hindsight-") as tmp:
        workdir = Path(tmp).resolve()
        copy = workdir / script.name
        copy.write_bytes(script.read_bytes())

        try:
            proc = subprocess.run(
                [sys.executable, str(_CHILD), str(copy), str(data_path.resolve())],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=workdir,
                env=_child_env(),
            )
        except subprocess.TimeoutExpired as expired:
            return RunRecord(
                run_id=run_id,
                outcome=SandboxOutcome.TIMED_OUT,
                stderr=f"sandbox: exceeded {timeout_s}s\n{expired.stderr or ''}",
                duration_s=time.perf_counter() - started,
            )
        except OSError as error:
            return RunRecord(
                run_id=run_id,
                outcome=SandboxOutcome.CRASHED,
                stderr=f"sandbox: could not start subprocess: {error}",
                duration_s=time.perf_counter() - started,
            )

    duration = time.perf_counter() - started
    payload = _parse(proc.stdout)

    if proc.returncode != 0 or payload is None:
        return RunRecord(
            run_id=run_id,
            outcome=SandboxOutcome.CRASHED,
            stderr=proc.stderr or proc.stdout[-4000:],
            duration_s=duration,
        )

    return RunRecord(
        run_id=run_id,
        outcome=SandboxOutcome(payload["outcome"]),
        metrics=payload["metrics"],
        position_changes=payload["position_changes"],
        stderr=proc.stderr,
        duration_s=duration,
        equity=tuple((str(d), float(v)) for d, v in payload["equity"]),
    )


def _parse(stdout: str) -> dict | None:
    """Read the result after the last sentinel — audited code prints too."""
    if SENTINEL not in stdout:
        return None
    try:
        return json.loads(stdout.rsplit(SENTINEL, 1)[1])
    except (ValueError, KeyError):
        return None
