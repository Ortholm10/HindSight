"""The sandbox's whole job is to turn every way untrusted code can misbehave
into a distinct, readable outcome. Each test here pins one of those ways."""

from pathlib import Path

import pytest

from hindsight_core.models import SandboxOutcome
from hindsight_core.sandbox import run_sandboxed

TRADES = """
import pandas as pd


def run_positions(df):
    sma = df["close"].rolling(3).mean()
    return (df["close"] > sma).shift(1).fillna(False).astype(int)


def run_strategy(df):
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
"""

NEVER_TRADES = """
import pandas as pd


def run_positions(df):
    return pd.Series(0, index=df.index)


def run_strategy(df):
    return pd.Series(1.0, index=df.index)
"""

CRASHES = """
def run_positions(df):
    raise ValueError("deliberate explosion")


def run_strategy(df):
    return run_positions(df)
"""

HANGS = """
def run_positions(df):
    while True:
        pass


def run_strategy(df):
    return run_positions(df)
"""

MISSING_IMPORT = """
import a_library_that_does_not_exist


def run_positions(df):
    return df["close"]


def run_strategy(df):
    return df["close"]
"""

NETWORK = """
import socket

socket.create_connection(("example.com", 80), timeout=5)


def run_positions(df):
    return df["close"]


def run_strategy(df):
    return df["close"]
"""

NOISY = """
import pandas as pd

print("strategy stdout noise that must not corrupt the protocol")


def run_positions(df):
    sma = df["close"].rolling(3).mean()
    return (df["close"] > sma).shift(1).fillna(False).astype(int)


def run_strategy(df):
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
"""


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    rows = ["date,open,high,low,close,volume"]
    price = 100.0
    for i in range(60):
        price *= 1.004 if i % 3 else 0.994
        day = f"2022-{1 + i // 28:02d}-{1 + i % 28:02d}"
        rows.append(f"{day},{price},{price},{price},{price},1000")
    path = tmp_path / "data.csv"
    path.write_text("\n".join(rows) + "\n", "utf-8")
    return path


@pytest.fixture
def make_script(tmp_path: Path):
    def _make(source: str, name: str = "strategy.py") -> Path:
        path = tmp_path / name
        path.write_text(source, "utf-8")
        return path

    return _make


def test_working_strategy_completes_with_metrics(make_script, data_path):
    record = run_sandboxed(make_script(TRADES), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.COMPLETED
    assert record.metrics["sharpe"] == record.metrics["sharpe"]  # not NaN
    assert record.position_changes > 0
    assert record.run_id
    assert len(record.equity) == 60


def test_strategy_that_never_trades_is_untestable_not_a_zero_score(
    make_script, data_path
):
    record = run_sandboxed(make_script(NEVER_TRADES), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.ZERO_TRADES
    assert record.metrics == {}


def test_crashing_strategy_returns_the_traceback(make_script, data_path):
    record = run_sandboxed(make_script(CRASHES), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert "deliberate explosion" in record.stderr
    assert record.metrics == {}


@pytest.mark.timeout(60)
def test_hanging_strategy_times_out(make_script, data_path):
    record = run_sandboxed(make_script(HANGS), data_path, timeout_s=3)

    assert record.outcome is SandboxOutcome.TIMED_OUT
    assert record.metrics == {}


def test_missing_dependency_is_a_crash_not_a_propagated_importerror(
    make_script, data_path
):
    record = run_sandboxed(make_script(MISSING_IMPORT), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert "a_library_that_does_not_exist" in record.stderr


def test_network_access_is_blocked(make_script, data_path):
    record = run_sandboxed(make_script(NETWORK), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert "network" in record.stderr.lower()


def test_writing_outside_the_sandbox_is_blocked(make_script, data_path, tmp_path):
    escape = tmp_path / "escaped.txt"
    source = (
        f"open({str(escape)!r}, 'w').write('pwned')\n\n\n"
        "def run_positions(df):\n    return df['close']\n\n\n"
        "def run_strategy(df):\n    return df['close']\n"
    )

    record = run_sandboxed(make_script(source), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert not escape.exists()


PATHLIB_ESCAPE = """
from pathlib import Path

Path({target!r}).write_text("pwned")


def run_positions(df):
    return df["close"]


def run_strategy(df):
    return df["close"]
"""

OS_ESCAPE = """
import os

os.write(os.open({target!r}, os.O_WRONLY | os.O_CREAT), b"pwned")


def run_positions(df):
    return df["close"]


def run_strategy(df):
    return df["close"]
"""


def test_writing_outside_the_sandbox_via_pathlib_is_blocked(
    make_script, data_path, tmp_path
):
    """pathlib reaches io.open, not builtins.open - guarding one is not
    guarding the other."""
    escape = tmp_path / "escaped_via_pathlib.txt"
    source = PATHLIB_ESCAPE.format(target=str(escape))

    record = run_sandboxed(make_script(source), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert not escape.exists()


def test_os_level_writes_outside_the_sandbox_are_blocked(
    make_script, data_path, tmp_path
):
    """os.open bypasses every Python-level open() wrapper."""
    escape = tmp_path / "escaped_via_os.txt"
    source = OS_ESCAPE.format(target=str(escape))

    record = run_sandboxed(make_script(source), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert not escape.exists()


MUTATION = """
import os

{call}


def run_positions(df):
    return df["close"]


def run_strategy(df):
    return df["close"]
"""

SOCKET_USE = """
import socket

{call}


def run_positions(df):
    return df["close"]


def run_strategy(df):
    return df["close"]
"""


@pytest.mark.parametrize(
    "call, kind, exists_after",
    [
        ("os.remove({target!r})", "file", True),
        ("os.unlink({target!r})", "file", True),
        ("os.truncate({target!r}, 0)", "file", True),
        ("os.rmdir({target!r})", "dir", True),
        ("os.mkdir({target!r})", "absent", False),
    ],
    ids=["remove", "unlink", "truncate", "rmdir", "mkdir"],
)
def test_os_mutations_outside_the_sandbox_are_blocked(
    make_script, data_path, tmp_path, call, kind, exists_after
):
    """Each guarded os.* entry point gets its own assertion.

    A patch installed in a loop and never exercised is indistinguishable from
    no patch at all - that is how the io.open hole survived its first review.
    """
    target = tmp_path / f"target_{kind}"
    if kind == "file":
        target.write_text("keepme", "utf-8")
    elif kind == "dir":
        target.mkdir()

    source = MUTATION.format(call=call.format(target=str(target)))
    record = run_sandboxed(make_script(source), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert target.exists() is exists_after
    if kind == "file":
        assert target.read_text("utf-8") == "keepme"


@pytest.mark.parametrize(
    "call",
    [
        "socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
        "socket.create_connection(('example.com', 80), timeout=5)",
        "socket.socketpair()",
    ],
    ids=["socket", "create_connection", "socketpair"],
)
def test_every_patched_socket_entry_point_is_blocked(make_script, data_path, call):
    record = run_sandboxed(make_script(SOCKET_USE.format(call=call)), data_path, 60)

    assert record.outcome is SandboxOutcome.CRASHED
    assert "network" in record.stderr.lower()


def test_the_audited_file_is_never_run_in_place(make_script, data_path):
    source = (
        TRADES + "\n\nwith open(__file__, 'a') as f:\n    f.write('# tampered\n')\n"
    )
    script = make_script(source)

    run_sandboxed(script, data_path, timeout_s=60)

    assert script.read_text("utf-8") == source


def test_strategy_stdout_does_not_corrupt_the_result(make_script, data_path):
    record = run_sandboxed(make_script(NOISY), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.COMPLETED


IMPORTS_SSL = """
import pandas as pd
import ssl  # noqa: F401 - a stand-in for sklearn, which reaches ssl via joblib


def run_positions(df):
    sma = df["close"].rolling(3).mean()
    return (df["close"] > sma).shift(1).fillna(False).astype(int)


def run_strategy(df):
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
"""


def test_containment_does_not_break_a_later_import_of_ssl(make_script, data_path):
    """The network guard must not make honest libraries unimportable.

    `ssl` does `class SSLSocket(socket)`, so replacing socket.socket with a
    function makes it unsubclassable and every strategy that imports sklearn -
    which reaches ssl through joblib and asyncio - dies before it ever trades.
    That failure is indistinguishable from an untestable strategy, which is
    exactly the confusion this sandbox exists to prevent.
    """
    record = run_sandboxed(make_script(IMPORTS_SSL), data_path, timeout_s=60)

    assert record.outcome is SandboxOutcome.COMPLETED, record.stderr
