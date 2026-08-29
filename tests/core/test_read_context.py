"""read_context is the main token cost per candidate, so the window is asserted
to be tight, not merely correct."""

from pathlib import Path

import pytest

from hindsight_core.models import LeakCandidate
from hindsight_core.tools.read_context import read_context

SOURCE = (
    "\n".join(
        ["import pandas as pd", ""]
        + [f"NOISE_{i} = {i}" for i in range(40)]
        + [
            "",
            "",
            "def run_positions(df):",
            "    sma = df['close'].rolling(20).mean()",
            "    signal = df['close'].shift(-1) > sma",
            "    return signal.astype(int)",
            "",
            "",
        ]
        + [f"MORE_{i} = {i}" for i in range(40)]
    )
    + "\n"
)

TARGET = 47  # the shift(-1) line


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "strategy.py"
    path.write_text(SOURCE, "utf-8")
    return path


def candidate(line: int) -> LeakCandidate:
    return LeakCandidate(
        leak_type="L01", file="strategy.py", line=line, snippet="", reason=""
    )


def test_the_window_contains_the_candidate_line(script):
    context = read_context(script, candidate(TARGET))

    assert "shift(-1)" in context
    assert str(TARGET) in context


def test_the_window_is_clamped_to_the_enclosing_function(script):
    context = read_context(script, candidate(TARGET), radius=15)

    assert "def run_positions(df):" in context
    assert "NOISE_39" not in context
    assert "MORE_0" not in context


def test_the_window_never_exceeds_the_radius(script):
    context = read_context(script, candidate(5), radius=4)

    assert len(context.splitlines()) <= 9


def test_lines_are_numbered_so_the_agent_can_name_a_line(script):
    context = read_context(script, candidate(TARGET))

    assert f"{TARGET}" in context
    assert all(line.strip() for line in context.splitlines())


def test_a_candidate_past_the_end_of_the_file_is_empty_not_a_crash(script):
    assert read_context(script, candidate(9999)) == ""
