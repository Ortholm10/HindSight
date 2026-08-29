"""apply_patch is pure: it returns a patched source and a diff, and touches no
file. That is what makes 'never a partially-written file' true by construction
rather than by care."""

from pathlib import Path

import pytest

from hindsight_core.models import LeakCandidate
from hindsight_core.tools.apply_patch import apply_patch

SOURCE = '''"""A docstring that must survive."""

import pandas as pd


def run_positions(df):
    # This comment must survive too, with its exact indentation.
    sma = df["close"].rolling(20, center=True).mean()
    signal = df["close"].shift(-1) > sma
    filled = df["close"].bfill()
    position = signal & filled.notna()
    return position
'''


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "strategy.py"
    path.write_text(SOURCE, "utf-8")
    return path


def candidate(line: int, leak_type: str = "L01") -> LeakCandidate:
    return LeakCandidate(
        leak_type=leak_type, file="strategy.py", line=line, snippet="", reason=""
    )


def test_future_shift_becomes_a_backward_shift(script):
    result = apply_patch(script, candidate(9), "future_shift")

    assert result.ok
    assert 'df["close"].shift(1) > sma' in result.patched_source
    assert ".shift(-1)" not in result.patched_source


def test_comments_and_formatting_survive(script):
    result = apply_patch(script, candidate(9), "future_shift")

    assert '"""A docstring that must survive."""' in result.patched_source
    assert (
        "    # This comment must survive too, with its exact indentation."
        in result.patched_source
    )


def test_centered_window_becomes_trailing(script):
    result = apply_patch(script, candidate(8, "L02"), "trailing_window")

    assert result.ok
    assert "rolling(20).mean()" in result.patched_source
    assert "center=True" not in result.patched_source


def test_backward_fill_becomes_forward_fill(script):
    result = apply_patch(script, candidate(10, "L06"), "forward_fill")

    assert result.ok
    assert 'df["close"].ffill()' in result.patched_source


def test_lag_wraps_the_assigned_expression(script):
    result = apply_patch(script, candidate(11, "L03"), "lag")

    assert result.ok
    assert "position = (signal & filled.notna()).shift(1)" in result.patched_source


def test_the_diff_is_unified_and_names_the_file(script):
    result = apply_patch(script, candidate(9), "future_shift")

    assert result.diff.startswith("---")
    assert "strategy.py" in result.diff
    assert "+" in result.diff and "-" in result.diff


def test_a_line_with_nothing_to_transform_is_a_structured_failure(script):
    result = apply_patch(script, candidate(3), "future_shift")

    assert not result.ok
    assert result.error
    assert result.patched_source == SOURCE
    assert result.diff == ""


def test_an_unknown_operation_is_a_structured_failure(script):
    result = apply_patch(script, candidate(9), "reticulate_splines")

    assert not result.ok
    assert "reticulate_splines" in result.error


def test_unparseable_source_is_a_structured_failure(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def oops(:\n    pass\n", "utf-8")

    result = apply_patch(path, candidate(1), "future_shift")

    assert not result.ok
    assert result.error


def test_the_file_on_disk_is_never_written(script):
    apply_patch(script, candidate(9), "future_shift")

    assert script.read_text("utf-8") == SOURCE
