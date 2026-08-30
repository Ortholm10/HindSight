"""apply_patch is pure: it returns a patched source and a diff, and touches no
file. That is what makes 'never a partially-written file' true by construction
rather than by care."""

from pathlib import Path

import pytest

from hindsight_core.models import LeakCandidate
from hindsight_core.tools.apply_patch import BOUNDARY_MARKER, apply_patch

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


# --------------------------------------------------------------------------
# The full taxonomy 7 vocabulary. One test per operation proves the transform
# can express its row of that table; one proves the result still runs, because
# a repair that does not execute proves nothing and must be told apart from a
# repair that does.
# --------------------------------------------------------------------------

from hindsight_core.models import SandboxOutcome  # noqa: E402
from hindsight_core.sandbox import run_sandboxed  # noqa: E402
from hindsight_core.tools.apply_patch import OPERATIONS  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "eval" / "data" / "SPY.csv"

TAIL = """

def run_strategy(df):
    returns = run_positions(df) * df["close"].pct_change()
    return (1 + returns.fillna(0)).cumprod()
"""

EXPANDING = (
    """import pandas as pd


def run_positions(df):
    closes = df["close"]
    spread = closes / closes.rolling(20).mean() - 1
    cutoff = spread.quantile(0.15)
    raw = spread < cutoff
    return raw.shift(1).fillna(False).astype(int)
"""
    + TAIL
)

RESAMPLE = (
    """import pandas as pd


def run_positions(df):
    closes = df["close"]
    monthly = closes.resample("MS").mean()
    filt = monthly.reindex(closes.index, method="ffill")
    raw = closes < filt
    return raw.shift(1).fillna(False).astype(int)
"""
    + TAIL
)

CONJUNCTION = (
    """import pandas as pd


def run_positions(df):
    closes = df["close"]
    forward_max = closes.rolling(10).max().shift(-10)
    trend = closes > closes.rolling(20).mean()
    signal = trend & (forward_max > closes)
    return signal.shift(1).fillna(False).astype(int)
"""
    + TAIL
)

FIT = (
    """import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def run_positions(df):
    closes = df["close"]
    features = pd.DataFrame({"r1": closes.pct_change()}).dropna()
    target = (closes.shift(-1) > closes).astype(int).reindex(features.index)
    split = int(len(features) * 0.5)
    model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1)
    model.fit(features, target)
    pred = pd.Series(model.predict(features), index=features.index)
    return pred.reindex(closes.index).fillna(0).astype(int).shift(1).fillna(0)
"""
    + TAIL
)

SPLIT = (
    """import pandas as pd
from sklearn.model_selection import train_test_split


def run_positions(df):
    closes = df["close"]
    features = pd.DataFrame({"r1": closes.pct_change()}).dropna()
    target = (closes.shift(-1) > closes).astype(int).reindex(features.index)
    x_train, _, y_train, _ = train_test_split(
        features, target, test_size=0.5, shuffle=True
    )
    hit = features["r1"] > x_train["r1"].mean()
    return hit.reindex(closes.index).fillna(False).astype(int).shift(1).fillna(0)
"""
    + TAIL
)


def _candidate(line: int, leak_type: str = "L05") -> LeakCandidate:
    return LeakCandidate(
        leak_type=leak_type, file="s.py", line=line, snippet="", reason=""
    )


def _patch(tmp_path: Path, source: str, line: int, operation: str):
    path = tmp_path / "strategy.py"
    path.write_text(source, "utf-8")
    return path, apply_patch(path, _candidate(line), operation)


def _runs(tmp_path: Path, patched: str) -> SandboxOutcome:
    path = tmp_path / "patched.py"
    path.write_text(patched, "utf-8")
    return run_sandboxed(path, DATA, timeout_s=180).outcome


def test_the_vocabulary_covers_every_taxonomy_operation():
    """Eight rows in docs/taxonomy.md 7 are patchable; L12 is report-only."""
    assert set(OPERATIONS) == {
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
    }


def test_expanding_stat_replaces_a_full_sample_statistic(tmp_path):
    _, result = _patch(tmp_path, EXPANDING, 7, "expanding_stat")

    assert result.ok, result.error
    assert "spread.expanding(min_periods=20).quantile(0.15)" in result.patched_source
    assert _runs(tmp_path, result.patched_source) is not SandboxOutcome.CRASHED


def test_resample_label_adds_an_explicit_label_and_closed(tmp_path):
    _, result = _patch(tmp_path, RESAMPLE, 6, "resample_label")

    assert result.ok, result.error
    assert 'resample("MS", label="right", closed="right")' in result.patched_source
    assert _runs(tmp_path, result.patched_source) is not SandboxOutcome.CRASHED


def test_drop_column_removes_the_leaking_operand_from_a_conjunction(tmp_path):
    _, result = _patch(tmp_path, CONJUNCTION, 8, "drop_column")

    assert result.ok, result.error
    assert "forward_max" not in result.patched_source.splitlines()[7]
    assert "signal = trend" in result.patched_source
    assert _runs(tmp_path, result.patched_source) is not SandboxOutcome.CRASHED


def test_fit_in_fold_restricts_the_fit_to_the_training_fold(tmp_path):
    _, result = _patch(tmp_path, FIT, 11, "fit_in_fold")

    assert result.ok, result.error
    assert "model.fit(features.iloc[:split], target.iloc[:split])" in (
        result.patched_source
    )
    assert _runs(tmp_path, result.patched_source) is not SandboxOutcome.CRASHED


def test_chronological_split_turns_a_shuffled_split_ordered(tmp_path):
    _, result = _patch(tmp_path, SPLIT, 9, "chronological_split")

    assert result.ok, result.error
    assert "shuffle=False" in result.patched_source
    assert "shuffle=True" not in result.patched_source
    assert _runs(tmp_path, result.patched_source) is not SandboxOutcome.CRASHED


def test_an_operation_that_cannot_express_the_repair_fails_cleanly(tmp_path):
    """Not every line admits every operation, and saying so is a result.

    drop_column removes one operand of a boolean conjunction. Where the signal
    IS the leaking comparison, rebuilding it needs a substitute column - a
    judgement, not a removal - so the transform must decline rather than invent
    one, and the original source must come back untouched.
    """
    source = EXPANDING
    _, result = _patch(tmp_path, source, 7, "drop_column")

    assert not result.ok
    assert result.patched_source == source
    assert "drop_column" in result.error


ROLLING_STAT = (
    """import pandas as pd


def run_positions(df):
    closes = df["close"]
    spread = closes / closes.rolling(20).mean() - 1
    cutoff = spread.quantile(0.15)
    raw = spread < cutoff
    return raw.shift(1).fillna(False).astype(int)
"""
    + TAIL
)


def test_rolling_stat_bounds_a_full_sample_statistic_to_a_window(tmp_path):
    """Taxonomy 7 allows expanding() OR rolling(n) for a full-sample statistic.

    They are not interchangeable in effect: an expanding quantile keeps every
    row ever seen, a rolling one forgets. l05's validated fix is the rolling
    form, so the vocabulary has to be able to say it.
    """
    _, result = _patch(tmp_path, ROLLING_STAT, 7, "rolling_stat")

    assert result.ok, result.error
    assert "spread.rolling(40, min_periods=20).quantile(0.15)" in result.patched_source
    assert _runs(tmp_path, result.patched_source) is not SandboxOutcome.CRASHED


MULTILINE_TARGET = (
    """import pandas as pd
from sklearn.model_selection import train_test_split


def run_positions(df):
    features = pd.DataFrame({"r1": df["close"].pct_change()}).dropna()
    target = (df["close"] > 0).astype(int).reindex(features.index)
    a, b, c, d = train_test_split(
        features,
        target,
        test_size=0.5,
        shuffle=True,
    )
    hit = features["r1"] > a["r1"].mean()
    return hit.reindex(df.index).fillna(False).astype(int).shift(1).fillna(0)
"""
    + TAIL
)


def test_a_candidate_on_an_argument_line_still_patches_its_call(tmp_path):
    """scan_file names the argument's line; apply_patch must find its call.

    The two tools have to agree on what a line refers to. When the scanner
    started naming `shuffle=True` on line 12 instead of the call opening on
    line 9, every transform keyed to the opening line silently stopped
    applying - the repair became unreachable without a single test failing.
    """
    _, result = _patch(tmp_path, MULTILINE_TARGET, 12, "chronological_split")

    assert result.ok, result.error
    assert "shuffle=False" in result.patched_source
    assert _runs(tmp_path, result.patched_source) is not SandboxOutcome.CRASHED


BOUNDARY_SOURCE = '''import pandas as pd


def run_positions(df):
    forward_max = df["close"].rolling(10).max().shift(-10)
    signal = forward_max > df["close"] * 1.05
    return signal
'''

NO_FORWARD_SOURCE = '''import pandas as pd


def run_positions(df):
    signal = df["close"] > df["open"]
    return signal
'''


def test_drop_column_reports_the_vocabulary_boundary_when_the_signal_is_the_leak(
    tmp_path,
):
    """L08's shape: the decision IS the leaking comparison. Removing an operand
    is not available, and rebuilding it needs a substitute column — a judgement,
    not a deletion. That has to read differently from 'operation not
    applicable', because the two ask the prover for different next moves."""
    path = tmp_path / "s.py"
    path.write_text(BOUNDARY_SOURCE, "utf-8")
    candidate = LeakCandidate(
        leak_type="L03", file=str(path), line=6, snippet="signal = ...", reason="r"
    )

    result = apply_patch(path, candidate, "drop_column")

    assert result.ok is False
    assert result.error.startswith(BOUNDARY_MARKER)
    assert "substitute" in result.error


def test_drop_column_that_simply_does_not_apply_is_not_a_boundary(tmp_path):
    """No forward-looking name anywhere: drop_column is the wrong tool here, not
    a limit of the vocabulary. The prover must keep trying other operations."""
    path = tmp_path / "s.py"
    path.write_text(NO_FORWARD_SOURCE, "utf-8")
    candidate = LeakCandidate(
        leak_type="L03", file=str(path), line=5, snippet="signal = ...", reason="r"
    )

    result = apply_patch(path, candidate, "drop_column")

    assert result.ok is False
    assert not result.error.startswith(BOUNDARY_MARKER)
