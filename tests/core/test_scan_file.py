"""scan_file is a candidate generator, not a detector. Tests assert recall and
that it never kills the agent loop — never precision. Killing false candidates
is the prover's job."""

from pathlib import Path

from hindsight_core.tools.scan_file import scan_file

REPO = Path(__file__).resolve().parents[2]
L01 = REPO / "eval" / "cases" / "l01_future_index" / "strategy.py"

MANY_PATTERNS = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def features(df):
    future = df["close"].shift(-1)
    centered = df["close"].rolling(20, center=True).mean()
    normed = (df["close"] - df["close"].mean()) / df["close"].std()
    filled = df["close"].bfill()
    weekly = df["close"].resample("W").last()
    peak = df["close"].idxmax()
    scaled = StandardScaler().fit_transform(df[["close"]])
    a, b = train_test_split(df)
    return future, centered, normed, filled, weekly, peak, scaled, a, b
"""


def test_finds_the_future_shift_at_the_ground_truth_line():
    candidates = scan_file(L01)

    hits = [c for c in candidates if c.line == 8]
    assert hits, (
        f"nothing found at line 8; got {[(c.line, c.leak_type) for c in candidates]}"
    )
    assert any(c.leak_type == "L01" for c in hits)


def test_over_produces_across_the_taxonomy(tmp_path):
    path = tmp_path / "many.py"
    path.write_text(MANY_PATTERNS, "utf-8")

    found = {c.leak_type for c in scan_file(path)}

    assert {"L01", "L02", "L05", "L06", "L07", "L09", "L10", "L11"} <= found


def test_every_candidate_carries_a_usable_line_and_snippet(tmp_path):
    path = tmp_path / "many.py"
    path.write_text(MANY_PATTERNS, "utf-8")
    lines = MANY_PATTERNS.splitlines()

    for candidate in scan_file(path):
        assert 1 <= candidate.line <= len(lines)
        assert candidate.snippet.strip()
        assert candidate.reason.strip()
        assert candidate.file == str(path)


def test_an_unparseable_file_yields_no_candidates_rather_than_crashing(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def oops(:\n    pass\n", "utf-8")

    assert scan_file(path) == []


CASES = Path(__file__).resolve().parents[2] / "eval" / "cases"


def test_l04_fires_on_an_unlagged_higher_timeframe_merge():
    """The HTF chain reaches merge_asof with no shift anywhere along it.

    Over-production is intended here - several links of the chain are flagged -
    but the ground-truth line must be among them, because the prover can only
    disprove candidates it was given.
    """
    candidates = scan_file(CASES / "l04_htf_merge" / "strategy.py")
    l04 = [c for c in candidates if c.leak_type == "L04"]

    assert l04, "no L04 candidate proposed"
    assert 13 in [c.line for c in l04]


def test_l04_stays_silent_on_the_control_that_lags_before_the_merge():
    """c04 differs by one token: weekly_sma.shift(1) before the join.

    A rule that cannot see that token would flag the correct code too, and a
    detector that flags everything scores zero on the false-positive metric.
    """
    candidates = scan_file(CASES / "c04_lagged_asof_merge" / "strategy.py")

    assert [c for c in candidates if c.leak_type == "L04"] == []
