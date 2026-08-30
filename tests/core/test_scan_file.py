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


MULTILINE_SPLIT = """import pandas as pd
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
    return a
"""

MULTILINE_ROLLING = """import pandas as pd


def run_positions(df):
    sma = df["close"].rolling(
        20,
        min_periods=5,
        center=True,
    ).mean()
    return sma
"""


def test_a_candidate_names_the_argument_line_not_the_call_line(tmp_path):
    """A multi-line call spans many lines; only one of them carries the leak.

    Naming the opening line points a reader at `train_test_split(`, which is
    not the problem - `shuffle=True` four lines down is. The rule is general:
    where one argument is the trigger, that argument's line is the candidate's.
    """
    path = tmp_path / "split.py"
    path.write_text(MULTILINE_SPLIT, "utf-8")
    source = MULTILINE_SPLIT.splitlines()

    l10 = [c for c in scan_file(path) if c.leak_type == "L10"]

    assert l10, "no L10 candidate proposed"
    assert source[l10[0].line - 1].strip() == "shuffle=True,"


def test_the_argument_line_rule_holds_for_a_different_rule_and_call_shape(tmp_path):
    """Generality check, deliberately not the shape that motivated the rule:
    a different leak type, a different call, and two unrelated arguments
    sitting between the opening line and the offending one."""
    path = tmp_path / "rolling.py"
    path.write_text(MULTILINE_ROLLING, "utf-8")
    source = MULTILINE_ROLLING.splitlines()

    l02 = [c for c in scan_file(path) if c.leak_type == "L02"]

    assert l02, "no L02 candidate proposed"
    assert source[l02[0].line - 1].strip() == "center=True,"


# --- training labels are not leaky features ---------------------------------
#
# A supervised label is forward-looking by definition: `y` for row t is what
# happened at t+1. That is not a leak, and patching it does not remove
# illegitimate information — it corrupts the model, which deflates the strategy
# in a way differential execution cannot tell apart from a real repair.
# Measured cost before this rule existed: one false positive on c06 and a
# localisation on l10, where flipping the label took Sharpe 3.44 to 0.21 and
# then hid the real leak, because a model fed noise stops responding to
# anything.
#
# The cut is where the information goes, not what it looks like: a value
# absorbed by .fit() is training data; a value that reaches the decision by any
# other route is a feature.

LABEL_STRAIGHT_TO_FIT = """
import pandas as pd
from sklearn.linear_model import LogisticRegression


def run_positions(df):
    features = df[["close"]].pct_change().dropna()
    target = (df["close"].shift(-1) > df["close"]).astype(int)
    model = LogisticRegression()
    model.fit(features, target)
    return pd.Series(model.predict(features), index=features.index)
"""

LABEL_THROUGH_A_SPLITTER = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run_positions(df):
    features = df[["close"]].pct_change().dropna()
    target = (df["close"].shift(-1) > df["close"]).astype(int)
    x_train, _, y_train, _ = train_test_split(features, target, shuffle=True)
    model = RandomForestClassifier().fit(x_train, y_train)
    return pd.Series(model.predict(features), index=features.index)
"""

FORWARD_COLUMN_IN_THE_SIGNAL = """
import pandas as pd


def run_positions(df):
    forward_max = df["close"].rolling(10).max().shift(-10)
    signal = forward_max > df["close"] * 1.05
    return signal.shift(1).fillna(False).astype(int)
"""

FORWARD_COLUMN_FOR_REPORTING = """
import pandas as pd


def report(df):
    forward = df["close"].shift(-5)
    return pd.DataFrame({"forward": forward})
"""

LABEL_THAT_ALSO_REACHES_THE_SIGNAL = """
import pandas as pd
from sklearn.linear_model import LogisticRegression


def run_positions(df):
    features = df[["close"]].pct_change().dropna()
    target = (df["close"].shift(-1) > df["close"]).astype(int)
    LogisticRegression().fit(features, target)
    return target.shift(1).fillna(False).astype(int)
"""


def _scan(tmp_path, source, name="s.py"):
    path = tmp_path / name
    path.write_text(source, "utf-8")
    return scan_file(path)


def _l01_lines(candidates):
    return [c.line for c in candidates if c.leak_type == "L01"]


def test_a_label_consumed_only_by_fit_is_not_a_candidate(tmp_path):
    assert _l01_lines(_scan(tmp_path, LABEL_STRAIGHT_TO_FIT)) == []


def test_a_label_that_reaches_fit_through_a_splitter_is_not_a_candidate(tmp_path):
    """l10's shape. train_test_split is not itself a sink — the reachability
    follows the names it returns, and both of them end at .fit()."""
    assert _l01_lines(_scan(tmp_path, LABEL_THROUGH_A_SPLITTER)) == []


def test_a_forward_column_that_reaches_the_signal_is_still_a_candidate(tmp_path):
    """l08's shape, and the reason the rule cannot key off the shift alone:
    forward_max is built identically to a label and is used as a feature."""
    assert _l01_lines(_scan(tmp_path, FORWARD_COLUMN_IN_THE_SIGNAL)) == [6]


def test_a_forward_column_that_never_reaches_fit_is_still_a_candidate(tmp_path):
    """c03's shape. A reporting column is not a leak either, but it is not a
    label — clearing it is the prover's job, not the scanner's."""
    assert _l01_lines(_scan(tmp_path, FORWARD_COLUMN_FOR_REPORTING)) == [6]


def test_a_label_that_also_reaches_the_signal_stays_a_candidate(tmp_path):
    """Feeding it to .fit() does not launder it. One use outside training is
    enough to make the whole name a feature again."""
    assert _l01_lines(_scan(tmp_path, LABEL_THAT_ALSO_REACHES_THE_SIGNAL)) == [8]


def test_the_frozen_ml_cases_no_longer_offer_their_labels_as_candidates():
    cases = REPO / "eval" / "cases"
    for case, label_line in (
        ("c06_timeseries_split_pipeline", 30),
        ("l10_random_split", 23),
        ("l11_preprocess_before_split", 28),
    ):
        lines = _l01_lines(scan_file(cases / case / "strategy.py"))
        assert label_line not in lines, f"{case}: label at {label_line} still flagged"


def test_the_real_leaks_in_those_cases_survive_the_exemption():
    """The exemption must remove the label and nothing else — l10's shuffled
    split and l11's pre-split fit are the leaks those cases exist for."""
    cases = REPO / "eval" / "cases"
    l10 = scan_file(cases / "l10_random_split" / "strategy.py")
    assert ("L10", 25) in [(c.leak_type, c.line) for c in l10]
    l11 = scan_file(cases / "l11_preprocess_before_split" / "strategy.py")
    assert ("L11", 31) in [(c.leak_type, c.line) for c in l11]
    l08 = scan_file(cases / "l08_forward_target" / "strategy.py")
    assert ("L01", 7) in [(c.leak_type, c.line) for c in l08]
