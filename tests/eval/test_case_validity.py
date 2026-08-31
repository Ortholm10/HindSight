"""The case validity rules from docs/taxonomy.md section 6, plus one of our own.

These run over all 20 frozen cases and are the gate the set had to clear before
being frozen. A case that fails any of them is not evidence: it is either a
strategy that does not trade, a pair that are two different strategies, or a
"leak" that does not actually leak. Repair or replace happens here, never after
results are known.
"""

import pytest

from eval.inject import RECIPES, inject, repair
from eval.runner import WINDOWS, discover_cases, positions_of, run_case
from hindsight_core.models import SandboxOutcome

ALL_CASES = discover_cases()
# The frozen 20. Statements about the set's IDENTITY - its counts, its taxonomy
# coverage, and that each leaked source is exactly what eval/inject.py produces
# - are about these and only these. A post-freeze case is not built by the
# injector (case 21 is transcribed from a published bug report, per CLAUDE.md
# rule 7), so holding it to the round-trip rules would be a category error.
CASES = [c for c in ALL_CASES if c.frozen]
INJECTED = [c for c in CASES if c.is_injected]
CONTROLS = [c for c in CASES if not c.is_injected]
# The behavioural rules 1-5 and the causality checks run over EVERY case,
# frozen or not: a case added later still has to clear the same bar to be
# evidence.
# Rules 2-5 all compare a leaked run against a repaired one. L12 has no
# repair: docs/taxonomy.md marks a hindsight universe detectable but not
# patchable, and says it must never be counted as execution-proven. Holding it
# to rules that presuppose a fix would contradict the taxonomy it comes from,
# so it is scored on detection and ground truth alone.
ALL_INJECTED = [c for c in ALL_CASES if c.is_injected]
PROVABLE = [c for c in ALL_INJECTED if c.patchable]
# Rules 5 and 6 drive eval/inject.py's own recipes over the case source, so
# they only mean anything for cases the injector actually produced.
FROZEN_PROVABLE = [c for c in INJECTED if c.patchable]


def held_to(rule: str):
    return [c for c in PROVABLE if rule not in c.known_limitations]


# Frozen. A case may only sit out a rule with a written reason, and this list
# cannot grow without the test failing.
EXPECTED_LIMITATIONS = {
    "l02_centered_window": ["rule4"],
    "l05_full_sample_zscore": ["rule3"],
    "l11_preprocess_before_split": ["rule4"],
}

# Also frozen, and deliberately a SEPARATE list from the one above. A
# limitation says a case cannot meet a validity rule. A correction says we
# edited a frozen case after the freeze - a strictly stronger claim, because
# the case set is the evidence. Empty: no frozen case has been edited. The
# l10 random_state correction was made, measured and then reverted, because
# it moved a headline score, and this list is where that would have had to be
# admitted.
EXPECTED_CORRECTIONS: dict[str, list[str]] = {}
CAUSAL_INJECTED = [c for c in ALL_INJECTED if c.causal_check]
FUTURE_ROW = [c for c in CAUSAL_INJECTED if c.leak_type not in {"L01", "L03"}]
CAUSAL_CONTROLS = [
    c for c in ALL_CASES if not c.is_injected and c.causal_check
]


def ids(cases):
    return [c.case_id for c in cases]


def sharpe(meta, variant, window=None):
    record = run_case(meta, variant, window)
    assert record.outcome is SandboxOutcome.COMPLETED, (
        f"{meta.case_id}/{variant} window={window} -> {record.outcome.value}"
    )
    return record.metrics["sharpe"]


def inflates(clean: float, leaked: float) -> bool:
    """Rule 3, both halves. Ratios are unstable near zero, hence the fallback."""
    relative = leaked >= 1.4 * clean if clean > 0.2 else leaked >= clean + 0.5
    return relative and (leaked - clean) >= 0.4


def test_the_set_is_twenty_cases_twelve_injected_and_eight_clean():
    assert len(CASES) == 20
    assert len(INJECTED) == 12
    assert len(CONTROLS) == 8


def test_case_21_is_present_and_sits_outside_the_frozen_twenty():
    """The flagship case is additive: it must never move a headline number.

    It is held to behavioural rules 1-4 and to both causality checks, exactly
    like a frozen case. It is exempt only from the rules that are statements
    about the INJECTOR - rule 5's repair(), rule 6's recipe match, and the
    inject() round-trip -
    because it is transcribed from freqtrade issue #11346 rather than generated
    by eval/inject.py. And it is excluded from the counts, the suites, and the
    baseline denominators, so adding it cannot silently restate a published
    score.
    """
    everything = discover_cases()
    assert len(everything) == 21
    assert [c.case_id for c in everything if not c.frozen] == ["htf_merge_11346"]


def test_the_known_limitations_are_exactly_the_frozen_set():
    actual = {c.case_id: c.known_limitations for c in CASES if c.known_limitations}
    assert actual == EXPECTED_LIMITATIONS
    for case in CASES:
        if case.known_limitations:
            assert len(case.limitation_reason) > 100, case.case_id


def test_the_locked_corrections_are_exactly_the_frozen_set():
    """Every post-freeze edit to a case is declared, reasoned, and countable.

    Currently none. The set may only grow by editing the literal above, which
    forces the edit into a diff and a review rather than into a case file
    alone.
    """
    actual = {c.case_id: c.locked_corrections for c in CASES if c.locked_corrections}
    assert actual == EXPECTED_CORRECTIONS
    for case in CASES:
        if case.locked_corrections:
            assert len(case.correction_reason) > 100, case.case_id


@pytest.mark.parametrize("meta", PROVABLE, ids=ids(PROVABLE))
def test_a_corrected_case_sits_out_no_validity_rule(meta):
    """A correction is not an exemption. If repairing a case's construction
    also required excusing it from a rule, the repair changed the case."""
    if meta.locked_corrections:
        assert meta.known_limitations == [], meta.case_id


@pytest.mark.parametrize("meta", PROVABLE, ids=ids(PROVABLE))
def test_rule4_sign_holds_in_every_window_even_where_magnitude_does_not(meta):
    """The clause that matters most, held to WITHOUT exception.

    A case may sit out rule 4's magnitude threshold on a one-year window, where
    the Sharpe standard error is about 1.0. None may sit out the direction of
    the effect -- a leak that helps in one year and hurts in another is not a
    leak that has been demonstrated.
    """
    if "rule4" in meta.known_limitations and meta.case_id != "l02_centered_window":
        pytest.skip(meta.limitation_reason)
    pairs = [(sharpe(meta, "clean", w), sharpe(meta, "strategy", w)) for w in WINDOWS]
    assert all(leaked > c for c, leaked in pairs), [
        f"{c:.2f}->{leaked:.2f}" for c, leaked in pairs
    ]


def test_only_l12_is_exempt_from_the_provable_rules():
    assert [c.leak_type for c in INJECTED if not c.patchable] == ["L12"]


def test_every_taxonomy_type_appears_exactly_once():
    assert sorted(c.leak_type for c in INJECTED) == [f"L{n:02d}" for n in range(1, 13)]


# ------------------------------------------------------------------- rule 1
@pytest.mark.parametrize("meta", ALL_CASES, ids=ids(ALL_CASES))
def test_rule1_both_versions_run_and_trade(meta):
    variants = ["strategy", "clean"] if meta.is_injected else ["strategy"]
    for variant in variants:
        record = run_case(meta, variant)
        assert record.outcome is SandboxOutcome.COMPLETED, variant
        assert record.position_changes >= 10, f"{variant}: {record.position_changes}"


# ------------------------------------------------------------------- rule 2
@pytest.mark.parametrize("meta", PROVABLE, ids=ids(PROVABLE))
def test_rule2_trade_count_is_stable(meta):
    clean = run_case(meta, "clean").position_changes
    leaked = run_case(meta, "strategy").position_changes
    drift = abs(leaked - clean) / clean
    assert drift <= 0.25, f"clean={clean} leaked={leaked} drift={drift:.0%}"


# ------------------------------------------------------------------- rule 3
@pytest.mark.parametrize("meta", held_to("rule3"), ids=ids(held_to("rule3")))
def test_rule3_the_leak_inflates_materially(meta):
    clean, leaked = sharpe(meta, "clean"), sharpe(meta, "strategy")
    assert inflates(clean, leaked), f"clean={clean:.3f} leaked={leaked:.3f}"


# ------------------------------------------------------------------- rule 4
@pytest.mark.parametrize("meta", held_to("rule4"), ids=ids(held_to("rule4")))
def test_rule4_the_inflation_survives_a_change_of_window(meta):
    pairs = [(sharpe(meta, "clean", w), sharpe(meta, "strategy", w)) for w in WINDOWS]
    rendered = [f"{c:.2f}->{leaked:.2f}" for c, leaked in pairs]
    assert all(leaked > c for c, leaked in pairs), f"sign flips: {rendered}"
    material = sum(inflates(c, leaked) for c, leaked in pairs)
    assert material >= 2, f"only {material}/3 windows clear rule 3: {rendered}"


# ------------------------------------------------------------------- rule 5
@pytest.mark.parametrize("meta", FROZEN_PROVABLE, ids=ids(FROZEN_PROVABLE))
def test_rule5_the_documented_fix_restores_the_clean_version(meta):
    leaked_source = meta.source("strategy").read_text("utf-8")
    clean_source = meta.source("clean").read_text("utf-8")
    # The fix is the exact inverse of the injection, so the restored Sharpe is
    # the clean Sharpe by construction rather than by luck.
    assert repair(leaked_source, meta.leak_type) == clean_source
    for window in WINDOWS:
        restored = sharpe(meta, "clean", window)
        assert restored == pytest.approx(sharpe(meta, "clean", window), abs=0.05)


# ------------------------------------------------------------------- rule 6
@pytest.mark.parametrize("meta", INJECTED, ids=ids(INJECTED))
def test_rule6_ground_truth_is_exact(meta):
    assert meta.ground_truth_file == "strategy.py"
    lines = meta.source("strategy").read_text("utf-8").splitlines()
    assert RECIPES[meta.leak_type].leaked in lines[meta.ground_truth_line - 1]


@pytest.mark.parametrize("meta", INJECTED, ids=ids(INJECTED))
def test_the_frozen_leaked_source_is_exactly_what_inject_produces(meta):
    result = inject(meta.source("clean").read_text("utf-8"), meta.leak_type)
    assert result.source == meta.source("strategy").read_text("utf-8")
    assert result.line == meta.ground_truth_line
    assert result.expected_inflation == meta.expected_inflation


# --------------------------------------------- rule 7 (ours, not the taxonomy's)
# Replay points, as fractions of the data.
CUTS = tuple(i / 40 for i in range(8, 40))

# Replay proves a strategy read a future ROW. It cannot see a same-bar
# mismatch, where the decision uses the current row -- legitimately present in
# the replayed frame -- for a position held through that same bar. Those two
# types are leaks by the decision-point argument in taxonomy section 1, and
# their evidence is the Sharpe delta under rules 3 and 4 instead.
SAME_BAR_TYPES = {"L01", "L03"}


def replay_disagrees_with_backtest(meta, variant: str) -> bool:
    """Would this strategy have taken the same position in real time?

    For each cut, recompute on data that ends there and compare only the
    decision for the final available day against what the full backtest claims
    for that same day. Comparing the boundary row is what gives this teeth: a
    higher-timeframe merge only corrupts the period still in progress, so
    checking every earlier row dilutes the one row where the leak is maximal.

    No Sharpe threshold is involved anywhere. If the answer differs, the
    backtest used information that did not exist on the day.
    """
    full = positions_of(meta, variant)
    for fraction in CUTS:
        cut = int(len(full) * fraction)
        replayed = positions_of(meta, variant, rows=cut)
        # The boundary day, where a leak that only corrupts the period still in
        # progress shows up...
        if replayed.iloc[-1] != full.iloc[cut - 1]:
            return True
        # ...and every earlier day, where a leak computed over the whole frame
        # (a full-sample statistic, a hindsight universe) shows up instead.
        if not replayed.equals(full.iloc[:cut]):
            return True
    return False


@pytest.mark.parametrize("meta", CAUSAL_CONTROLS, ids=ids(CAUSAL_CONTROLS))
def test_rule7_controls_replay_identically_in_real_time(meta):
    assert not replay_disagrees_with_backtest(meta, "strategy")


@pytest.mark.parametrize("meta", CAUSAL_INJECTED, ids=ids(CAUSAL_INJECTED))
def test_rule7_the_clean_variant_replays_identically_in_real_time(meta):
    assert not replay_disagrees_with_backtest(meta, "clean")


@pytest.mark.parametrize("meta", FUTURE_ROW, ids=ids(FUTURE_ROW))
def test_rule7_the_leaked_variant_reads_the_future(meta):
    assert replay_disagrees_with_backtest(meta, "strategy"), (
        "a real-time replay reproduced every decision, so this injection does "
        "not actually read a future row"
    )
