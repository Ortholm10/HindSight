"""Two runs of the same case must produce the same numbers, to the last digit.

If this fails, every measured improvement in the changelog is unreproducible and
the eval is worthless as evidence.
"""

import json

import pytest

from eval.runner import WINDOWS, discover_cases, run_case

CASES = discover_cases()
IDS = [c.case_id for c in CASES]


def fingerprint(record):
    return json.dumps(
        {
            "outcome": record.outcome.value,
            "metrics": record.metrics,
            "position_changes": record.position_changes,
        },
        sort_keys=True,
    )


@pytest.mark.parametrize("meta", CASES, ids=IDS)
def test_two_consecutive_runs_are_byte_identical(meta):
    first = run_case(meta, "strategy")
    second = run_case(meta, "strategy")
    assert fingerprint(first) == fingerprint(second)
    assert first.run_id == second.run_id


@pytest.mark.parametrize("meta", CASES, ids=IDS)
def test_a_run_does_not_depend_on_what_ran_before_it(meta):
    """The seed is set per run, not once per process.

    A case that inherits RNG state from whatever executed before it produces
    different numbers depending on suite order, which is the same as not being
    deterministic at all.
    """
    alone = fingerprint(run_case(meta, "strategy"))
    for other in CASES:
        run_case(other, "strategy" if other.is_injected else "strategy")
        break
    for other in reversed(CASES):
        run_case(other, "strategy")
        break
    assert fingerprint(run_case(meta, "strategy")) == alone


def test_windowed_runs_are_also_deterministic():
    meta = next(c for c in CASES if c.case_id == "l10_random_split")
    for window in WINDOWS:
        assert fingerprint(run_case(meta, "strategy", window)) == fingerprint(
            run_case(meta, "strategy", window)
        )


def test_the_data_cache_is_pinned_and_offline():
    """The eval must never reach the network, and the range must never drift."""
    from eval.cache_data import END, START, load

    frame = load("SPY")
    assert str(frame.index[0].date()) == START
    assert str(frame.index[-1].date()) == END
    assert len(frame) == 753
