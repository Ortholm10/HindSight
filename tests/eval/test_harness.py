"""The harness must be able to detect a bad detector, not just a good one."""

import pytest

from eval.detectors import everything_detector, null_detector
from eval.harness import format_table, run_suite


@pytest.fixture(scope="module")
def null():
    return run_suite(null_detector, with_metrics=False)


@pytest.fixture(scope="module")
def everything():
    return run_suite(everything_detector, with_metrics=False)


def test_suite_selection_filters_cases():
    assert len(run_suite(null_detector, "injected", with_metrics=False).results) == 12
    assert len(run_suite(null_detector, "clean", with_metrics=False).results) == 8
    assert len(run_suite(null_detector, "all", with_metrics=False).results) == 20


def test_a_single_case_can_be_run_by_id():
    result = run_suite(null_detector, case_id="l03_same_bar_execution")
    assert [r.case_id for r in result.results] == ["l03_same_bar_execution"]


def test_unknown_case_id_is_rejected():
    with pytest.raises(ValueError, match="no_such_case"):
        run_suite(null_detector, case_id="no_such_case")


def test_null_detector_finds_nothing_and_accuses_nothing(null):
    assert null.detected == 0
    assert null.injected_total == 12
    assert null.false_positives == 0
    assert null.clean_total == 8
    assert null.localised == 0


def test_everything_detector_finds_every_leak(everything):
    assert everything.detected == 12
    assert everything.localised == 12


def test_everything_detector_accuses_every_clean_case(everything):
    assert everything.false_positives == 8


def test_everything_detector_has_near_zero_localisation_precision(everything):
    # Recall alone cannot separate a detector from a highlighter: flagging every
    # line hits the ground-truth line every time. Precision is what exposes it.
    assert everything.localisation_precision < 0.10
    assert everything.candidates_on_injected > 200


def test_null_detector_localisation_precision_is_zero_not_undefined(null):
    assert null.localisation_precision == 0.0


def test_table_reports_both_stubs_distinguishably(null, everything):
    assert "0/12" in format_table(null)
    assert "12/12" in format_table(everything)
    assert "8/8" in format_table(everything)
