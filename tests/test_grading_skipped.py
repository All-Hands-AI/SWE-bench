"""Tests for skipped-test grading (#545, PR #598).

A skipped FAIL_TO_PASS test is not a resolution; a skipped PASS_TO_PASS test is
not a regression. Before this, skipped tests landed in neither the success nor
failure list, so an all-skipped F2P set scored RESOLVED_FULL off an empty ratio.
"""

from swebench.harness.constants import (
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.grading import test_failed as grade_failed
from swebench.harness.grading import test_maintained as grade_maintained

PASSED = TestStatus.PASSED.value
FAILED = TestStatus.FAILED.value
SKIPPED = TestStatus.SKIPPED.value


def _gold(f2p, p2p):
    return {
        FAIL_TO_PASS: f2p,
        PASS_TO_PASS: p2p,
        FAIL_TO_FAIL: [],
        PASS_TO_FAIL: [],
    }


def test_skipped_f2p_is_a_failure():
    assert grade_failed("t", {"t": SKIPPED})


def test_skipped_p2p_is_maintained():
    assert grade_maintained("t", {"t": SKIPPED})
    assert grade_maintained("t", {"t": PASSED})
    assert not grade_maintained("t", {"t": FAILED})


def test_all_skipped_f2p_is_not_resolved():
    """Regression: skipping every F2P test used to score RESOLVED_FULL."""
    sm = {"t_a": SKIPPED, "t_b": SKIPPED, "t_c": PASSED}
    report = get_eval_tests_report(
        sm, _gold(["t_a", "t_b"], ["t_c"]), eval_type=EvalType.PASS_AND_FAIL
    )
    assert report[FAIL_TO_PASS]["success"] == []
    assert report[FAIL_TO_PASS]["failure"] == ["t_a", "t_b"]
    assert get_resolution_status(report) == ResolvedStatus.NO.value


def test_skipped_p2p_does_not_block_resolution():
    sm = {"t_a": PASSED, "t_c": SKIPPED}
    report = get_eval_tests_report(
        sm, _gold(["t_a"], ["t_c"]), eval_type=EvalType.PASS_AND_FAIL
    )
    assert report[PASS_TO_PASS]["success"] == ["t_c"]
    assert get_resolution_status(report) == ResolvedStatus.FULL.value
