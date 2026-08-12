"""Results emitted outside the START/END markers must still be graded (#402).

`set -x` traces the marker no-ops to stderr, so depending on how a runner
interleaves stdout and stderr the results can land outside them. Slicing then
yields nothing and the run looks resultless -- fastlane__fastlane-21857 in
SWE-bench_Multilingual fails this way (208 chars inside the markers, 3 real
results outside them).
"""

from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from swebench.harness.grading import get_logs_eval
from swebench.types import TestSpec


def _spec(**kw):
    return TestSpec(
        **{
            "instance_id": "x__x-1",
            "image": "x:latest",
            "eval_script_list": [],
            "repo": "x/x",
            "version": "1.0",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
            "log_parser": "parse_log_pytest",
            "eval_type": "pass_and_fail",
            **kw,
        }
    )


def test_results_inside_markers_are_used(tmp_path):
    log = tmp_path / "test_output.txt"
    log.write_text(
        f"{START_TEST_OUTPUT}\nPASSED tests/test_a.py::test_in\n{END_TEST_OUTPUT}\n"
    )
    sm, found = get_logs_eval(_spec(), str(log))
    assert found and sm == {"tests/test_a.py::test_in": "PASSED"}


def test_results_outside_markers_fall_back_to_whole_log(tmp_path):
    """Empty between the markers, real results outside -> use the whole log."""
    log = tmp_path / "test_output.txt"
    log.write_text(
        f"PASSED tests/test_a.py::test_outside\n{START_TEST_OUTPUT}\n{END_TEST_OUTPUT}\n"
    )
    sm, found = get_logs_eval(_spec(), str(log))
    assert found and sm == {"tests/test_a.py::test_outside": "PASSED"}


def test_missing_markers_still_reports_not_found(tmp_path):
    log = tmp_path / "test_output.txt"
    log.write_text("PASSED tests/test_a.py::test_x\n")
    sm, found = get_logs_eval(_spec(), str(log))
    assert (sm, found) == ({}, False)
