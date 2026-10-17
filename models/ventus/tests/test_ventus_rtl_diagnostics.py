import pytest

from codesign.ventus.rtl_diagnostics import classify_rtl_logs


def classify(stdout="[RTL     info]@123 kernel0  probe finished", stderr="", **kwargs):
    return classify_rtl_logs(stdout, stderr, kernel_name="probe", returncode=0, **kwargs)


def test_exact_kernel_completion_and_duplicate_output_streams():
    assert classify()["kernel_completed"]
    line = "[RTL     info]@123 kernel0  probe finished"
    result = classify(line, line)
    assert result["kernel_completed"]
    assert len(result["kernel_completion_markers"]) == 2


@pytest.mark.parametrize(
    "line",
    [
        "[RTL     info]@123 Simulation finished in 123 unit time",
        "[RTL     info]@123 block0 finished (kernel0 probe block0)",
        "[RTL     info]@123 kernel1 probe finished",
        "[RTL     info]@123 kernel0 other-probe finished",
        "[RTL     info]@123 kernel0 probe finished unexpectedly",
    ],
)
def test_process_exit_or_wrong_completion_is_not_success(line):
    assert not classify(line)["kernel_completed"]


@pytest.mark.parametrize(
    "line",
    [
        "[RTL    error]@124 invalid request",
        "[2026-10-16] [critical] failed",
        "%Error: assertion failure",
        "Assertion `ready' failed.",
        "[RTL info]@124 Simulation exited ABNORMALLY at time 124",
        "simulation time limit exceeded",
        "maximum simulation time reached",
        "external timeout",
    ],
)
def test_stderr_failures_override_completed_marker(line):
    result = classify(stderr=line)
    assert not result["kernel_completed"]
    assert result["error_diagnostics"] or result["limit_diagnostics"]


def test_external_timeout_and_ambiguous_repeat_are_rejected():
    assert not classify(timed_out=True)["kernel_completed"]
    assert not classify(timed_out=None)["kernel_completed"]
    assert not classify(
        "[RTL info]@123 kernel0 probe finished\n[RTL info]@456 kernel0 probe finished"
    )["kernel_completed"]


def test_kernel_name_is_literal_and_nonzero_exit_is_rejected():
    result = classify_rtl_logs(
        "[RTL info]@123 kernel0 a.b finished", "", kernel_name="a.b", returncode=1
    )
    assert len(result["kernel_completion_markers"]) == 1
    assert not result["kernel_completed"]
    assert not classify_rtl_logs(
        "[RTL info]@123 kernel0 axb finished", "", kernel_name="a.b", returncode=0
    )["kernel_completed"]
