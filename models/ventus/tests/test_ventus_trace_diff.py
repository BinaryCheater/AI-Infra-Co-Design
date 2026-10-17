from copy import deepcopy

from codesign.ventus.ir import Operation, Workload
from codesign.ventus.trace_diff import compare_trace


def inputs():
    workload = Workload(
        tuple(
            Operation(f"w{warp}.pc80000000", "vector", warp=warp, destination="v1")
            for warp in range(2)
        ),
        warps_per_block=2,
    )
    prediction = {
        "first_collect": 5,
        "instruction_events": {
            "w0.pc80000000": {"collect": 5, "writeback": 10},
            "w1.pc80000000": {"collect": 8, "writeback": 13},
        },
    }
    rows = [
        ["105", "collect", "0", "80000000", "1234"],
        ["108", "collect", "1", "80000000", "1234"],
        ["110", "writeback", "0", "80000000", "1234"],
        ["113", "writeback", "1", "80000000", "1234"],
    ]
    return workload, [0x1234], prediction, rows


def test_one_global_alignment_preserves_warp_skew():
    args = inputs()
    assert compare_trace(*args)["status"] == "match"
    args[3][1][0] = "109"
    args[3][3][0] = "114"
    report = compare_trace(*args)
    assert report["first_divergence"]["warp"] == 1
    assert report["first_divergence"]["signed_residual"] == -1
    assert all(row["signed_residual"] == 0 for row in report["instruction_services"])


def test_missing_writeback_does_not_pass_and_wrong_word_prevents_alignment():
    args = inputs()
    args[3].pop()
    report = compare_trace(*args)
    assert report["status"] == "incomplete_events"
    assert len(report["missing_events"]) == 1
    args[3][0][4] = "4321"
    report = compare_trace(*args)
    assert report["status"] == "incompatible_trace"
    assert report["first_divergence"] is None


def test_duplicate_or_extra_warp_never_collapses_to_matching_trace():
    args = inputs()
    args[3].append(deepcopy(args[3][0]))
    assert compare_trace(*args)["status"] == "incompatible_trace"
    args[3][-1][2] = "2"
    assert compare_trace(*args)["status"] == "incompatible_trace"


def test_equal_final_writeback_does_not_hide_service_difference():
    args = inputs()
    args[3][1][0] = "109"
    report = compare_trace(*args)
    assert report["status"] == "different"
    assert report["instruction_services"][1]["signed_residual"] == 1
