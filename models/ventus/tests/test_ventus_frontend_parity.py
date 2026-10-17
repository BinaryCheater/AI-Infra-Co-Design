"""The validation harness must retain mismatches and not grant speedup credit."""

from copy import deepcopy

from codesign.ventus.config import Hardware
from codesign.ventus.frontend_parity import (
    MECHANISM_COUNTERS,
    compare,
    measure,
    python_reference,
)
from codesign.ventus.ir import Operation, Workload


def case():
    return {
        "id": "minimal",
        "hardware": Hardware(),
        "workload": Workload(
            (Operation("pc80000000", "vector", destination="v1"), Operation("pc80000004", "endprg"))
        ),
        "endpoint_names": ["pc80000000"],
    }


def test_comparison_catches_internal_difference_despite_equal_endpoint():
    item = case()
    reference = python_reference(
        item["workload"], item["hardware"], endpoint_names=item["endpoint_names"]
    )
    actual = deepcopy(reference)
    for name in MECHANISM_COUNTERS:
        actual["counters"].setdefault(name, 0)
    assert compare(reference, actual) == []
    actual["instruction_events"]["pc80000000"]["collect"] += 1
    actual["counters"]["frontend_packets"] += 1
    differences = compare(reference, actual)
    assert {d["field"] for d in differences} == {"pc80000000.collect", "counters.frontend_packets"}


def test_exception_and_mismatch_stay_in_denominator_without_speedup():
    def failed(*args, **kwargs):
        raise RuntimeError("intentional unavailable backend")

    result = measure([case()], failed, repetitions=1)
    assert result["summary"] == {
        "cases": 1,
        "passes": 0,
        "failures": 0,
        "open": 1,
        "all_cases_match": False,
    }

    def mismatched(workload, hardware, *, endpoint_names, trace=False):
        result = python_reference(workload, hardware, endpoint_names=endpoint_names)
        for name in MECHANISM_COUNTERS:
            result["counters"].setdefault(name, 0)
        result["prediction_cycles"] += 1
        return result

    result = measure([case()], mismatched, repetitions=1)
    assert result["summary"]["failures"] == 1
    assert result["cases"][0]["speedup"] is None
