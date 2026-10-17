"""Exact Python/native frontend parity and same-host fresh-state measurements.

This compares implementations of one model, not predictions against RTL. Frozen
RTL accuracy receipts remain separate. All cases, including exceptions and timing
differences, are retained; a faster mismatching result is never a speedup claim.
"""

import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

from .__main__ import read_candidate
from .config import Hardware
from .frontend import FRONTEND_VERSION, predict_span
from .graph import execute
from .physical_operands import with_physical_operands
from .termination import append_endprg
from .timing import build_graph

TRACE_EVENTS = ("collect", "operands", "issue", "writeback", "endprg")
MECHANISM_COUNTERS = (
    "frontend_packets",
    "frontend_replays",
    "rf_duplicate_grants",
    "lsu_mshr_idle_cycles",
    "lsu_mshr_add_cycles",
    "lsu_mshr_out_cycles",
    "lsu_mshr_returns",
    "lsu_mshr_deferred_allocations",
)


def python_reference(workload, hardware, *, endpoint_names):
    """Capture named operation edges without inferring them from final spans."""
    graph = build_graph(workload, hardware, frontend=True)
    result = execute(graph)
    times = dict(zip(graph.names, result["times"], strict=True))
    instruction_events = {
        op.name: {
            event: times[f"{op.name}.{event}"]
            for event in TRACE_EVENTS
            if f"{op.name}.{event}" in times
        }
        for op in workload.operations
    }
    first = min(events["collect"] for events in instruction_events.values())
    endpoints = {name: instruction_events[name]["writeback"] for name in endpoint_names}
    return {
        "model_version": FRONTEND_VERSION,
        "prediction_cycles": max(endpoints.values()) - first,
        "first_collect": first,
        "endpoint_times": endpoints,
        "instruction_events": instruction_events,
        "counters": result["counters"],
    }


def compare(reference, actual):
    """Return every supported observable mismatch; zero and absent counts agree."""
    differences = []
    for field in ("model_version", "prediction_cycles", "first_collect", "endpoint_times"):
        if reference[field] != actual.get(field):
            differences.append(
                {"field": field, "python": reference[field], "native": actual.get(field)}
            )
    expected_events = reference["instruction_events"]
    actual_events = actual.get("instruction_events", {})
    for name in sorted(expected_events.keys() | actual_events.keys()):
        for event in TRACE_EVENTS:
            expected = expected_events.get(name, {}).get(event)
            observed = actual_events.get(name, {}).get(event)
            if expected != observed:
                differences.append(
                    {"field": f"{name}.{event}", "python": expected, "native": observed}
                )
    for counter in sorted(reference["counters"].keys() | actual.get("counters", {}).keys()):
        expected = reference["counters"].get(counter, 0)
        observed = actual.get("counters", {}).get(counter, 0)
        if expected != observed:
            differences.append(
                {"field": f"counters.{counter}", "python": expected, "native": observed}
            )
    return differences


def frozen_cases(path, banks):
    """Read the exact manifest; never regenerate or overwrite a frozen input."""
    path = Path(path)
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for case in json.loads(path.read_text()):
        hardware = Hardware().with_changes(rf_banks=banks)
        workload = read_candidate({"workload": case["workload"]}).workload
        workload = with_physical_operands(append_endprg(workload, case["words"]), case["words"])
        endpoints = [f"w{warp}.pc{case['final_tensor_pc']:08x}" for warp in range(case["warps"])]
        yield {
            "id": f"{path.parent.name}-rf{banks}-{case['name']}",
            "manifest_sha256": manifest_hash,
            "hardware": hardware,
            "workload": workload,
            "endpoint_names": endpoints,
        }


def measure(cases, native_predict, *, repetitions=3, prepare_native=None):
    """Measure full-call native and Python summaries after checking detailed parity.

    The caller must make an initial native call before this function and record
    that cost separately. Each callable must start with fresh modeled hardware.
    Native encoding, repeated bridge/library access, source hashing, FFI and
    result decoding are included in the timed full call.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    rows = []
    for case in cases:
        row = {key: case[key] for key in ("id", "manifest_sha256") if key in case}
        row["hardware"] = case["hardware"].to_dict()
        kwargs = {"endpoint_names": case["endpoint_names"]}
        try:
            reference = python_reference(case["workload"], case["hardware"], **kwargs)
            actual = native_predict(case["workload"], case["hardware"], trace=True, **kwargs)
            differences = compare(reference, actual)
            row.update(
                status="fail" if differences else "pass",
                differences=differences,
                operations=len(case["workload"].operations),
                prediction_cycles=reference["prediction_cycles"],
                compared_instruction_events=sum(
                    len(events) for events in reference["instruction_events"].values()
                ),
                mechanism_counters={
                    name: reference["counters"].get(name, 0) for name in MECHANISM_COUNTERS
                },
                counters=reference["counters"],
                native_identity={
                    name: actual.get(name)
                    for name in (
                        "native_version",
                        "native_source_sha256",
                        "native_runtime_sha256",
                        "prepared_input_sha256",
                    )
                },
            )
            python_samples, native_samples, prepared_samples = [], [], []
            prepared = None
            if prepare_native is not None:
                before = perf_counter()
                prepared = prepare_native(case["workload"], case["hardware"])
                row["native_prepare_seconds"] = perf_counter() - before
            for repeat in range(repetitions):
                # Alternate order to avoid giving one runtime a fixed warmup advantage.
                calls = [("python", predict_span), ("native", native_predict)]
                for name, function in calls if repeat % 2 == 0 else reversed(calls):
                    start = perf_counter()
                    output = function(case["workload"], case["hardware"], **kwargs)
                    elapsed = perf_counter() - start
                    (python_samples if name == "python" else native_samples).append(elapsed)
                    for field in ("prediction_cycles", "first_collect", "endpoint_times"):
                        if output.get(field) != reference[field]:
                            row["status"] = "fail"
                            row["differences"].append(
                                {
                                    "field": f"{name}.repeat{repeat}.{field}",
                                    "python": reference[field],
                                    "actual": output.get(field),
                                }
                            )
                if prepared is not None:
                    before = perf_counter()
                    output = native_predict(prepared, case["hardware"], **kwargs)
                    prepared_samples.append(perf_counter() - before)
                    for field in ("prediction_cycles", "first_collect", "endpoint_times"):
                        if output.get(field) != reference[field]:
                            row["status"] = "fail"
                            row["differences"].append(
                                {
                                    "field": f"prepared.repeat{repeat}.{field}",
                                    "python": reference[field],
                                    "actual": output.get(field),
                                }
                            )
            if prepared_samples:
                row.update(
                    native_prepared_seconds=prepared_samples,
                    native_prepared_median_seconds=median(prepared_samples),
                    prepared_speedup=median(python_samples) / median(prepared_samples)
                    if row["status"] == "pass"
                    else None,
                    prepared_scope="encoding reused; fresh native hardware and full execution per call",
                )
            row.update(
                python_seconds=python_samples,
                native_seconds=native_samples,
                python_median_seconds=median(python_samples),
                native_median_seconds=median(native_samples),
                speedup=median(python_samples) / median(native_samples)
                if row["status"] == "pass"
                else None,
            )
        except Exception as error:
            row.update(status="open", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    return {
        "schema": "ventus-frontend-native-parity-v1",
        "model_version": FRONTEND_VERSION,
        "scope": "implementation parity; does not establish additional RTL accuracy",
        "timing_scope": (
            "fresh hardware per call; encoding, repeated bridge/library access, source hashing, "
            "FFI and decoding included; initial build/first call excluded and recorded separately"
        ),
        "python_runtime": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "repetitions": repetitions,
        "counter_scope": "all Python and native counters, absent entries interpreted as zero",
        "mechanism_counter_names": list(MECHANISM_COUNTERS),
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "passes": sum(row["status"] == "pass" for row in rows),
            "failures": sum(row["status"] == "fail" for row in rows),
            "open": sum(row["status"] == "open" for row in rows),
            "all_cases_match": bool(rows) and all(row["status"] == "pass" for row in rows),
        },
    }


def main(argv=None):
    import argparse

    from . import rust_frontend

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--banks", type=int, choices=(4, 8), action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    # Refuse overwrite before executing either predictor.
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [
        case for path in args.cases for bank in args.banks for case in frozen_cases(path, bank)
    ]
    # One complete warm-up call makes the measured scope independent of library
    # build/load. Its preparation and execution cost is recorded without claiming
    # that the entire call is library initialization.
    before = perf_counter()
    warmup_error = None
    try:
        first = cases[0]
        rust_frontend.predict_span(
            first["workload"], first["hardware"], endpoint_names=first["endpoint_names"]
        )
    except Exception as error:
        warmup_error = f"{type(error).__name__}: {error}"
    warmup_seconds = perf_counter() - before
    result = measure(
        cases,
        rust_frontend.predict_span,
        repetitions=args.repetitions,
        prepare_native=rust_frontend.PreparedFrontend.from_workload,
    )
    result["initial_native_call"] = {"seconds": warmup_seconds, "error": warmup_error}
    result["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    (args.out / "parity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
    if not result["summary"]["all_cases_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
