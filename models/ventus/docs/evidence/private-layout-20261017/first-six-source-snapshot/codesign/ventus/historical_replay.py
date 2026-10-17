"""Replay current default timing against immutable historical mixed-kernel RTL.

This is new prediction evidence against old traces, not a fresh RTL run or a
holdout study. Frozen archive integrity is checked before replay; current code is recorded separately.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from ventus_project.config import ROOT

from . import MODEL_VERSION
from .__main__ import model_hashes, read_candidate
from .accuracy_audit import audit
from .graph import execute
from .rtl_diagnostics import classify_rtl_logs
from .timing import build_graph


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def native_speed(candidate, expected_cycles):
    """Time fresh native sessions; verify only equivalent default-model terminal."""
    from .rust import RUST_VERSION, PreparedInstructions, RustSession, library

    begin = perf_counter()
    library()
    runtime_load_seconds = perf_counter() - begin
    begin = perf_counter()
    prepared = PreparedInstructions.from_workload(candidate.workload, candidate.hardware)
    encode_seconds = perf_counter() - begin
    durations = []
    for _ in range(5):
        begin = perf_counter()
        with RustSession(candidate.hardware) as session:
            result = session.dispatch(prepared)
        durations.append(perf_counter() - begin)
        if result["cycles"] != expected_cycles:
            raise ValueError("native/default graph terminal disagreement")
    return {
        "model_version": RUST_VERSION,
        "native_graph_terminal_agrees": True,
        "terminal_cycles": expected_cycles,
        "runtime_load_seconds": runtime_load_seconds,
        "encode_seconds": encode_seconds,
        "fresh_session_seconds_samples": durations,
        "fresh_session_median_seconds": median(durations),
        "scope": "prepared input, fresh hardware state per call; initialization and execution included; no frontend, no new RTL accuracy claim",
    }


def replay(out, *, root=ROOT, benchmark_native=False):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "archive/export-manifest.json").read_text())
    entries = [r for r in manifest["files"] if r["path"].startswith("archive/")]
    if not entries:
        raise ValueError("missing historical export manifest entries")
    for entry in entries:
        path = root / entry["path"]
        if not path.resolve().is_relative_to((root / "archive").resolve()):
            raise ValueError("archive manifest path escapes archive")
        if digest(path) != entry["export_sha256"]:
            raise ValueError(f"Historical artifact changed: {entry['path']}")
    integrity = {
        "archive_integrity": True,
        "files": len(entries),
        "scope": "archive only; current model code intentionally differs from original export",
    }
    flow = root / "archive/analysis/ventus_flow_20261006"
    identities = json.loads((flow / "search_model/kernel-provenance-v3.json").read_text())["hashes"]
    expected_identity = {
        "rtl_binary_sha256": identities[
            "${REMOTE_PROJECT_ROOT}/sim-verilator/build/driver_example/debug/sim-VentusRTL"
        ],
        "rtl_library_sha256": identities[
            "${REMOTE_PROJECT_ROOT}/sim-verilator/build/libVentusRTL/debug/libVentusRTL.so"
        ],
        "trace_tool_sha256": identities["${REMOTE_FLOW_ROOT}/trace_rtl-v4.so"],
    }
    rows = []
    for suite, source, observed in (
        ("quick_complete_20261006", "probes", "rtl"),
        ("performance-v7", "memory-probes", "memory-rf4-v7"),
        ("performance-v7", "combined-probes", "combined-rf4-v7"),
    ):
        directory = flow / suite
        frozen_path = directory / source / "pre_run_predictions.json"
        frozen = json.loads(frozen_path.read_text())
        for case in frozen["cases"]:
            case_id = f"{suite}/{source}/{case['name']}"
            case_dir = directory / observed / case["name"]
            common = {
                "family": "gemm" if source == "memory-probes" else "mixed-kernel",
                "split": "calibration",
                "provenance": "historical-rtl",
                "model_version": MODEL_VERSION,
                "completed": False,
                "functional_ok": False,
                "frozen_inputs_sha256": digest(frozen_path),
                "source_hashes": model_hashes(),
                "endpoint_scope": "first instruction collection to selected endpoint; excludes launch",
            }
            try:
                common["reference_sha256"] = digest(case_dir / "summary.json")
                common["trace_sha256"] = digest(case_dir / "trace.csv")
                for name, sha in case["input_sha256"].items():
                    if digest(directory / source / case["name"] / name) != sha:
                        raise ValueError("frozen input hash mismatch")
                raw = json.loads((case_dir / "summary.json").read_text())
                if raw["returncode"] or raw["warnings"]:
                    raise ValueError("RTL run diagnostic failure")
                if raw["output_words"] != case["expected_words"]:
                    raise ValueError("RTL output differs from frozen numerical reference")
                diagnostics = classify_rtl_logs(
                    (case_dir / "stdout.log").read_text(errors="replace"),
                    (case_dir / "stderr.log").read_text(errors="replace"),
                    kernel_name=case["name"],
                    returncode=raw["returncode"],
                    timed_out=raw.get("timed_out", False),
                )
                common["diagnostics"] = diagnostics
                common["stdout_sha256"] = digest(case_dir / "stdout.log")
                common["stderr_sha256"] = digest(case_dir / "stderr.log")
                if not diagnostics["kernel_completed"]:
                    raise ValueError("exact named kernel completion not established")
                if any(raw[k] != v for k, v in expected_identity.items()):
                    raise ValueError("RTL binary/observer identity mismatch")
                if raw["launch_not_before_cycle"] != 512:
                    raise ValueError("launch identity mismatch")
                with (case_dir / "trace.csv").open() as stream:
                    trace = list(csv.reader(stream))
                collected = [e for e in trace if e[1] == "collect"]
                warps = case.get("warps", 1)
                if {int(e[2]) for e in collected} != set(range(warps)):
                    raise ValueError("unexpected warp set")
                for warp in range(warps):
                    actual = [
                        (int(e[3], 16), int(e[4], 16)) for e in collected if int(e[2]) == warp
                    ]
                    expected = [(0x80000000 + 4 * i, w) for i, w in enumerate(case["words"])]
                    if actual != expected:
                        raise ValueError("actual collected instruction stream differs")
                start = min(int(e[0]) for e in collected)
                ends = [
                    int(e[0]) - start
                    for e in trace
                    if e[1] == "writeback" and int(e[3], 16) == case["final_pc"]
                ]
                if len(ends) != warps:
                    raise ValueError("missing/ambiguous compute endpoint")
                stores = [
                    e
                    for e in trace
                    if e[1] == "memory.write"
                    and 0x90002000 <= int(e[3], 16) < 0x90002000 + len(case["expected_words"]) * 4
                ]
                if {int(e[3], 16) for e in stores} != {0x90002000 + 128 * w for w in range(warps)}:
                    raise ValueError("missing output visibility endpoint")
                common.update(
                    completed=True, functional_ok=True, rtl_host_seconds=raw["wall_seconds"]
                )
                candidate = read_candidate({"workload": case["workload"]})
                durations = []
                for _ in range(3):
                    before = perf_counter()
                    graph = build_graph(candidate.workload, candidate.hardware)
                    prediction = execute(graph)
                    durations.append(perf_counter() - before)
                names = (
                    [f"w{w}.pc{case['final_pc']:08x}.writeback" for w in range(warps)]
                    if warps > 1
                    else [f"pc{case['final_pc']:08x}.writeback"]
                )
                predicted_compute = max(prediction["times"][graph.names.index(n)] for n in names)
                predicted_visible = prediction["times"][graph.names.index("outputs.visible")]
                common.update(
                    host_seconds=median(durations),
                    prediction_host_seconds_samples=durations,
                    speed_scope="graph construction and execution; input parsing excluded; 3 same-process uncached calls",
                    hardware_config=vars(candidate.hardware),
                )
                if benchmark_native:
                    try:
                        common["native_speed"] = native_speed(candidate, prediction["cycles"])
                    except (ValueError, RuntimeError, OSError, AttributeError) as error:
                        common["native_speed"] = {"status": "open", "error": str(error)}
                endpoints = [
                    ("compute", predicted_compute, max(ends)),
                    ("visible", predicted_visible, max(int(e[0]) - start for e in stores)),
                ]
                for endpoint, predicted, reference in endpoints:
                    row = {
                        **common,
                        "id": f"{case_id}/{endpoint}",
                        "endpoint": f"first-collect-to-{endpoint}",
                        "prediction_cycles": predicted,
                        "reference_cycles": reference,
                    }
                    if source == "memory-probes":
                        row.update(
                            ranking_group=f"equal-tensor-work-memory-pattern/{endpoint}",
                            workload_sha256=object_digest(
                                "four frozen equal-Tensor-work memory patterns"
                            ),
                            target_sha256=object_digest(expected_identity),
                            initial_state_sha256=object_digest({"launch": 512, "reset": True}),
                        )
                    rows.append(row)
            except (ValueError, KeyError, OSError) as error:
                for endpoint in ("compute", "visible"):
                    rows.append(
                        {
                            **common,
                            "id": f"{case_id}/{endpoint}",
                            "endpoint": f"first-collect-to-{endpoint}",
                            "reference_cycles": None,
                            "prediction_cycles": None,
                            "producer_error": str(error),
                        }
                    )
    (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    result = audit(rows)
    result["export_integrity"] = integrity
    result["producer_checks"] = [
        "export manifest",
        "input hashes",
        "binary and observer identities",
        "PC/instruction/warp streams",
        "output words",
        "exact named kernel completion and stdout/stderr diagnostics",
        "endpoint cardinality",
    ]
    result["study_scope"] = (
        "Historical mixed-kernel cases, two endpoints each when available; current default legacy timing, calibration replay only"
    )
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-native", action="store_true")
    args = parser.parse_args()
    replay(args.out, benchmark_native=args.benchmark_native)


if __name__ == "__main__":
    main()
