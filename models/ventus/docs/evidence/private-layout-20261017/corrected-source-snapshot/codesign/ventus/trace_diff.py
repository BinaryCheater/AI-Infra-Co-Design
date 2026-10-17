"""Locate observable frontend/RTL divergence in a frozen straight-line probe.

This is diagnostic replay, not a new accuracy certificate. Matching collect
streams is required before comparing clocks; the first *observed* divergence
does not establish the first internal cause. RTL completion and hardware identity
remain separate evidence obligations.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .__main__ import read_candidate
from .config import Hardware
from .frontend_parity import python_reference
from .physical_operands import with_physical_operands
from .termination import append_endprg


def compare_trace(workload, words, prediction, rows, *, base_pc=0x80000000):
    """Compare collect/writeback edges after one global first-collect alignment.

    Never align warps independently: that would hide instruction-supply skew.
    Reject duplicate events instead of collapsing repeated PCs into a dictionary.
    Other observer events are outside this comparator's scope.
    """
    if {op.block for op in workload.operations} != {0}:
        raise ValueError("observer comparison currently requires one block")
    expected = {}
    streams = defaultdict(list)
    for op in workload.operations:
        pc = int(op.name.rsplit("pc", 1)[1], 16)
        index, remainder = divmod(pc - base_pc, 4)
        if remainder or not 0 <= index < len(words):
            raise ValueError("operation PC is outside the supplied machine words")
        key = op.warp, pc
        if key in expected:
            raise ValueError("repeated PC requires dynamic instruction identities")
        expected[key] = op
        streams[op.warp].append((pc, words[index]))
    observed, actual_streams, errors = {}, defaultdict(list), []
    for line, row in enumerate(rows, 1):
        if len(row) != 5:
            errors.append({"line": line, "reason": "expected five observer CSV fields"})
            continue
        if row[1] not in {"collect", "writeback"}:
            continue
        try:
            cycle, warp, pc, word = int(row[0]), int(row[2]), int(row[3], 16), int(row[4], 16)
            if min(cycle, warp, pc, word) < 0:
                raise ValueError
        except ValueError:
            errors.append({"line": line, "reason": "invalid observer integer"})
            continue
        event = row[1]
        key = warp, pc, event
        if key in observed:
            errors.append({"line": line, "reason": "duplicate instruction event", "key": key})
        observed[key] = cycle
        op = expected.get((warp, pc))
        if op is None or words[(pc - base_pc) // 4] != word:
            errors.append({"line": line, "reason": "unexpected warp, PC or instruction word"})
        if event == "collect":
            actual_streams[warp].append((pc, word))
    for warp in sorted(streams.keys() | actual_streams.keys()):
        if streams[warp] != actual_streams[warp]:
            errors.append({"warp": warp, "reason": "collected instruction sequence differs"})
    if errors:
        return {"status": "incompatible_trace", "errors": errors, "first_divergence": None}
    rtl_origin = min(t for (_, _, event), t in observed.items() if event == "collect")
    model_origin = prediction["first_collect"]
    differences, services, missing = [], [], []
    for (warp, pc), op in expected.items():
        events = prediction["instruction_events"][op.name]
        for event in ("collect", "writeback"):
            model, rtl = events.get(event), observed.get((warp, pc, event))
            if model is None and rtl is None:
                continue
            identity = {"name": op.name, "kind": op.kind, "warp": warp, "pc": pc, "event": event}
            if model is None or rtl is None:
                missing.append({**identity, "model_cycle": model, "rtl_cycle": rtl})
                continue
            model, rtl = model - model_origin, rtl - rtl_origin
            if model != rtl:
                differences.append(
                    {
                        **identity,
                        "model_cycle": model,
                        "rtl_cycle": rtl,
                        "signed_residual": model - rtl,
                    }
                )
        if "writeback" in events and (warp, pc, "writeback") in observed:
            model = events["writeback"] - events["collect"]
            rtl = observed[warp, pc, "writeback"] - observed[warp, pc, "collect"]
            services.append(
                {
                    "name": op.name,
                    "kind": op.kind,
                    "model_cycles": model,
                    "rtl_cycles": rtl,
                    "signed_residual": model - rtl,
                }
            )
    differences.sort(
        key=lambda d: (min(d["model_cycle"], d["rtl_cycle"]), d["warp"], d["pc"], d["event"])
    )
    return {
        "status": "incomplete_events" if missing else "different" if differences else "match",
        "alignment": {"model_first_collect": model_origin, "rtl_first_collect": rtl_origin},
        "first_divergence": differences[0] if differences else None,
        "differences": differences,
        "missing_events": missing,
        "instruction_services": services,
        "compared_operations": len(expected),
        "scope": "Single-block straight-line collect/writeback; no completion or accuracy certification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--banks", type=int, choices=(4, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    matches = [case for case in json.loads(args.cases.read_text()) if case["name"] == args.case]
    if len(matches) != 1:
        raise ValueError("case must identify exactly one frozen input")
    case = matches[0]
    hardware = Hardware().with_changes(rf_banks=args.banks)
    workload = read_candidate({"workload": case["workload"]}).workload
    workload = with_physical_operands(append_endprg(workload, case["words"]), case["words"])
    endpoints = [f"w{warp}.pc{case['final_tensor_pc']:08x}" for warp in range(case["warps"])]
    prediction = python_reference(workload, hardware, endpoint_names=endpoints)
    with args.trace.open() as stream:
        result = compare_trace(workload, case["words"], prediction, csv.reader(stream))
    result.update(
        schema="ventus.frontend.trace-diff.v1",
        case=args.case,
        hardware=hardware.to_dict(),
        hardware_identity_status="caller_declared_not_rtl_verified",
        model_version=prediction["model_version"],
        prediction_cycles=prediction["prediction_cycles"],
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        python_source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("case", "status", "first_divergence")}))


if __name__ == "__main__":
    main()
