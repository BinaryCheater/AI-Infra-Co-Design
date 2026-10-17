"""Generate complete numerical GEMM probes, predict, and collect bounded RTL traces.

Use the configured execution wrapper. RTL paths come from explicit arguments or
private local.toml, never hard-coded owner infrastructure. New output directories
are mandatory; historical inputs/measurements are never overwritten.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from ventus_project.config import load_profile

from . import MODEL_VERSION
from .__main__ import model_hashes, read_candidate
from .config import Hardware
from .frontend import FRONTEND_VERSION
from .graph import execute
from .ir import Operation, Workload
from .physical_operands import with_physical_operands
from .program import packed_gemm, vector
from .rtl_diagnostics import classify_rtl_logs
from .termination import append_endprg
from .timing import build_graph


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def complete_gemm(steps, warps, strategy="resident", rf_conflict=False):
    """Retain every warp's epilogue so early output traffic competes with late MMA."""
    if type(warps) is not int or not 1 <= warps <= 8:
        raise ValueError("probe warps must be in [1,8]")
    program = packed_gemm(steps, strategy=strategy, rf_conflict=rf_conflict)
    count = (program.final_tensor_pc - 0x80000000) // 4 + 1
    words = list(program.words[:count])
    operations = list(program.workload.operations[:count])
    tail = [
        (0x90002137, "scalar", (), "x2"),
        (vector(0x14, 0, 17, 2, 14), "vector", (), "v14"),
        (vector(0x25, 14, 2, 3, 14), "vector", ("v14",), "v14"),
        (vector(0, 14, 2, 4, 14), "vector", ("v14", "x2"), "v14"),
        ((0x800 << 20) | (2 << 12) | (17 << 7) | 0x73, "scalar", ("x0",), "x17"),
        ((2 << 20) | (17 << 15) | (1 << 12) | (17 << 7) | 0x13, "scalar", ("x17",), "x17"),
        (vector(0, 14, 17, 4, 14), "vector", ("v14", "x17"), "v14"),
        ((1 << 20) | (14 << 15) | (6 << 12) | 0x7B, "store", ("v14", "v1"), None),
    ]
    for word, kind, sources, destination in tail:
        operations.append(
            Operation(
                f"pc{0x80000000 + 4 * len(words):08x}",
                kind,
                sources=sources,
                destination=destination,
                addresses=tuple(0x90002000 + 4 * i for i in range(32)) if kind == "store" else (),
            )
        )
        words.append(word)
    words.append(0x400B)  # ENDPRG/flush is explicitly outside the timing IR.
    workload = Workload(
        tuple(
            replace(
                op,
                name=f"w{warp}.{op.name}",
                warp=warp,
                addresses=tuple(a + 128 * warp for a in op.addresses)
                if op.name == operations[-1].name
                else op.addresses,
            )
            for op in operations
            for warp in range(warps)
        ),
        warps_per_block=warps,
        lds_per_block=4096 if strategy == "shared" else 0,
    )
    return replace(
        program,
        words=tuple(words),
        workload=workload,
        expected=program.expected * warps,
        output_store_pc=0x80000000 + 4 * (len(words) - 2),
    )


def write_probe(program, path, fixture):
    program.write(path, fixture)
    lines = (path / "input.metadata").read_text().splitlines()
    meta = [int(lines[i], 16) | int(lines[i + 1], 16) << 32 for i in range(0, len(lines), 2)]
    data = [int(line, 16) for line in (path / "input.data").read_text().splitlines()]
    count = meta[13]
    sizes = meta[14 + count : 14 + 2 * count]
    pos = meta[14 : 14 + count].index(0x90002000)
    offset = sum(sizes[:pos]) // 4
    data[offset : offset + sizes[pos] // 4] = [0] * len(program.expected)
    meta[6] = program.workload.warps_per_block
    meta[14 + count + pos] = meta[14 + 2 * count + pos] = len(program.expected) * 4
    (path / "input.metadata").write_text(
        "".join(f"{v & 0xFFFFFFFF:08x}\n{v >> 32:08x}\n" for v in meta)
    )
    (path / "input.data").write_text("".join(f"{v:08x}\n" for v in data))
    return {name: sha(path / name) for name in ("input.metadata", "input.data")}


def generate(out, fixture, suite):
    out.mkdir(parents=True)
    cases = []
    # Calibration shape pairs are old evidence; holdout shapes are frozen before RTL.
    shapes = [(2, 8), (4, 8), (8, 8), (2, 32), (4, 32), (8, 32)]
    if suite == "holdout":
        shapes = [(1, 3), (3, 12), (6, 24), (8, 12)]
    elif suite == "validation":
        shapes = [(2, 5), (5, 16), (7, 20)]
    elif suite == "validation-rf8":
        shapes = [(4, 7), (6, 18)]
    for warps, steps in shapes:
        strategies = ["resident"] if suite == "calibration" else ["resident", "reload", "stream"]
        if suite in {"validation", "validation-rf8"}:
            strategies.append("shared")
        for strategy in strategies:
            for conflict in [False] if suite == "calibration" else [False, True]:
                name = f"{strategy}-w{warps}-k{steps * 8}-conflict{int(conflict)}"
                program = complete_gemm(steps, warps, strategy, conflict)
                hashes = write_probe(program, out / name, fixture)
                cases.append(
                    dict(
                        name=name,
                        split="holdout" if suite.startswith("validation") else suite,
                        generator_suite=suite,
                        warps=warps,
                        steps=steps,
                        strategy=strategy,
                        rf_conflict=conflict,
                        words=program.words,
                        expected_words=program.expected,
                        workload=asdict(program.workload),
                        final_tensor_pc=program.final_tensor_pc,
                        output_store_pc=program.output_store_pc,
                        input_sha256=hashes,
                    )
                )
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return {"cases": len(cases), "cases_sha256": sha(out / "cases.json")}


def predict(cases_path, out, frontend, banks, backend="auto", max_cycles=10_000_000):
    if backend not in {"auto", "python", "rust"}:
        raise ValueError("backend must be auto, python or rust")
    backend = ("rust" if frontend else "python") if backend == "auto" else backend
    if backend == "rust" and not frontend:
        raise ValueError("Rust probe prediction requires --frontend")
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for case in json.loads(cases_path.read_text()):
        hardware = Hardware().with_changes(rf_banks=banks)
        candidate = read_candidate(dict(workload=case["workload"], hardware=hardware.to_dict()))
        begin = perf_counter()
        workload = candidate.workload
        if frontend:
            # The frozen cases keep original ISA. Add its terminal control request
            # explicitly; it can contend before another warp's compute endpoint.
            workload = with_physical_operands(append_endprg(workload, case["words"]), case["words"])
        endpoint_names = [
            f"w{warp}.pc{case['final_tensor_pc']:08x}" for warp in range(case["warps"])
        ]
        if backend == "rust":
            from .rust_frontend import predict_span

            result = predict_span(
                workload, hardware, endpoint_names=endpoint_names, max_cycles=max_cycles
            )
            start = result["first_collect"]
            ends = list(result["endpoint_times"].values())
        else:
            graph = build_graph(workload, hardware, frontend=frontend)
            result = execute(graph)
            times = dict(zip(graph.names, result["times"], strict=True))
            start = min(t for name, t in times.items() if name.endswith(".collect"))
            ends = [times[name + ".writeback"] for name in endpoint_names]
        seconds = perf_counter() - begin
        rows.append(
            dict(
                name=case["name"],
                banks=banks,
                prediction_cycles=max(ends) - start,
                host_seconds=seconds,
                model_version=FRONTEND_VERSION if frontend else MODEL_VERSION,
                first_collect=start,
                last_tensor_writeback=max(ends),
                backend=backend,
                native_identity={
                    key: result[key]
                    for key in ("native_version", "native_source_sha256", "native_runtime_sha256")
                    if key in result
                },
                limitations=result.get(
                    "limitations",
                    ["source-frontend-v3 scope; native parity does not improve RTL accuracy"],
                ),
                counters=result["counters"],
            )
        )
    if backend == "rust" and any(
        row["native_identity"] != rows[0]["native_identity"] for row in rows
    ):
        raise RuntimeError(
            "native runtime changed during prediction; no mixed-version artifact written"
        )
    payload = dict(
        cases_sha256=sha(cases_path),
        code_sha256=model_hashes(),
        predictions=rows,
        frontend=frontend,
        prediction_hardware={"rf_banks": banks},
        backend=backend,
        max_cycles=max_cycles if backend == "rust" else None,
    )
    with out.open("x") as stream:
        json.dump(payload, stream, indent=2)
    return {"predictions": len(rows), "maximum_host_seconds": max(r["host_seconds"] for r in rows)}


def run_rtl(cases_path, out, rtl_root, observer, spdlog, jobs, timeout):
    if jobs not in (1, 2) or not 1 <= timeout <= 600:
        raise ValueError("RTL collection is bounded to 1–2 workers and 1–600 s per case")
    rtl = rtl_root / "sim-verilator/build/driver_example/debug/sim-VentusRTL"
    library = rtl_root / "sim-verilator/build/libVentusRTL/debug/libVentusRTL.so"
    identities = dict(
        rtl_binary_sha256=sha(rtl), rtl_library_sha256=sha(library), trace_tool_sha256=sha(observer)
    )
    cases = json.loads(cases_path.read_text())
    out.mkdir(parents=True)
    source = cases_path.parent.resolve()

    def run(case):
        for name, digest in case["input_sha256"].items():
            if sha(source / case["name"] / name) != digest:
                raise ValueError("probe input hash differs")
        cwd = (out / case["name"]).resolve()
        cwd.mkdir()
        command = [
            str(rtl),
            "--kernel",
            f"name={case['name']},metafile={source / case['name'] / 'input.metadata'},"
            f"datafile={source / case['name'] / 'input.data'}",
            "--sim-time-max",
            "2000000",
            "--snapshot",
            "20000000",
            "--dump-mem",
            f"0x90002000,0x{0x90002000 + 4 * len(case['expected_words']) - 4:08x}",
        ]
        env = os.environ.copy()
        env.update(
            VENTUS_LAUNCH_NOT_BEFORE="512",
            LD_PRELOAD=str(observer),
            VENTUS_TRACE_PATH=str(cwd / "trace.csv"),
            LD_LIBRARY_PATH=str(spdlog / "lib") + ":" + env.get("LD_LIBRARY_PATH", ""),
        )
        start = perf_counter()
        timed_out = False
        with (cwd / "stdout.log").open("w") as stdout, (cwd / "stderr.log").open("w") as stderr:
            try:
                process = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode, timed_out = -1, True
        log = (cwd / "stdout.log").read_text(errors="replace")
        words = [
            int(x, 16)
            for x in re.findall(r"dump-mem: mem\[0x[0-9A-F]+ \+: 4\] = 0x([0-9A-F]+)", log)
        ]
        warnings = [line for line in log.splitlines() if "[RTL    error]" in line]
        diagnostics = classify_rtl_logs(
            log,
            (cwd / "stderr.log").read_text(errors="replace"),
            kernel_name=case["name"],
            returncode=returncode,
            timed_out=timed_out,
        )
        trace_path = cwd / "trace.csv"
        trace = list(csv.reader(trace_path.open())) if trace_path.exists() else []
        collect = [row for row in trace if row[1] == "collect"]
        first = min((int(row[0]) for row in collect), default=None)
        ends = [
            row
            for row in trace
            if row[1] == "writeback" and int(row[3], 16) == case["final_tensor_pc"]
        ]
        stream_matches = all(
            [(int(row[3], 16), int(row[4], 16)) for row in collect if int(row[2]) == warp]
            == [(0x80000000 + 4 * i, word) for i, word in enumerate(case["words"])]
            for warp in range(case["warps"])
        )
        valid_ends = sorted(int(row[2]) for row in ends) == list(range(case["warps"]))
        summary = dict(
            name=case["name"],
            **identities,
            returncode=returncode,
            timed_out=timed_out,
            wall_seconds=perf_counter() - start,
            output_words=words,
            warnings=warnings,
            functional_ok=words == case["expected_words"] and not warnings,
            stream_matches=stream_matches,
            launch_not_before_cycle=512,
            compute_span_observed=stream_matches and valid_ends,
            completed=diagnostics["kernel_completed"] and stream_matches and valid_ends,
            diagnostics=diagnostics,
            reference_cycles=max(int(row[0]) for row in ends) - first
            if first is not None and valid_ends
            else None,
            trace_sha256=sha(trace_path) if trace_path.exists() else None,
            input_sha256=case["input_sha256"],
        )
        (cwd / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k not in {"output_words", "input_sha256"}}
            ),
            flush=True,
        )
        return summary

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(run, cases))
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return {"cases": len(rows), "completed": sum(row["completed"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generator = commands.add_parser("generate")
    generator.add_argument("--out", type=Path, required=True)
    generator.add_argument("--fixture", type=Path, required=True)
    generator.add_argument(
        "--suite", choices=("calibration", "holdout", "validation", "validation-rf8"), required=True
    )
    predictor = commands.add_parser("predict")
    predictor.add_argument("--cases", type=Path, required=True)
    predictor.add_argument("--out", type=Path, required=True)
    predictor.add_argument("--frontend", action="store_true")
    predictor.add_argument("--backend", choices=("auto", "python", "rust"), default="auto")
    predictor.add_argument("--max-cycles", type=int, default=10_000_000)
    predictor.add_argument("--banks", type=int, choices=(4, 8), required=True)
    runner = commands.add_parser("rtl")
    runner.add_argument("--cases", type=Path, required=True)
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--rtl-root", type=Path)
    runner.add_argument("--observer", type=Path)
    runner.add_argument("--spdlog", type=Path)
    runner.add_argument("--jobs", type=int, default=1)
    runner.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args.out, args.fixture, args.suite)
    elif args.command == "predict":
        result = predict(
            args.cases, args.out, args.frontend, args.banks, args.backend, args.max_cycles
        )
    else:
        profile = load_profile().get("remote", {})
        rtl_root = args.rtl_root or Path(profile["project_root"])
        observer = args.observer or Path(profile["flow_root"]) / "trace_rtl-v4.so"
        spdlog = args.spdlog or Path(profile["spdlog_root"])
        result = run_rtl(args.cases, args.out, rtl_root, observer, spdlog, args.jobs, args.timeout)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
