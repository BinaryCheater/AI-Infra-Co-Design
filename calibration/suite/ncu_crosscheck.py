#!/usr/bin/env python3
"""Counter cross-checks for the structure suite (B13_plus.md, the `ncu` column), run on the GPU host after P0.

Each check launches one kernel of bench/bx_counters.cu under `ncu` and writes a record `counter_<name>[...]`
to the raw directory in the suite's format (quantity, unit, samples, detail with every raw metric). The
fitter turns them into the contract's `counter_crosscheck` block next to the pointer-chase inferences.

  ncu_crosscheck.py --out raw --build build [--arch sm_121]
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMS = None

L2 = "lts__t_requests_srcunit_tex.sum,lts__t_sectors_srcunit_tex.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sector_hit_rate.pct,gpu__time_duration.sum"
ISSUE = "sm__cycles_active.avg,sm__inst_issued.sum,sm__issue_active.avg,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_alu.sum,sm__pipe_fma_cycles_active.avg,smsp__warps_active.avg,gpu__time_duration.sum"
STALL = ",".join(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio" for s in ("long_scoreboard", "short_scoreboard", "math_pipe_throttle", "mio_throttle", "not_selected", "wait", "lg_throttle"))
SMEM = "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,sm__sass_inst_executed_op_shared_ld.sum,sm__sass_data_bytes_mem_shared.sum,gpu__time_duration.sum"
L1 = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_hit_rate.pct,l1tex__t_requests.sum,gpu__time_duration.sum"

TENSOR = "sm__cycles_active.avg,sm__pipe_tensor_cycles_active.avg,sm__inst_executed_pipe_tensor.sum,sm__inst_executed.sum,smsp__issue_active.avg,smsp__cycles_active.avg,sm__pipe_fma_cycles_active.avg,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum," + ",".join(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio" for s in ("mio_throttle", "short_scoreboard", "not_selected", "math_pipe_throttle", "wait"))
# round-2 checks run a benchmark binary's --probe mode (warm-up + one profiled launch of the saturated configuration)
PROBES = [
    ("b31_tensor", "b31_mma_issue", TENSOR, [("0", "fp16"), ("1", "bf16"), ("2", "tf32"), ("3", "fp8")], "B31: tensor pipe active cycles and mma instructions at the saturated point (a = 16, 4 warps per sub-partition) per shape; rate vs B20"),
    ("b32_ldsm", "b32_ldmatrix_mma", TENSOR + ",l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", [("1", "x1"), ("2", "x2"), ("4", "x4"), ("pipelined", "pipelined"), ("unswizzled", "unswizzled")], "B32: ldmatrix wavefronts per instruction by x1/x2/x4; the pipelined loop's tensor activity and stalls, swizzled and not"),
    ("b33_mix", "b33_mma_mio", TENSOR + ",l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", [("lds128", "lds128"), ("cp_async", "cp_async"), ("both", "both")], "B33: tensor pipe activity, shared wavefronts and stalls with one tensor block and one load block per SM"),
    ("c1_tile", "c1_tensor_tile", TENSOR + ",smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,smsp__warps_active.avg", [("1", "bps1"), ("2", "bps2"), ("3", "bps3")], "C1 composite: tensor pipe activity, issue-active, shared wavefronts and the stall families of the 128x128x32 tile kernel at 1-3 blocks per SM"),
    ("b34_mixed", "b34_mixed_issue_r255", TENSOR + ",l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", [("ratio4", "ratio4"), ("loop128", "loop128"), ("split", "split")], "B34: issue-active, FMA and tensor pipe activity with 4 FFMAs per mma; the 128-accumulator loop's local-memory sectors"),
]
CHECKS = [
    # name, kernel, metrics, [(a, b)...], what it cross-checks
    ("l2_fill", "strided", L2, [(32, 0), (64, 0), (128, 0), (256, 0), (512, 0)], "B13.1 fill granularity: L2 sectors per request and DRAM (miss) sectors per request at each stride"),
    ("l2_hit_rate", "footprint", L2, [(4, 0), (8, 0), (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0), (48, 0)], "B13.4 / B4: L2 sector hit rate of an 8-pass per-block cyclic stream (L1 bypassed) vs total footprint: capacity from counters"),
    ("issue", "fma_issue", ISSUE, [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0)], "B17.4: instructions issued per active SM cycle and issue-active fraction vs warps"),
    ("inner_loop", "inner_T1", ISSUE + "," + STALL, [(0, 0)], "B21: issue-active fraction, FMA pipe activity, LSU share and stall reasons of the T1 inner loop"),
    ("inner_loop_T5", "inner_T5", ISSUE + "," + STALL, [(0, 0)], "B21: the same for T5 128x128x32"),
    ("inner_loop_R2", "inner_R2", ISSUE + "," + STALL, [(0, 0)], "B21: the same for R2 32x32x32"),
    ("smem_banks", "smem_stride", SMEM, [(1, 0), (2, 0), (4, 0), (8, 0), (16, 0), (32, 0)], "B6 / B26.6: shared-memory wavefronts and bank conflicts per instruction vs word stride"),
    ("lds_wavefronts", "lds_width", SMEM + ",sm__cycles_active.avg,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
     [(4, 1), (4, 2), (4, 4), (4, 8), (8, 1), (8, 2), (8, 4), (16, 1), (16, 2), (16, 4)], "B30.3: shared-memory wavefronts per ld.shared instruction by width (a, bytes) and word stride (b), 16 warps per SM, with the mio_throttle stall"),
    ("lds_occupancy", "lds_occ", SMEM + ",sm__cycles_active.avg,smsp__issue_active.avg,smsp__cycles_active.avg,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
     [(4, 1), (4, 2), (4, 4), (4, 8), (4, 12), (16, 1), (16, 2), (16, 4), (16, 8), (16, 12)], "B30 occupancy sweep: shared-load instructions per cycle, issue-active and mio_throttle at w warps per sub-partition (a = width bytes, b = w)"),
    ("l1_hit", "l1_chase", L1, [(16, 0), (32, 0), (48, 0), (64, 0), (96, 0), (112, 0), (128, 0), (160, 0), (16, 16384), (32, 16384), (64, 16384), (96, 16384), (16, 49152), (32, 49152), (64, 49152), (16, 65536), (24, 65536), (32, 65536), (16, 98304), (24, 98304), (32, 98304)], "B14.1: L1 sector hit rate of a 128 B-stride chase at footprint KiB (a) and smem request bytes (b)"),
]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def parse_csv(text):
    vals, units = {}, {}
    for line in text.splitlines():
        if not line.startswith('"'):
            continue
        cells = [c.strip().strip('"') for c in line.split('","')]
        # ncu --csv: ... "Metric Name","Metric Unit","Metric Value"
        if len(cells) >= 3 and re.match(r"^[a-z0-9_]+(\.[a-z_]+)+$", cells[-3]):
            try:
                vals[cells[-3]] = float(cells[-1].replace(",", "")); units[cells[-3]] = cells[-2]
            except ValueError:
                pass
    return vals, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--build", required=True); ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); build = Path(a.build); build.mkdir(exist_ok=True)
    exe = build / "bx_counters"; src = HERE / "bench" / "bx_counters.cu"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        r = sh(["nvcc", "-O3", f"-arch={a.arch}", "-std=c++17", "-o", str(exe), str(src)])
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr); sys.exit(1)
    ncu = sh(["which", "ncu"]).stdout.strip()
    if not ncu:
        print("ncu not on PATH", file=sys.stderr); sys.exit(1)
    for name, exe_name, metrics, probes, what in PROBES:
        if a.only and name not in a.only:
            continue
        pexe = build / exe_name
        if not pexe.exists():
            print(f"[ncu] {name}: {exe_name} not built, skipped", file=sys.stderr); continue
        for probe, label in probes:
            t0 = time.time()
            r = sh([ncu, "--csv", "--metrics", metrics, "--launch-skip", "1", "--launch-count", "1", str(pexe), "--reps", "1", "--probe", probe])
            vals, units = parse_csv(r.stdout)
            q = f"counter_{name}[probe={label}]"
            rec = {"quantity": q, "unit": "metrics", "samples": [1.0] if vals else [], "value": 1.0 if vals else None, "ci95": 0.0 if vals else None, "n": 1 if vals else 0,
                   "repeat_policy": "once", "method": f"ncu: {what}", "benchmark": exe_name, "arch": a.arch, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "detail": {"probe": probe, "metrics": {k: [v, units.get(k)] for k, v in vals.items()}, "measured": bool(vals), "reason": None if vals else (r.stderr or r.stdout)[-400:].strip(), "seconds": round(time.time() - t0, 1)}}
            (out / (re.sub(r"[^A-Za-z0-9_.=\[\]-]", "_", q) + ".json")).write_text(json.dumps(rec, indent=1) + "\n")
            print(f"[ncu] {q}: {len(vals)} metrics, {time.time() - t0:.1f} s" + ("" if vals else f" FAILED: {rec['detail']['reason'][:200]}"), file=sys.stderr)
    for name, kernel, metrics, args, what in CHECKS:
        if a.only and name not in a.only:
            continue
        for (x, y) in args:
            t0 = time.time()
            cmd = [ncu, "--csv", "--metrics", metrics, "--launch-skip", "1", "--launch-count", "1", str(exe), "--kernel", kernel, "--a", str(x), "--b", str(y)]
            r = sh(cmd)
            vals, units = parse_csv(r.stdout)
            q = f"counter_{name}[a={x},b={y}]"
            rec = {"quantity": q, "unit": "metrics", "samples": [1.0] if vals else [], "value": 1.0 if vals else None, "ci95": 0.0 if vals else None, "n": 1 if vals else 0,
                   "repeat_policy": "once", "method": f"ncu: {what}", "benchmark": "bx_counters", "arch": a.arch, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "detail": {"kernel": kernel, "a": x, "b": y, "metrics": {k: [v, units.get(k)] for k, v in vals.items()}, "measured": bool(vals),
                              "reason": None if vals else (r.stderr or r.stdout)[-400:].strip(), "seconds": round(time.time() - t0, 1)}}
            fn = re.sub(r"[^A-Za-z0-9_.=\[\]-]", "_", q) + ".json"
            (out / fn).write_text(json.dumps(rec, indent=1) + "\n")
            print(f"[ncu] {q}: {len(vals)} metrics, {time.time() - t0:.1f} s" + ("" if vals else f" FAILED: {rec['detail']['reason'][:200]}"), file=sys.stderr)


if __name__ == "__main__":
    main()
