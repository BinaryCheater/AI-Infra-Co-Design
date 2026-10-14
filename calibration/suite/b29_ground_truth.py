#!/usr/bin/env python3
"""B29: the generated kernels at the bad corners under counters (B27_plus.md §1). Generates the six kernels with codegen, a
`stripped` variant (global loads removed, tile left as is; shared loads, FFMAs, per-phase overhead and barriers kept) and the
`real` one, runs each at b in {1, 2, 4, 6} resident blocks per SM (one wave, residency forced by smem padding) on the host with
timing and the ncu issue/stall breakdown, and writes records `inner_loop_ground_truth[kernel=,variant=,b=]` into raw/.

  b29_ground_truth.py --host dgx-spark [--out calibration/spark-6951/raw]
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agentic.chips import load_chip, scope_chip                                   # noqa: E402
from agentic.examples.gemm_tiling import tiling_contract                        # noqa: E402
from agentic.agents.evaluator.cuda import kernel_artifact                        # noqa: E402
from agentic.templates.smem_tiling import Workload                               # noqa: E402

REMOTE = "~/work/research/AI-Infra-Co-Design-calib/b29"
KERNELS = [  # name, workload, design
    ("skinny_16x32x8_S2_P0", Workload("skinny_64x4096x4096", 64, 4096, 4096), dict(Tm=16, Tn=32, Tk=8, S=2, P=0, beta=1)),
    ("ragged_256x16x32_S2_P0", Workload("ragged_1000x1000x1000", 1000, 1000, 1000), dict(Tm=256, Tn=16, Tk=32, S=2, P=0, beta=0)),
    ("square_32x64x32_S2_P1", Workload("square_4096", 4096, 4096, 4096), dict(Tm=32, Tn=64, Tk=32, S=2, P=1, beta=1)),
    ("square_32x128x32_S3_P0", Workload("square_4096", 4096, 4096, 4096), dict(Tm=32, Tn=128, Tk=32, S=3, P=0, beta=1)),
    ("T5_128x128x32_S3_P1", Workload("ragged_1000x1000x1000", 1000, 1000, 1000), dict(Tm=128, Tn=128, Tk=32, S=3, P=1, beta=0)),
    ("T1_128x256x16_S2_P1", Workload("square_4096", 4096, 4096, 4096), dict(Tm=128, Tn=256, Tk=16, S=2, P=1, beta=1)),
]
STALLS = ("not_selected", "short_scoreboard", "long_scoreboard", "mio_throttle", "wait", "math_pipe_throttle", "barrier", "lg_throttle", "no_instruction")
METRICS = ("sm__cycles_active.avg,sm__issue_active.avg,sm__inst_issued.sum,smsp__inst_executed.sum,sm__pipe_fma_cycles_active.avg,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,"
           "sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_fma.sum,sm__warps_active.avg.pct_of_peak_sustained_active," + ",".join(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio" for s in STALLS))


def strip_loads(src: str) -> str:
    """Remove the global->shared copies: load_tile becomes a no-op (the tile keeps whatever is in shared memory)."""
    i = src.index("__device__ __forceinline__ void load_tile(")
    j = src.index("{", i)
    # find the matching brace of the function body
    depth = 0; k = j
    while True:
        if src[k] == "{": depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0: break
        k += 1
    return src[:j] + "{ (void)As; (void)Bs; (void)A; (void)B; (void)M; (void)N; (void)K; (void)row0; (void)col0; (void)u; (void)use_async; }" + src[k + 1:]


def parse_csv(text):
    vals = {}
    for line in text.splitlines():
        if not line.startswith('"'): continue
        cells = [c.strip().strip('"') for c in line.split('","')]
        if len(cells) >= 3 and re.match(r"^[a-z0-9_]+(\.[a-z_]+)+$", cells[-3]):
            try: vals[cells[-3]] = float(cells[-1].replace(",", ""))
            except ValueError: pass
    return vals


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--host", required=True); ap.add_argument("--out", default=str(ROOT / "calibration" / "spark-6951" / "raw")); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--no-ncu", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    chip = scope_chip(load_chip("spark-6951-gb10"))
    work = Path("/tmp/b29_local"); work.mkdir(exist_ok=True)
    for f in ("harness.cu",):
        (work / f).write_text((ROOT / "calibration" / "suite" / "b29" / f).read_text())
    for name, w, v in KERNELS:
        if a.only and name not in a.only: continue
        T = tiling_contract(w, hw=chip, precondition="exact"); T.setdefault("id", "tmpl_b29"); T.setdefault("template", "smem_tiling")
        src = kernel_artifact(T, {"id": f"b29_{name}", "values": v})["source"]
        (work / f"{name}_real.cu").write_text(src); (work / f"{name}_stripped.cu").write_text(strip_loads(src))
    subprocess.run(["ssh", "-o", "BatchMode=yes", a.host, f"mkdir -p {REMOTE}"], check=True)
    subprocess.run(["rsync", "-az", f"{work}/", f"{a.host}:{REMOTE}/"], check=True)
    for name, w, v in KERNELS:
        if a.only and name not in a.only: continue
        for variant in ("stripped", "real"):
            build = (f"export PATH=$PATH:/usr/local/cuda/bin; cd {REMOTE} && mkdir -p build_{name}_{variant} && cp {name}_{variant}.cu build_{name}_{variant}/kernel.cu && cp harness.cu build_{name}_{variant}/ && "
                     f"cd build_{name}_{variant} && nvcc -O3 -arch=sm_121 -std=c++17 -o run harness.cu 2>&1 | tail -3")
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", a.host, build], capture_output=True, text=True)
            for b in (1, 2, 4, 6):
                t0 = time.time()
                cmd = f"export PATH=$PATH:/usr/local/cuda/bin; cd {REMOTE}/build_{name}_{variant} && ./run --m {w.m} --n {w.n} --k {w.k} --b {b} --reps {a.reps}"
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", a.host, cmd], capture_output=True, text=True)
                line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
                timing = json.loads(line) if line else {"error": (r.stdout + r.stderr)[-300:]}
                prof = {}
                if not a.no_ncu and "error" not in timing:
                    ncmd = (f"export PATH=$PATH:/usr/local/cuda/bin; cd {REMOTE}/build_{name}_{variant} && ncu --csv --metrics {METRICS} --launch-skip 3 --launch-count 1 "
                            f"--kernel-name smem_tiling_gemm ./run --m {w.m} --n {w.n} --k {w.k} --b {b} --reps 1")
                    rn = subprocess.run(["ssh", "-o", "BatchMode=yes", a.host, ncmd], capture_output=True, text=True)
                    prof = parse_csv(rn.stdout)
                cyc = prof.get("sm__cycles_active.avg")
                stalls = {k.split("stalled_")[1].split("_per")[0]: vv for k, vv in prof.items() if "issue_stalled" in k}
                q = f"inner_loop_ground_truth[kernel={name},variant={variant},b={b}]"
                rec = {"quantity": q, "unit": "cycles", "samples": [timing.get("cycles_per_step_per_block")] if "error" not in timing else [], "value": timing.get("cycles_per_step_per_block"),
                       "ci95": 0.0, "n": 1 if "error" not in timing else 0, "repeat_policy": "once", "benchmark": "b29_ground_truth", "arch": "sm_121", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "method": "B29: generated kernel (codegen) at the bad corners, one wave of 48 x b blocks with residency forced by smem padding; cycles per k-step per block from the wall time at the in-run clock; ncu issue and stall breakdown",
                       "detail": {"kernel": name, "variant": variant, "blocks_per_sm": b, "workload": w.name, "design": v, "timing": timing, "measured": "error" not in timing,
                                  "ncu": {"issue_active_fraction": (prof.get("sm__issue_active.avg") / cyc) if cyc and prof.get("sm__issue_active.avg") is not None else None,
                                          "inst_issued_per_active_cycle_per_sm": (prof.get("sm__inst_issued.sum") / 48 / cyc) if cyc and prof.get("sm__inst_issued.sum") else None,
                                          "fma_pipe_active_fraction": (prof.get("sm__pipe_fma_cycles_active.avg") / cyc / 8) if cyc and prof.get("sm__pipe_fma_cycles_active.avg") is not None else None,
                                          "lsu_wavefronts_per_active_cycle_per_sm": (prof.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum") / 48 / cyc) if cyc and prof.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum") else None,
                                          "warps_active_pct": prof.get("sm__warps_active.avg.pct_of_peak_sustained_active"), "stalls_per_issue_active": stalls, "dominant_stall": max(stalls, key=stalls.get) if stalls else None, "raw": prof}}}
                fn = re.sub(r"[^A-Za-z0-9_.=\[\]-]", "_", q) + ".json"
                (out / fn).write_text(json.dumps(rec, indent=1) + "\n")
                print(f"[b29] {name:<24} {variant:<8} b={b}: {timing.get('cycles_per_step_per_block', timing.get('error'))} cycles/step  issue {rec['detail']['ncu']['issue_active_fraction']}  stall {rec['detail']['ncu']['dominant_stall']}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
