#!/usr/bin/env python3
"""Calibration driver (runs ON the GPU host). Builds each benchmark with nvcc, runs it, repeats until
the 95 % confidence interval of every quantity is under --ci (default 4 %, since clocks cannot be
locked on the Spark) or --max-reps repetitions, and writes one JSON record per quantity to --out:

  {quantity, value, unit, ci95, n, method, clocks, timestamp, detail, samples}

value = median of the samples; ci95 = half-width of the 95 % interval of the median (bootstrap),
relative to the value. Clocks are sampled with nvidia-smi before and after each benchmark.
"""
import argparse, json, os, re, statistics, subprocess, sys, time, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = {
    "b1_props": "cudaGetDeviceProperties / cudaDeviceGetAttribute (declared values)",
    "b2_fma": "dependent-FMA chains, ILP 8, 8 blocks per SM x 256 threads, cudaEvent timing (Jia et al. 2018)",
    "b4_bandwidth": "grid-stride float4 read/write/copy over 1 MB..8 GB, >=256 MB per sample (STREAM-style; Accel-Sim tuner)",
    "b5_latency": "single-thread pointer chase over a random cyclic permutation of 64 B lines (Mei & Chu 2017; Jia et al. 2018)",
    "b6_smem": "ld.shared throughput at word strides 1..32, 4 blocks per SM (Jia et al. 2018; Luo et al. 2024)",
    "b7_barrier": "__syncthreads loop, one block per SM, block sizes 128..1024",
    "b3_tensor": "cublasGemmEx large square GEMM per precision; mma.sync microkernel per SM",
    "b8_async_copy": "cp.async / cp.async.bulk issue-to-completion timing per tile size",
    "b9_occupancy": "cudaOccupancyMaxActiveBlocksPerMultiprocessor sweep over registers and smem",
    "b10_launch": "1000 empty launches",
    "b11_clocks": "sustained FMA load with nvidia-smi sampling",
    "b12_unified": "streaming over cudaMallocManaged buffers touched by CPU first vs GPU first",
}
MINIMUM = ["b1_props", "b2_fma", "b4_bandwidth", "b5_latency", "b6_smem", "b7_barrier"]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def clocks():
    r = sh(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_event_reasons.sw_power_cap",
            "--format=csv,noheader,nounits"])
    if r.returncode != 0:
        return {"error": r.stderr.strip()[:200]}
    sm, mx, temp, pwr, cap = [x.strip() for x in r.stdout.strip().split(",")]
    return {"sm_mhz": float(sm), "max_sm_mhz": float(mx), "temp_c": float(temp), "power_w": float(pwr) if pwr != "[N/A]" else None,
            "sw_power_cap": cap}


def ci95_rel(samples):
    """Bootstrap 95 % interval of the median, half-width relative to the median."""
    if len(samples) < 3:
        return 1.0
    rng = random.Random(0)
    meds = sorted(statistics.median(rng.choices(samples, k=len(samples))) for _ in range(400))
    lo, hi = meds[int(0.025 * len(meds))], meds[int(0.975 * len(meds)) - 1]
    m = statistics.median(samples)
    return (hi - lo) / 2 / m if m else 1.0


def build(name, arch, build_dir):
    src = HERE / "bench" / f"{name}.cu"
    exe = build_dir / name
    if exe.exists() and exe.stat().st_mtime > src.stat().st_mtime and exe.stat().st_mtime > (HERE / "bench/common.cuh").stat().st_mtime:
        return exe
    extra = ["-lcublas"] if name == "b3_tensor" else []
    r = sh(["nvcc", "-O3", f"-arch={arch}", "-std=c++17", "-o", str(exe), str(src)] + extra)
    if r.returncode != 0:
        print(f"[build] {name} FAILED:\n{r.stderr}", file=sys.stderr)
        return None
    return exe


def run_once(exe, reps, extra):
    r = sh([str(exe), "--reps", str(reps)] + extra)
    if r.returncode != 0:
        raise RuntimeError(f"{exe.name} failed: {r.stderr[:500]}")
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--ci", type=float, default=0.04); ap.add_argument("--max-reps", type=int, default=50)
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--full", action="store_true")
    ap.add_argument("--extra", default="", help="extra args passed to every benchmark")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    build_dir = HERE / "build"; build_dir.mkdir(exist_ok=True)
    names = a.only or (list(METHODS) if a.full else MINIMUM)
    names = [n for n in names if (HERE / "bench" / f"{n}.cu").exists()]
    summary = {"host": os.uname().nodename, "arch": a.arch, "ci_target": a.ci, "max_reps": a.max_reps,
               "nvcc": sh("nvcc --version | tail -1").stdout.strip(), "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").stdout.strip(),
               "ncu": sh("which ncu").stdout.strip() or None, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "benchmarks": {}}
    for name in names:
        exe = build(name, a.arch, build_dir)
        if exe is None:
            summary["benchmarks"][name] = "build failed"; continue
        t0 = time.time(); c0 = clocks()
        records = {}
        reps = 10 if name != "b4_bandwidth" else 5
        total = 0
        while True:
            try:
                batch = run_once(exe, reps, a.extra.split())
            except RuntimeError as e:
                print(f"[run] {e}", file=sys.stderr); summary["benchmarks"][name] = f"run failed: {e}"[:200]; batch = []; break
            total += reps
            for rec in batch:
                q = rec["quantity"]
                r = records.setdefault(q, {"quantity": q, "unit": rec["unit"], "samples": [], "detail": rec.get("detail", {})})
                r["samples"].extend(rec["samples"])
            worst = max((ci95_rel(r["samples"]) for r in records.values() if r["samples"]), default=0.0)
            if not records or all(not r["samples"] for r in records.values()) or worst <= a.ci or total >= a.max_reps:
                break
        c1 = clocks()
        for q, r in records.items():
            s = r["samples"]
            r.update({"value": statistics.median(s) if s else None, "ci95": ci95_rel(s) if s else None, "n": len(s),
                      "method": METHODS.get(name, name), "benchmark": name, "clocks": {"before": c0, "after": c1},
                      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arch": a.arch})
            fn = re.sub(r"[^A-Za-z0-9_.=\[\]-]", "_", q) + ".json"
            (out / fn).write_text(json.dumps(r, indent=1) + "\n")
        summary["benchmarks"][name] = {"quantities": len(records), "seconds": round(time.time() - t0, 1), "reps": total,
                                       "worst_ci95": max((r["ci95"] or 0 for r in records.values()), default=None)}
        print(f"[done] {name}: {len(records)} quantities, {total} reps, {time.time() - t0:.1f} s", file=sys.stderr)
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
