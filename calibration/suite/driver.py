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
    "b13_l2_structure": "pointer chases: sector/line fill, set-conflict (stride, n) grid for ways and sets, one-bit address flips for slice bits, cyclic vs random footprints for replacement (Mei & Chu 2017; Jia et al. 2018)",
    "b14_l1_carveout": "pointer chase vs footprint per shared-memory carve-out; L1 allocation after cp.async.cg / .ca / TMA",
    "b15_copy_queue": "cp.async group issue time vs pending groups; completion-time distribution; cp.async vs TMA contention; TMA 2-D and mbarrier cost; per-instruction issue cost (Luo et al. 2024)",
    "b16_block_scheduler": "blocks record %smid and %globaltimer at entry: dispatch order, rate, refill latency, wave staircase (Accel-Sim)",
    "b17_latency_hiding": "bandwidth vs resident warps (DRAM and L2), per-warp MLP, per-SM outstanding requests, issue width, dependent-chain latencies (Volkov 2016; Jia et al. 2018)",
    "b18_dram": "bandwidth vs power-of-two stride (interleave bits, channels), sequential vs random (page locality), read/write mix, CPU contention, 60 s drift",
    "b19_registers_local": "FMA kernel under -maxrregcount (spill curve from ptxas -v); local-memory chase and stream; allocation granularity from B9",
    "b20_tensor_detail": "cublasLt FP8; mma.sync shapes per precision; dependent mma chain latency; ldmatrix feed; ISA probes",
    "b21_compute_block": "per-opcode throughput (B2 pattern); the generated inner loop's cycles per MAC step with its SASS mix",
    "b22_clock_governor": "in-kernel clock64/globaltimer sampling: step response, clock by load type, power cap, clock after idle gaps",
    "b23_unified_mechanics": "first-touch cost by page size, prefetch, host-resident access, managed-vs-device overhead with the CPU idle, allocation limits",
    "b24_atomics_reductions": "atomicAdd / red throughput by scope and address pattern; contention curve",
    "b25_multi_kernel": "kernel-to-kernel gap, CUDA graph gap, two-stream concurrency, L2 persistence across kernels",
    "b26_functional_probes": "in-flight cp.async visibility and chunk order, wait_group semantics, barrier/fence visibility, TMA early bytes, FP32 contraction/ftz/tensor rounding, bank map, independent thread scheduling",
}
MICRO = [f"b{i}_" for i in range(13, 27)]
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
    if max(samples) == min(samples):
        return 0.0                               # a constant (a boolean probe, a count, a zero) has no spread
    rng = random.Random(0)
    meds = sorted(statistics.median(rng.choices(samples, k=len(samples))) for _ in range(400))
    lo, hi = meds[int(0.025 * len(meds))], meds[int(0.975 * len(meds)) - 1]
    m = statistics.median(samples)
    return (hi - lo) / 2 / m if m else 1.0


def build(name, arch, build_dir, flags=(), suffix=""):
    """Compile bench/<name>.cu; returns (exe, ptxas stderr). Extra nvcc flags and an exe suffix give B19 its per-cap variants."""
    src = HERE / "bench" / f"{name}.cu"
    exe = build_dir / (name + suffix)
    log = build_dir / (name + suffix + ".ptxas")
    if exe.exists() and exe.stat().st_mtime > src.stat().st_mtime and exe.stat().st_mtime > (HERE / "bench/common.cuh").stat().st_mtime:
        return exe, (log.read_text() if log.exists() else "")
    extra = {"b3_tensor": ["-lcublas"], "b20_tensor_detail": ["-lcublas", "-lcublasLt"], "b15_copy_queue": ["-lcuda"]}.get(name, [])
    r = sh(["nvcc", "-O3", f"-arch={arch}", "-std=c++17", "-Xptxas", "-v", "-o", str(exe), str(src)] + extra + list(flags))
    if r.returncode != 0:
        print(f"[build] {name}{suffix} FAILED:\n{r.stderr}", file=sys.stderr)
        return None, ""
    log.write_text(r.stderr)
    return exe, r.stderr


B19_CAPS = [255, 128, 96, 64, 48, 32]
# quantities in these units are inferred structure (a count, a yes/no, a knee), not a rate: the interval target does not apply
STRUCTURAL_UNITS = {"bool", "count", "loads", "warps", "ways", "groups", "channels", "sms", "cycles_structural"}


def rec_unit(r):
    return r.get("unit")


def variants(name, arch, build_dir):
    """(exe, extra run args) per build variant. B19 is rebuilt once per register cap with the ptxas spill count passed in."""
    if name != "b19_registers_local":
        exe, _ = build(name, arch, build_dir)
        return [(exe, [])] if exe else []
    out = []
    for cap in B19_CAPS:
        exe, log = build(name, arch, build_dir, [f"-maxrregcount={cap}"], f"_r{cap}")
        if not exe:
            continue
        m = re.search(r"Function properties for _Z5acc64Pfi\s*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
        spill = int(m.group(2)) + int(m.group(3)) if m else -1
        out.append((exe, ["--maxrreg", str(cap), "--spill-bytes", str(spill)]))
    return out


def sass_mix(exe, kernel_substr):
    """Opcode histogram of every SASS function whose mangled name contains kernel_substr, from cuobjdump -sass."""
    r = sh(["cuobjdump", "-sass", str(exe)])
    if r.returncode != 0:
        return {}
    mixes, cur = {}, None
    for line in r.stdout.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1) if kernel_substr in m.group(1) else None
            if cur:
                mixes[cur] = {}
            continue
        if cur:
            m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_]*)", line)
            if m:
                op = m.group(1).split(".")[0]
                mixes[cur][op] = mixes[cur].get(op, 0) + 1
    return mixes


def run_once(exe, reps, extra):
    r = sh([str(exe), "--reps", str(reps)] + extra)
    if r.returncode != 0:
        raise RuntimeError(f"{exe.name} failed: {r.stderr[:500]}")
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {exe.name}: unparsable record skipped ({e}): {line[:300]} ... {line[max(0, e.pos - 80):e.pos + 40]}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--ci", type=float, default=0.04); ap.add_argument("--max-reps", type=int, default=50)
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--full", action="store_true")
    ap.add_argument("--micro", action="store_true", help="run B13-B26 (the structure suite)")
    ap.add_argument("--extra", default="", help="extra args passed to every benchmark")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    build_dir = HERE / "build"; build_dir.mkdir(exist_ok=True)
    names = a.only or ([n for n in METHODS if any(n.startswith(m) for m in MICRO)] if a.micro else (list(METHODS) if a.full else MINIMUM))
    names = [n for n in names if (HERE / "bench" / f"{n}.cu").exists()]
    summary = {"host": os.uname().nodename, "arch": a.arch, "ci_target": a.ci, "max_reps": a.max_reps,
               "nvcc": sh("nvcc --version | tail -1").stdout.strip(), "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").stdout.strip(),
               "ncu": sh("which ncu").stdout.strip() or None, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "benchmarks": {}}
    for name in names:
        vs = variants(name, a.arch, build_dir)
        if not vs:
            summary["benchmarks"][name] = "build failed"; continue
        t0 = time.time(); c0 = clocks()
        records = {}
        reps = 10 if name != "b4_bandwidth" else 5
        if name in ("b13_l2_structure", "b16_block_scheduler", "b18_dram", "b22_clock_governor", "b25_multi_kernel", "b26_functional_probes", "b23_unified_mechanics"):
            reps = 3
        # benchmarks whose single invocation already takes minutes (60 s loads, drift runs) or that report a structure rather
        # than a rate run once: repeating them to the interval target would cost an hour each and change nothing
        once = name in ("b13_l2_structure", "b18_dram", "b22_clock_governor", "b23_unified_mechanics", "b26_functional_probes", "b16_block_scheduler")
        total = 0
        failed = False
        while True:
            batch = []
            for exe, run_extra in vs:
                try:
                    batch += run_once(exe, reps, a.extra.split() + run_extra)
                except RuntimeError as e:
                    print(f"[run] {e}", file=sys.stderr); summary["benchmarks"][name] = f"run failed: {e}"[:200]; failed = True; break
            if failed:
                break
            total += reps
            for rec in batch:
                q = rec["quantity"]
                r = records.setdefault(q, {"quantity": q, "unit": rec["unit"], "samples": [], "detail": rec.get("detail", {})})
                r["samples"].extend(rec["samples"])
            worst = max((ci95_rel(r["samples"]) for r in records.values() if r["samples"]), default=0.0)
            if once or not records or all(not r["samples"] for r in records.values()) or worst <= a.ci or total >= a.max_reps:
                break
        if name == "b21_compute_block" and records:
            mixes = sass_mix(vs[0][0], "inner_loop")
            for q, r in records.items():
                d = r["detail"]
                if q.startswith("compute_block_efficiency") and {"TM", "TN", "TK"} <= set(d):
                    key = f"inner_loopILi{d['TM']}ELi{d['TN']}ELi{d['TK']}E"
                    d["sass_mix"] = next((m for k, m in mixes.items() if key in k), {})
        c1 = clocks()
        for q, r in records.items():
            s = r["samples"]
            r.update({"value": statistics.median(s) if s else None, "ci95": ci95_rel(s) if s else None, "n": len(s), "repeat_policy": "once" if once else ("structural" if rec_unit(r) in STRUCTURAL_UNITS else "to_ci"),
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
