#!/usr/bin/env python3
"""agentic-runner: compile a generated tiling kernel with the harness, run it, check it, print JSON.

  agentic_runner.py --kernel kernel.cu --m 1000 --n 1000 --k 1000 --reps 50 [--arch sm_121] [--json out.json]

Runs ON the GPU host. The JSON holds the median and 95 % interval of the kernel time (cudaEvent),
the cuBLAS reference check (`reference_agrees`, first disagreeing index), the launch geometry and
the clocks after the run. Exit code 0 even when the reference disagrees: that is a result, not an error.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# The GB10 exposes no dram__* counters (unified LPDDR5X behind the L2): DRAM traffic is the L2 lookup misses, 32 B sectors.
NCU_METRICS = "lts__t_bytes.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,gpu__time_duration.sum"


def ncu_traffic(args: list[str]) -> dict:
    """Profile the sixth launch (after the five warm-ups) with ncu; returns the metric values or why not."""
    ncu = shutil.which("ncu")
    if not ncu:
        return {"available": False, "reason": "ncu not on PATH"}
    cmd = [ncu, "--csv", "--metrics", NCU_METRICS, "--launch-skip", "5", "--launch-count", "1", "--kernel-name", "smem_tiling_gemm"] + args[:1] + args[1:9] + ["--reps", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    vals, units = {}, {}
    for line in r.stdout.splitlines():
        cells = [c.strip().strip('"') for c in line.split('","')]
        if len(cells) >= 3 and any(cells[i] in NCU_METRICS.split(",") for i in range(len(cells))):
            for i, c in enumerate(cells):
                if c in NCU_METRICS.split(",") and i + 2 < len(cells):
                    try:
                        vals[c] = float(cells[i + 2].replace(",", "")); units[c] = cells[i + 1]
                    except ValueError:
                        pass
    if not vals:
        err = next((l for l in (r.stderr + r.stdout).splitlines() if "ERR" in l or "error" in l.lower()), None)
        return {"available": True, "ok": False, "reason": (err or (r.stderr or r.stdout)[-400:]).strip()[:400],
                "permission_denied": "ERR_NVGPUCTRPERM" in (r.stderr + r.stdout)}
    scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    def bytes_of(k):
        v = vals.get(k); u = units.get(k, "byte")
        return None if v is None else v * scale.get(u, 1)
    def sectors_bytes(k):
        v = vals.get(k); return None if v is None else v * 32
    rd, wr = sectors_bytes("lts__t_sectors_op_read_lookup_miss.sum"), sectors_bytes("lts__t_sectors_op_write_lookup_miss.sum")
    return {"available": True, "ok": True, "dram_bytes": (rd or 0) + (wr or 0) if rd is not None or wr is not None else None, "l2_bytes": bytes_of("lts__t_bytes.sum"),
            "dram_bytes_read": rd, "dram_bytes_write": wr, "dram_source": "L2 lookup misses x 32 B (no dram__ counters on this chip)",
            "l2_read_bytes": sectors_bytes("lts__t_sectors_op_read.sum"), "l2_write_bytes": sectors_bytes("lts__t_sectors_op_write.sum"),
            "raw": {k: [vals[k], units.get(k)] for k in vals}}


def derive_policy(contract_path, k, tol_cli, warmup_cli):
    """Warm-up and reference tolerance from the measured contract (B13_plus.md §3, runner changes).

    warm-up: B22 clock_warmup_s (the time the clock takes to reach its sustained value from idle), else 0 s (five launches).
    tolerance: an FP32 FMA kernel checked against cublasSgemm differs by the accumulation-order rounding, bounded by
    K x 2^-24 relative to the output scale, times the measured ulp bound of the accumulate (B26.5; 1 when contraction is the
    default and nothing larger was measured). The CLI --tol stays as a floor.
    """
    pol = {"warmup_s": warmup_cli if warmup_cli is not None else 0.0, "tol": tol_cli, "source": "cli defaults"}
    if not contract_path:
        return pol
    try:
        import yaml
        c = yaml.safe_load(Path(contract_path).read_text())
    except Exception as e:                      # a missing yaml module or file leaves the defaults, and says so
        pol["source"] = f"contract unreadable: {e}"
        return pol
    fn = c.get("functional") or {}
    if warmup_cli is None and c.get("clock_warmup_s"):
        pol["warmup_s"] = float(c["clock_warmup_s"])
    ulp = fn.get("tensor_accumulate_ulp_bound")
    ulp_fp32 = 1.0
    if isinstance(ulp, dict):
        ulp_fp32 = float(ulp.get("fp32") or ulp.get("fp32_fma") or 1.0)
    elif isinstance(ulp, (int, float)):
        ulp_fp32 = 1.0                           # the measured bound is for tensor accumulates; the FP32 FMA path is 1 ulp per op
    derived = ulp_fp32 * k * 2.0 ** -24
    pol.update({"tol": max(tol_cli, derived), "tol_derived": derived, "ulp_bound_fp32": ulp_fp32,
                "fma_contraction_default": fn.get("fma_contraction_default"), "source": str(contract_path)})
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True); ap.add_argument("--m", type=int, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True); ap.add_argument("--reps", type=int, default=50); ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--tol", type=float, default=1e-4); ap.add_argument("--json", default=None); ap.add_argument("--keep", action="store_true")
    ap.add_argument("--amplify-warp", type=int, default=-1, help="race amplifier: this warp spins before each compute")
    ap.add_argument("--amplify-cycles", type=int, default=20000)
    ap.add_argument("--ncu", action="store_true", help="also profile one launch with ncu for DRAM and L2 bytes, when ncu is available")
    ap.add_argument("--contract", default=None, help="measured contract YAML: warm-up from B22 clock_warmup_s, tolerance from B26.5 (see derive_policy)")
    ap.add_argument("--warmup-s", type=float, default=None, help="override the warm-up time in seconds")
    a = ap.parse_args()
    policy = derive_policy(a.contract, a.k, a.tol, a.warmup_s)
    src = Path(a.kernel).read_text()
    digest = hashlib.sha256(src.encode()).hexdigest()[:12]
    work = Path(tempfile.mkdtemp(prefix=f"agentic-runner-{digest}-"))
    (work / "kernel.cu").write_text(src)
    shutil.copy(HERE / "harness.cu", work / "harness.cu")
    t0 = time.time()
    build = subprocess.run(["nvcc", "-O3", f"-arch={a.arch}", "-std=c++17", "-o", str(work / "run"), str(work / "harness.cu"), "-lcublas"],
                           capture_output=True, text=True)
    out = {"kernel_sha256": hashlib.sha256(src.encode()).hexdigest(), "arch": a.arch, "host": os.uname().nodename,
           "compile_seconds": round(time.time() - t0, 2), "compiled": build.returncode == 0}
    if build.returncode != 0:
        out.update({"ran": False, "error": build.stderr[-2000:]})
    else:
        args = [str(work / "run"), "--m", str(a.m), "--n", str(a.n), "--k", str(a.k), "--reps", str(a.reps), "--tol", str(policy["tol"]),
                "--warmup-s", str(policy["warmup_s"]), "--amplify-warp", str(a.amplify_warp), "--amplify-cycles", str(a.amplify_cycles)]
        out["policy"] = policy
        r = subprocess.run(args, capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        if line:
            out.update(json.loads(line))
        else:
            out.update({"ran": False, "error": (r.stdout + r.stderr)[-2000:]})
        if a.ncu and out.get("ran"):
            out["ncu"] = ncu_traffic(args)
    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    text = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
