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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True); ap.add_argument("--m", type=int, required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True); ap.add_argument("--reps", type=int, default=50); ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--tol", type=float, default=1e-4); ap.add_argument("--json", default=None); ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
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
        r = subprocess.run([str(work / "run"), "--m", str(a.m), "--n", str(a.n), "--k", str(a.k), "--reps", str(a.reps), "--tol", str(a.tol)],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        if line:
            out.update(json.loads(line))
        else:
            out.update({"ran": False, "error": (r.stdout + r.stderr)[-2000:]})
    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    text = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
