"""The real evaluator backend (notes/hardware_calibration_design.md §3): a solved tiling design is
compiled into a CUDA kernel, run on the GPU host through the runner, checked against cuBLAS, and the
measured time is recorded beside the prediction. Selected by evaluator_backend = "cuda:<ssh-alias>".

Utilizations for regime detection divide the design's byte and FLOP counts (from the cost model's
traffic terms, not its time) by the measured time and the measured contract's bandwidth and FP32
rate; a counter-based measurement (ncu) is a later refinement.
"""
from __future__ import annotations

import hashlib

from ...chips import hardware_contract
from ...templates.smem_tiling import Workload, cost_model
from ...templates.smem_tiling.codegen import codegen
from ...templates.smem_tiling.remote import run_remote
from ..verifier import decode

ARCH_BY_CC = {"12.1": "sm_121", "12.0": "sm_120", "10.0": "sm_100", "9.0": "sm_90"}


def kernel_artifact(T: dict, x: dict) -> dict:
    src = codegen(T, x)
    cc = str(T["scope"]["chip"].get("compute_capability", ""))
    return {"contract": T["id"], "assignment": x["id"], "template": T["template"], "language": "cuda",
            "arch": ARCH_BY_CC.get(cc), "source": src, "sha256": hashlib.sha256(src.encode()).hexdigest(),
            "provenance": {"generator": "agentic.templates.smem_tiling.codegen", "chip": T["scope"]["chip"].get("name")}}


def measure(T: dict, x: dict, host: str, kernel: dict, reps: int = 50, ncu: bool = True) -> dict:
    w = Workload(**T["scope"]["workload"])
    hw = hardware_contract(T["scope"]["chip"])
    chip = T["scope"]["chip"]
    d = decode(T, x["values"])
    cm = cost_model(d, w, hw)
    arch = kernel.get("arch") or "sm_121"
    r = run_remote(host, kernel["source"], w.m, w.n, w.k, reps=reps, arch=arch, ncu=ncu)
    if not r.get("ran"):
        raise RuntimeError(f"kernel did not run on {host}: {r.get('error', '?')[:400]}")
    measured = float(r["median_us"])
    t = measured * 1e-6
    flops = 2.0 * w.m * w.n * w.k
    # traffic: ncu's DRAM bytes when the profiler ran, else the cost model's count (then the regime label is modeled)
    prof = r.get("ncu") or {}
    dram_bytes = prof.get("dram_bytes") if prof.get("ok") else None
    regime_source = "measured" if dram_bytes is not None else "modeled"
    util = {"compute": flops / t / (hw.fp32_flops_per_s_per_sm * hw.sms),
            "DRAM": (dram_bytes if dram_bytes is not None else cm["dram_bytes"]) / t / hw.dram_bytes_per_s}
    dom = max(util, key=util.get)
    predicted = x["objective"]
    return {"contract": T["id"], "assignment": x["id"], "design": d.to_dict(),
            "predicted_cost": predicted, "measured_cost": measured,
            "discrepancy": (measured - predicted) / predicted if predicted else 0.0,
            "utilizations": util, "dominant_resource": dom,
            "simulator": f"cuda:{host}", "seed": 12345, "kernel": kernel["id"],
            "reference_agrees": bool(r.get("reference_agrees")), "ci95": r.get("ci95_us"),
            "clocks": {"after": r.get("clocks_after"), "sm_mhz_in_run": r.get("sm_clock_mhz_in_run"), "sm_mhz": r.get("sm_clock_mhz")},
            "host": r.get("host"), "regime_source": regime_source,
            "cycles": {"median": r.get("median_cycles"), "ci95": r.get("ci95_cycles"), "sm_mhz": r.get("sm_clock_mhz")},
            "detail": {"reps": r.get("reps"), "warmups": r.get("warmups"), "min_us": r.get("min_us"), "grid": r.get("grid"),
                       "threads": r.get("threads"), "smem_bytes": r.get("smem_bytes"), "cp_async": r.get("cp_async"),
                       "disagreeing": r.get("disagreeing"), "first_disagreeing_rc": r.get("first_disagreeing_rc"),
                       "worst_rel_error": r.get("worst_rel_error"), "tolerance": r.get("tolerance"), "reference": r.get("reference"),
                       "dram_bytes_modeled": cm["dram_bytes"], "dram_bytes_ncu": dram_bytes, "l2_bytes_ncu": prof.get("l2_bytes"),
                       "ncu": {k: v for k, v in prof.items() if k != "raw"}, "flops": flops, "compile_seconds": r.get("compile_seconds"),
                       "kernel_sha256": r.get("kernel_sha256"), "chip": chip.get("name")},
            "provenance": {"service": "agentic.agents.evaluator.cuda", "backend": f"cuda:{host}", "arch": arch}}
