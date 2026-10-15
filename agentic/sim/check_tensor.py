"""The tensor model (sim/tensor.py) against B31-B34 and the C1 composite, as predictions: every parameter is a contract number and
the programs are the benchmarks' loops. B31's chain and saturated point are inputs (latency, issue cycles); everything else is a check.

  python -m agentic.sim.check_tensor            # prints the tables; exit 1 if the composite misses 10 % at any residency
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load
from .tensor import (TensorModel, prog_b31, prog_b32_pipelined, prog_b32_dependent, prog_b32_ldsm_latency, prog_b33_load, prog_b34_ratio,
                     prog_ffma_only, prog_c1_warp)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "calibration" / "spark-6951" / "raw"


def raw(q):
    f = RAW / (q.replace(",", "_") + ".json")          # the driver writes commas in a quantity's key as underscores
    return json.loads(f.read_text()) if f.exists() else None


def val(q):
    r = raw(q)
    return r["value"] if r else None


def dominant(hist):
    fam = {k: v for k, v in hist.items() if k not in ("issued", "done")}
    return max(fam, key=fam.get) if fam else "-"


def row(name, measured, simulated, hist=None):
    err = simulated / measured - 1 if measured else float("nan")
    print(f"  {name:<44} measured {measured:8.1f}  model {simulated:8.1f}  {err:+6.1%}  {dominant(hist) if hist else ''}")
    return err


def b31(m: TensorModel):
    print("B31 (cycles per mma per warp; a accumulators, w warps per sub-partition) -- the chain and a=16,w=1 are inputs")
    out = []
    for shape in ("bf16_m16n8k16", "tf32_m16n8k8", "fp8_e4m3_m16n8k32"):
        for a, w in ((1, 1), (1, 2), (1, 4), (2, 1), (4, 1), (4, 2), (4, 4), (16, 1), (16, 2), (16, 4)):
            meas = val(f"mma_cycles_per_op[shape={shape},a={a},w={w}]")
            if meas is None:
                continue
            it = 40
            cyc, h = m.run([prog_b31(shape, a, it) for _ in range(4 * w)], 4 * w)
            out.append(row(f"{shape} a={a} w={w}", meas, cyc / (it * a), h))
    return out


def b32(m: TensorModel):
    print("B32 (cycles per k-step per warp: 2 ldmatrix.x4 + 4 mma; 1 block = 4 warps)")
    out = []
    for depth in (1, 2, 3):
        for bps in (1, 2):
            meas = val(f"ldmatrix_mma_cycles_per_step[swizzled_depth{depth},bps={bps}]")
            if meas is None:
                continue
            trips = 6
            cyc, h = m.run([prog_b32_pipelined(depth, trips) for _ in range(4 * bps)], 4)
            out.append(row(f"pipelined depth={depth} bps={bps}", meas, cyc / (trips * 16), h))
    meas = val("ldmatrix_mma_cycles_per_step[dependent_swizzled,bps=1]")
    if meas:
        cyc, h = m.run([prog_b32_dependent(40) for _ in range(4)], 4)
        out.append(row("dependent chain (per link)", meas, cyc / 40, h))
    for x in (1, 2, 4):
        meas = val(f"ldmatrix_latency_cycles[x{x}]")
        if meas:
            cyc, h = m.run([prog_b32_ldsm_latency(x, 40) for _ in range(4)], 4)
            out.append(row(f"ldmatrix.x{x} latency link", meas, cyc / 40, h))
    return out


def b33(m: TensorModel):
    print("B33 (tensor block's cycles per k-step with load blocks on the same SM; 4 warps per block)")
    out = []
    for load in ("tensor_only", "lds128", "cp_async", "both"):
        for pairs in (1, 2):
            meas = val(f"tensor_under_load[load={load},tensor_blocks={pairs}]")
            if meas is None:
                continue
            trips = 6
            progs = [prog_b32_pipelined(2, trips) for _ in range(4 * pairs)]
            if load != "tensor_only":
                progs += [prog_b33_load(load, trips) for _ in range(4 * pairs)]
            # the tensor block's own time: run until the tensor warps finish; the load loop is open-ended in the benchmark, so stop the
            # load programs at the same length by measuring the tensor warps' completion from a trace
            done = {}
            def tr(t, w, s, ins): done[w] = t
            cyc, h = m.run(progs, 4, trace=tr)
            tensor_end = max(done.get(w, 0) for w in range(4 * pairs))
            out.append(row(f"{load} {pairs}+{pairs if load != 'tensor_only' else 0}", meas, tensor_end / (trips * 16), h))
    return out


def b34(m: TensorModel):
    print("B34 (a) (cycles per mma with ff FFMAs between mma, one warp per sub-partition) and (a2) split warps")
    out = []
    for ff in (0, 1, 4, 16):
        meas = val(f"mma_ffma_cycles_per_mma[ffma_per_mma={ff}]")
        if meas is None:
            continue
        it = 40
        cyc, h = m.run([prog_b34_ratio(ff, it) for _ in range(4)], 4)
        out.append(row(f"ffma_per_mma={ff}", meas, cyc / (4 * it), h))
    mt = val("mma_split_cycles_per_mma[together]"); ft = val("ffma_split_cycles_per_ffma[together]")
    if mt and ft:
        it = 40
        progs = [prog_b31("bf16_m16n8k16", 4, it) for _ in range(4)] + [prog_ffma_only(8, 4 * it) for _ in range(4)]
        done = {}
        def tr(t, w, s, ins): done[w] = t
        cyc, h = m.run(progs, 4, trace=tr)
        out.append(row("split: mma warp, together", mt, max(done[w] for w in range(4)) / (4 * it), h))
        out.append(row("split: ffma warp, together", ft, max(done[w] for w in range(4, 8)) / (8 * 4 * it), h))
    return out


def c1(m: TensorModel):
    print("C1 composite (cycles per k-step per block; 128x128x32 bf16, 8 warps per block) -- a prediction")
    out = []
    for variant, kw in (("", {}), ("no_copy_", {"copies": False}), ("copy_only_", {"compute": False})):
        for bps in (1, 2, 3):
            r = raw(f"tensor_tile_{variant}cycles_per_step[bps={bps}]")
            if not r:
                continue
            d = r["detail"]; resident = d.get("resident_blocks_per_sm", bps)
            steps, passes = 16, 2
            cyc, h = m.run([prog_c1_warp(steps, passes, **kw) for _ in range(8 * resident)], 8)
            err = row(f"{variant or 'kernel '}bps={bps} (resident {resident})", r["value"], cyc / (steps * passes), h)
            if not variant:
                out.append((bps, err, h))
    return out


def main():
    cfg = load("spark-6951-gb10"); m = TensorModel(cfg["contract"], cfg["structural"])
    print(f"inputs: mma issue {m.issue_cycles('bf16_m16n8k16')} cycles, latency {m.latency('bf16_m16n8k16')}, ldmatrix latency {m.ldsm_lat}, mma holds warp {m.mma_holds_warp}, copy latency {m.copy_latency_cycles:.0f} cycles")
    b31(m); b32(m); b33(m); b34(m); res = c1(m)
    bad = [b for b, e, _ in res if abs(e) > 0.10]
    print("composite:", "within 10 % at every residency" if not bad else f"misses at bps {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
