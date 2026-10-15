"""Step 4 of the instruction-stream round: the thirty GEMM designs and composites C1-C3 through both inputs of the warp model,
`counts` (per-step instruction counts, the G4 model) and `stream` (the kernel's SASS stream, sim/sass.py). Nothing else changes:
the event-level simulator, the memory hierarchy, the copy engines and the composition rules of check_tensor / check_round2 are the
same for both columns. Writes calibration/spark-6951/input_comparison.json and prints the table.

  python -m agentic.sim.compare_input
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

from .config import load
from .isa import PipeCosts
from .memory import Hierarchy
from .run import SimConfig, Fifo, block_waves
from .sass import Function, function_text, read_dump, gemm_phase_stream
from .tensor import TensorModel, prog_c1_warp, prog_c3_row
from .tiling import simulate_design
from .validate import WORKLOADS, instr_per_phase_for
from .warp import WarpModel
from ..chips import load_chip, scope_chip
from ..templates.smem_tiling import TilingDesign

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "calibration" / "spark-6951" / "raw"
OUT = ROOT / "calibration" / "spark-6951" / "input_comparison.json"


def raw(q):
    f = RAW / (q.replace(",", "_") + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def _one_gemm(i: int) -> dict:
    """One design through both inputs, in its own process (the simulator is single-threaded Python)."""
    cfg = load("spark-6951-gb10"); C, S, F = cfg["contract"], cfg["structural"], cfg["fixed"]; sc = SimConfig(C, S, F, "spark-6951-gb10")
    costs = PipeCosts(C); chip = scope_chip(load_chip("spark-6951-gb10"))
    v = json.loads((ROOT / "calibration" / "spark-6951" / "validation.json").read_text())
    return _gemm_row(v["rows"][i], C, S, F, sc, costs, chip)


def _gemm_row(row, C, S, F, sc, costs, chip) -> dict:
    d = row["design"]; w = next(x for x in WORKLOADS if x.name == row["workload"])
    des = TilingDesign(**{k: val for k, val in d.items() if k in TilingDesign.__dataclass_fields__})
    ipp = instr_per_phase_for(copy.deepcopy(row), w, chip, None)
    res = {}
    for mode in ("counts", "stream"):
        wm = WarpModel(C, dict(S, input=mode))
        st = gemm_phase_stream(row["sass_key"], d["tm"], d["tn"], d["tk"], costs)[0] if mode == "stream" else None
        r = simulate_design(des, w, sc, Hierarchy(C, S, F), issue=wm, instr_per_phase=ipp, stream=st, costs=costs)
        fam = r.get("critical_counts") or {}
        res[mode] = {"us": r["time_us"], "critical": max(fam, key=fam.get) if fam else None}
    return {"kind": "gemm", "name": f"{row['workload']} {d['tm']}x{d['tn']}x{d['tk']} S{d['stages']} P{d['prefetch']}", "measured": row["measured_us"], "unit": "us",
            "counts": res["counts"]["us"], "stream": res["stream"]["us"], "counts_critical": res["counts"]["critical"], "stream_critical": res["stream"]["critical"],
            "ncu_family": row.get("ncu_dominant_family")}


def gemm_rows(C, S, F, sc, costs, chip, jobs: int = 1):
    v = json.loads((ROOT / "calibration" / "spark-6951" / "validation.json").read_text())
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        # the largest designs first, so the slowest ones start at once
        W = {x.name: x for x in WORKLOADS}
        size = lambda r: -(-W[r["workload"]].m // r["design"]["tm"]) * -(-W[r["workload"]].n // r["design"]["tn"]) * -(-W[r["workload"]].k // r["design"]["tk"])
        order = sorted(range(len(v["rows"])), key=lambda i: -size(v["rows"][i]))
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            got = dict(zip(order, ex.map(_one_gemm, order)))
        out = [got[i] for i in range(len(v["rows"]))]
        for r in out:
            print(f"  {r['name']:<38} measured {r['measured']:9.1f}  counts {r['counts']:9.1f} ({r['counts'] / r['measured'] - 1:+6.1%})  stream {r['stream']:9.1f} ({r['stream'] / r['measured'] - 1:+6.1%})", flush=True)
        return out
    out = []
    for row in v["rows"]:
        d = row["design"]; w = next(x for x in WORKLOADS if x.name == row["workload"])
        des = TilingDesign(**{k: val for k, val in d.items() if k in TilingDesign.__dataclass_fields__})
        ipp = instr_per_phase_for(copy.deepcopy(row), w, chip, None)
        res = {}
        for mode in ("counts", "stream"):
            wm = WarpModel(C, dict(S, input=mode))
            st = gemm_phase_stream(row["sass_key"], d["tm"], d["tn"], d["tk"], costs)[0] if mode == "stream" else None
            r = simulate_design(des, w, sc, Hierarchy(C, S, F), issue=wm, instr_per_phase=ipp, stream=st, costs=costs)
            fam = r.get("critical_counts") or {}
            res[mode] = {"us": r["time_us"], "critical": max(fam, key=fam.get) if fam else None, "warp_hist": {k: round(val, 9) for k, val in (r.get("wait_families_s") or {}).items()}}
        out.append({"kind": "gemm", "name": f"{row['workload']} {d['tm']}x{d['tn']}x{d['tk']} S{d['stages']} P{d['prefetch']}", "measured": row["measured_us"], "unit": "us",
                    "counts": res["counts"]["us"], "stream": res["stream"]["us"], "counts_critical": res["counts"]["critical"], "stream_critical": res["stream"]["critical"],
                    "ncu_family": row.get("ncu_dominant_family")})
        print(f"  {out[-1]['name']:<38} measured {row['measured_us']:9.1f}  counts {res['counts']['us']:9.1f} ({res['counts']['us'] / row['measured_us'] - 1:+6.1%})  stream {res['stream']['us']:9.1f} ({res['stream']['us'] / row['measured_us'] - 1:+6.1%})", flush=True)
    return out


def c1_rows(C, S, costs):
    """C1: counts = sim/tensor.py's hand-built program (check_tensor); stream = the kernel's SASS pass loop with the k loop x 16 and
    each copy loop x 2, two passes, in kernel mode (barriers, copy groups with B8's stage latency and the chip copy queue)."""
    tm = TensorModel(C, S)
    F = Function(function_text(read_dump("c1_tensor_tile"), "tensor_tileILi0EE"))
    k = F.inner_loop(("HMMA",), exclude=()); pas = F.enclosing(k)
    # k-steps per pass of the loop body from its content (32 HMMA per k-step per warp): the compiler may unroll the step loop
    trips = {k: max(1, 16 // max(1, round(F.count(k, ("HMMA",)) / 32)))}
    for r in F.loops:
        if pas[0] <= r[0] and r[1] <= pas[1] and r not in (k, pas) and F.count(r, ("LDGSTS",)):
            trips[r] = 2
    one_pass = F.walk(pas[0], pas[1], trips, costs)
    stream = one_pass * 2
    wm = WarpModel(C, dict(S, input="stream"))
    out = []
    per_res = {}                                       # residency -> (counts, stream) cycles per step
    for bps in (1, 2, 3):
        r = raw(f"tensor_tile_cycles_per_step[bps={bps}]")
        if not r: continue
        res = r["detail"].get("resident_blocks_per_sm", bps)
        if res not in per_res:
            cc, _ = tm.run([prog_c1_warp(16, 2) for _ in range(8 * res)], 8)
            cs, hs = wm.run_stream([stream] * 8, 8, res, costs, mode="kernel", copy_rule=tm.stage_cycles, stage_bytes=16384)
            per_res[res] = (cc / 32, cs / 32)
        if bps > res:                                  # more blocks than fit: the block scheduler runs the rest as a second wave
            steps = 8 * 32
            dc = block_waves(bps, res, steps, {k: v[0] for k, v in per_res.items()}); ds = block_waves(bps, res, steps, {k: v[1] for k, v in per_res.items()})
            med = lambda xs: sorted(xs)[len(xs) // 2]
            out.append({"kind": "C1", "name": f"C1 bf16 tile bps={bps} (resident {res}, second wave)", "measured": r["value"], "unit": "cycles/step",
                        "counts": med(dc) / steps, "stream": med(ds) / steps, "block_durations_stream": [x / steps for x in ds]})
        else:
            out.append({"kind": "C1", "name": f"C1 bf16 tile bps={bps} (resident {res})", "measured": r["value"], "unit": "cycles/step", "counts": per_res[res][0], "stream": per_res[res][1]})
    # the kernel's two halves (the C1 no-copy and copy-only variants are compiled kernels of their own)
    for variant, fn, kw in (("no_copy_", "tensor_tileILi2EE", {"copies": False}), ("copy_only_", "tensor_tileILi1EE", {"compute": False})):
        G = Function(function_text(read_dump("c1_tensor_tile"), fn))
        kv = G.inner_loop(("HMMA",), exclude=()) if variant == "no_copy_" else G.inner_loop(("BAR",), exclude=())
        pv = G.enclosing(kv) or kv
        per_body = round(G.count(kv, ("HMMA",)) / 32) if variant == "no_copy_" else round(G.count(kv, ("BAR",)) / 2)
        tr = {kv: max(1, 16 // max(1, per_body))}
        for r in G.loops:
            if pv[0] <= r[0] and r[1] <= pv[1] and r not in (kv, pv) and G.count(r, ("LDGSTS",)):
                tr[r] = 2
        sv = G.walk(pv[0], pv[1], tr, costs) * 2
        for bps in (1, 2):
            r = raw(f"tensor_tile_{variant}cycles_per_step[bps={bps}]")
            if not r: continue
            res = r["detail"].get("resident_blocks_per_sm", bps)
            cc, _ = tm.run([prog_c1_warp(16, 2, **kw) for _ in range(8 * res)], 8)
            cs, _ = wm.run_stream([sv] * 8, 8, res, costs, mode="kernel", copy_rule=tm.stage_cycles, stage_bytes=16384)
            out.append({"kind": "C1", "name": f"C1 {variant.rstrip('_').replace('_', '-')} bps={bps}", "measured": r["value"], "unit": "cycles/step", "counts": cc / 32, "stream": cs / 32})
    return out, {"pass_loop": [hex(pas[0]), hex(pas[1])], "k_loop": [hex(k[0]), hex(k[1])], "instructions_per_pass": len(one_pass)}


def c2_rows(C, S, costs):
    """C2: compute phase (counts: the warp model's T5 counts; stream: the kernel's SASS k-step loop) + two barriers, overlapped with
    the TMA engine's stage time, as check_round2 composes them."""
    h = Hierarchy(C, S, {}); t = h.tma
    barrier = 2 * float(C["barrier_s"]["256"] if isinstance(C["barrier_s"], dict) else C["barrier_s"]) * t.hz
    out = []; c2_res = {}
    for tk in (16, 32):
        F = Function(function_text(read_dump("c2_tma_tile"), f"tma_tileILi{tk}ELi0EE"))
        kk = F.inner_loop(("FFMA",), exclude=("BAR",)); step = F.enclosing(kk)
        unroll = max(1, round(F.count(kk, ("FFMA",)) / 64)); stream = F.walk(step[0], step[1], {kk: -(-tk // unroll)}, costs)
        for bps in (1, 2, 3):
            r = raw(f"tma_tile_cycles_per_step[tk={tk},bps={bps}]")
            if not r or r["detail"].get("measured") is False: continue
            d = r["detail"]; res = d.get("resident_blocks_per_sm", bps); stage = d["stage_bytes"]
            wm = WarpModel(C, dict(S, input="counts")); ffma, l32, l128 = wm.counts(8, 8)
            cc, _ = wm.phase(8, res, tk, ffma, l32, l128, 0)
            ws = WarpModel(C, dict(S, input="stream")); ws.set_stream([stream] * 8, costs)
            cs, _ = ws.phase(8, res, tk, ffma, l32, l128, 0)
            t.chip_q = Fifo(); end = 0.0
            for _ in range(2 * res):
                end = max(end, t.complete(0, 0.0, stage // 2, res, two_d=True))
            copy_c = end * t.hz + t.mbarrier_cycles
            per = {"counts": max(cc + barrier, copy_c), "stream": max(cs + barrier, copy_c)}
            c2_res[(tk, res)] = per
            if bps > res and (tk, 1) in c2_res:            # a second wave, with the one-block step time for the blocks that run alone
                steps = (4 if tk == 16 else 2) * (1024 // tk); med = lambda xs: sorted(xs)[len(xs) // 2]
                tab = lambda m: {rr: c2_res[(tk, rr)][m] for rr in (1, res) if (tk, rr) in c2_res}
                dc = block_waves(bps, res, steps, tab("counts")); ds = block_waves(bps, res, steps, tab("stream"))
                out.append({"kind": "C2", "name": f"C2 T5+TMA tk={tk} bps={bps} (resident {res}, second wave)", "measured": r["value"], "unit": "cycles/step",
                            "counts": med(dc) / steps, "stream": med(ds) / steps, "block_durations_stream": [x / steps for x in ds]})
                continue
            out.append({"kind": "C2", "name": f"C2 T5+TMA tk={tk} bps={bps} (resident {res})", "measured": r["value"], "unit": "cycles/step",
                        "counts": per["counts"], "stream": per["stream"], "compute_counts": cc + barrier, "compute_stream": cs + barrier, "copy": copy_c})
            rc = raw(f"tma_tile_compute_only_cycles_per_step[tk={tk},bps={bps}]")
            if rc and bps < 3:
                G = Function(function_text(read_dump("c2_tma_tile"), f"tma_tileILi{tk}ELi2EE"))
                kv = G.inner_loop(("FFMA",), exclude=("BAR",)); sv = G.enclosing(kv)
                uv = max(1, round(G.count(kv, ("FFMA",)) / 64)); stv = G.walk(sv[0], sv[1], {kv: -(-tk // uv)}, costs)
                wv = WarpModel(C, dict(S, input="stream")); wv.set_stream([stv] * 8, costs); cv, _ = wv.phase(8, res, tk, ffma, l32, l128, 0)
                out.append({"kind": "C2", "name": f"C2 compute-only tk={tk} bps={bps}", "measured": rc["value"], "unit": "cycles/step", "counts": cc + barrier, "stream": cv + barrier})
    return out


def c3_rows(C, S, costs):
    """C3: counts = the hand-built row program (sim/tensor.py prog_c3_row, one row after another); stream = the kernel's SASS row
    loop (the compiler's body holds two rows), run in kernel mode."""
    tm = TensorModel(C, S)
    F = Function(function_text(read_dump("c3_softmax"), "softmax_rows"))
    lo, hi = F.inner_loop(("MUFU",)); body = F.walk(lo, hi, {}, costs, _skip_loop=(lo, hi))
    rows_per_iter = sum(1 for i in body if i.opcode.startswith("LDG")); iters = 12
    stream = body * iters
    wm = WarpModel(C, dict(S, input="stream"))
    out = []
    for bps in (1, 2, 4, 6):
        r = raw(f"softmax_cycles_per_row[bps={bps}]")
        if not r: continue
        cc, _ = tm.run([prog_c3_row(24) for _ in range(4 * bps)], 4)
        cs, hs = wm.run_stream([stream] * 4, 4, bps, costs, mode="kernel")
        out.append({"kind": "C3", "name": f"C3 softmax rows bps={bps}", "measured": r["value"], "unit": "cycles/row", "counts": cc / 24, "stream": cs / (iters * rows_per_iter),
                    "stream_hist": {k2: v for k2, v in hs.items() if isinstance(v, (int, float))}})
    return out, {"loop": [hex(lo), hex(hi)], "rows_per_iteration": rows_per_iter, "instructions_per_iteration": len(body)}


def _composite(name: str):
    cfg = load("spark-6951-gb10"); C, S = cfg["contract"], cfg["structural"]; costs = PipeCosts(C)
    return {"c1": lambda: c1_rows(C, S, costs), "c2": lambda: (c2_rows(C, S, costs), None), "c3": lambda: c3_rows(C, S, costs)}[name]()


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1, help="processes: the designs and composites run in parallel")
    ap.add_argument("--composites-only", action="store_true", help="rerun C1-C3 only; the GEMM rows are taken from input_comparison.json")
    a = ap.parse_args()
    cfg = load("spark-6951-gb10"); C, S, F = cfg["contract"], cfg["structural"], cfg["fixed"]; sc = SimConfig(C, S, F, "spark-6951-gb10")
    costs = PipeCosts(C); chip = scope_chip(load_chip("spark-6951-gb10"))
    print("thirty GEMM designs (us per kernel)")
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=3) as ex:
            fut = {n: ex.submit(_composite, n) for n in ("c1", "c2", "c3")}
            if a.composites_only:
                rows = [r for r in json.loads(OUT.read_text())["rows"] if r["kind"] == "gemm"]
            else:
                rows = gemm_rows(C, S, F, sc, costs, chip, jobs=max(1, a.jobs - 3))
            (c1, c1_info), (c2, _), (c3, c3_info) = fut["c1"].result(), fut["c2"].result(), fut["c3"].result()
    else:
        rows = gemm_rows(C, S, F, sc, costs, chip)
        c1, c1_info = c1_rows(C, S, costs); c2 = c2_rows(C, S, costs); c3, c3_info = c3_rows(C, S, costs)
    allrows = rows + c1 + c2 + c3
    for r in allrows:
        r["counts_error"] = r["counts"] / r["measured"] - 1; r["stream_error"] = r["stream"] / r["measured"] - 1
    g = [r for r in allrows if r["kind"] == "gemm"]
    summ = {}
    for mode in ("counts", "stream"):
        e = sorted(abs(r[f"{mode}_error"]) for r in g)
        summ[mode] = {"median_abs": statistics.median(e), "worst": e[-1], "within_10": sum(1 for x in e if x <= 0.10), "within_20": sum(1 for x in e if x <= 0.20)}
    print("\n| design | measured | counts | error | stream | error |\n|---|---|---|---|---|---|")
    for r in allrows:
        print(f"| {r['name']} | {r['measured']:.1f} {r['unit']} | {r['counts']:.1f} | {r['counts_error']:+.1%} | {r['stream']:.1f} | {r['stream_error']:+.1%} |")
    print(json.dumps(summ, indent=1))
    OUT.write_text(json.dumps({"rows": allrows, "summary_gemm": summ, "c1_stream": c1_info, "c3_stream": c3_info}, indent=1) + "\n")
    c3s = {r["name"]: r for r in c3}
    moved = all(abs(c3s[f"C3 softmax rows bps={b}"]["stream_error"]) < abs(c3s[f"C3 softmax rows bps={b}"]["counts_error"]) for b in (1, 2) if f"C3 softmax rows bps={b}" in c3s)
    print("softmax rows at 1 and 2 blocks moved toward the card:", moved)
    return 0 if moved else 2


if __name__ == "__main__":
    sys.exit(main())
