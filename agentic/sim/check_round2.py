"""Groups 2 and 3 of round 2 as predictions: the TMA engine (sim/memory.py) against B35-B37's own points and the C2 composite, the
sfu / shfl / cvt classes and the register budget (sim/warp.py, the runner in sim/tensor.py) against B38-B40's points and the C3
composite. Every parameter is a contract number; nothing is fitted.

  python -m agentic.sim.check_round2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load
from .memory import Hierarchy
from .tensor import TensorModel, Instr, prog_c3_row, prog_ffma_only, SFU, SHFL, CVT, ALU, FFMA
from .warp import WarpModel
from .check_tensor import raw, val, row, dominant

ROOT = Path(__file__).resolve().parents[2]


def group2(cfg, h: Hierarchy, wm: WarpModel):
    print("B35 / B36 as the engine reads them (inputs) and B36's mixes as predictions of the shared path")
    c = cfg["contract"]; out = []
    t = h.tma
    for res in (1, 2, 4):
        r = raw(f"tma_latency_dist[tile=16384,resident={res}]")
        if r: print(f"  TMA 16 KB p50 at {res} resident: {r['detail']['p50_cycles']:.0f} cycles, p99 {r['detail']['p99_cycles']:.0f}, per-SM {r['detail'].get('bytes_per_s_per_sm', 0)/1e9:.0f} GB/s (input)")
    b8 = (c.get("tma_bulk_latency_s") or {}).get("16384")
    if b8:
        out.append(row("B35 16 KB p50 at 1 resident vs B8's one-block TMA latency", float(b8) * t.hz, t.latency_cycles(16384, 1)))
    mix = c.get("tma_mix_bytes_per_s") or {}
    if mix.get("tma4") and mix.get("cp4") and mix.get("tma1_cp1"):
        # the shared-path prediction: the 1:1 mix moves bytes at the mean of the two pure rates (one path, two issuers)
        out.append(row("B36 1:1 mix bytes/s (shared-path rule: mean of pure)", mix["tma1_cp1"] / 1e9, (mix["tma4"] + mix["cp4"]) / 2 / 1e9))
    print("C2 composite (T5 128x128 FP32 tile with 2-D TMA, cycles per k-step per block) -- a prediction")
    res_out = []
    for tk in (16, 32):
        for bps in (1, 2, 3):
            r = raw(f"tma_tile_cycles_per_step[tk={tk},bps={bps}]")
            if not r or r["detail"].get("measured") is False:
                continue
            d = r["detail"]; resident = d.get("resident_blocks_per_sm", bps); stage = d["stage_bytes"]
            # the compute phase: the warp model's T5 loop (RM = RN = 8: 64 FFMA, 8 lds32, 2 lds128 per k-step), every resident block's warps
            ffma, l32, l128 = wm.counts(8, 8)
            cyc, hist = wm.phase(8, resident, tk, ffma, l32, l128, 0)
            barrier = 2 * float(c["barrier_s"]["256"] if isinstance(c["barrier_s"], dict) else c["barrier_s"]) * t.hz
            compute = cyc + barrier
            # the copies: two 2-D tiles of stage/2 bytes each, one mbarrier, issued one step ahead; the stage lands after the engine's time
            # the stage: two 2-D copies issued together by every resident block on every SM; the engine's rule (sim/memory.py) gives
            # the stage's completion; per step the queue carries resident x stage bytes per SM
            from .run import Fifo
            t.chip_q = Fifo(); t0 = 0.0; end = t0
            for _ in range(2 * resident):                       # two 2-D copies (A and B) per block per stage, every resident block at once
                end = max(end, t.complete(0, t0, stage // 2, resident, two_d=True))
            copy = (end - t0) * t.hz + t.mbarrier_cycles
            pred = max(compute, copy)
            fam = "copy" if copy > compute else dominant(hist)
            err = row(f"tk={tk} bps={bps} (resident {resident})  compute {compute:.0f}, copy {copy:.0f}", r["value"], pred); print(f"        model family: {fam}")
            res_out.append((tk, bps, err))
            for variant, pv in (("copy_only_", copy), ("compute_only_", compute)):
                rv = raw(f"tma_tile_{variant}cycles_per_step[tk={tk},bps={bps}]")
                if rv: row(f"    {variant}tk={tk} bps={bps}", rv["value"], pv)
    return out, res_out


def group3(cfg, m: TensorModel):
    print("B38 / B39 points as the classes read them (inputs: throughput and latency) and the mixes as predictions")
    wm = m.wm; out = []
    for ff in (0, 1, 4, 16):
        meas = val(f"sfu_ffma_mixed_cycles_per_iter[ffma_per_ex2={ff}]")
        if meas is None: continue
        it = 40; p = []
        for _ in range(it):
            for i in range(8): p.append(Instr(SFU, reads=(f"x{i}",), writes=f"x{i}", op="ex2"))
            for r in range(ff):
                for i in range(8): p.append(Instr(FFMA, reads=(f"f{i}",), writes=f"f{i}"))
            p.append(Instr(ALU))
        cyc, h = m.run([p for _ in range(4)], 4)
        out.append(row(f"B38 8 ex2 + {ff}x8 FFMA per iteration", meas, cyc / it, h))
    wr = val("warp_reduce_cycles")
    if wr:
        it = 40; p = []
        for _ in range(it):
            for d in range(5): p.append(Instr(SHFL, reads=("v",), writes="w", op="bfly")); p.append(Instr(FFMA, reads=("v", "w"), writes="v"))
            p.append(Instr(FFMA, reads=("v",), writes="v")); p.append(Instr(ALU))
        cyc, h = m.run([p for _ in range(4)], 4)
        out.append(row("B39 butterfly reduction (5 shfl + 5 FADD + FMUL)", wr, cyc / it, h))
    bl = val("shfl_cycles_beside_lds[lds=1]"); al = val("shfl_cycles_beside_lds[lds=0]")
    if bl and al:
        from .tensor import LDS128
        it = 40; ps = []; pl = []
        for _ in range(it):
            for i in range(8): ps.append(Instr(SHFL, reads=(f"v{i}",), writes=f"v{i}", op="bfly"))
            ps.append(Instr(ALU))
            for u in range(8): pl.append(Instr(LDS128, writes=f"l{u}", wavefronts=4.0)); pl.append(Instr(FFMA, reads=(f"l{u}", "acc"), writes="acc"))
            pl.append(Instr(ALU))
        done = {}
        def tr(t, w, s, ins): done[w] = t
        cyc, h = m.run([ps for _ in range(4)] + [pl for _ in range(4)], 8, trace=tr)
        out.append(row("B39 shuffles beside LDS.128 warps (cycles per shfl)", bl, max(done[w] for w in range(4)) / (it * 8), h))
    print("C3 composite (online softmax rows, cycles per row per block; 4 warps per block) -- a prediction")
    res_out = []
    for bps in (1, 2, 4, 6):
        r = raw(f"softmax_cycles_per_row[bps={bps}]")
        if not r: continue
        rows = 24
        cyc, h = m.run([prog_c3_row(rows) for _ in range(4 * bps)], 4)
        err = row(f"bps={bps}", r["value"], cyc / rows, h)
        res_out.append((bps, err, h))
    return out, res_out


def main():
    cfg = load("spark-6951-gb10"); C, S, F = cfg["contract"], cfg["structural"], cfg["fixed"]
    h = Hierarchy(C, S, F); wm = WarpModel(C, S); m = TensorModel(C, S)
    print(f"inputs: sfu cost {wm.sfu_cost}, sfu latency {wm.sfu_lat}, shfl cost {wm.shfl_cost}, shfl latency {wm.shfl_lat}, shfl in LSU queue {wm.shfl_in_lsu_queue}, cvt cost {wm.cvt_cost}")
    _, c2 = group2(cfg, h, wm); _, c3 = group3(cfg, m)
    bad2 = [(tk, b) for tk, b, e in c2 if abs(e) > 0.10]; bad3 = [b for b, e, _ in c3 if abs(e) > 0.15]
    print("C2:", "within 10 % at every point" if not bad2 else f"misses at {bad2}"); print("C3:", "within 15 % at every residency" if not bad3 else f"misses at bps {bad3}")
    return 1 if bad2 or bad3 else 0


if __name__ == "__main__":
    sys.exit(main())
