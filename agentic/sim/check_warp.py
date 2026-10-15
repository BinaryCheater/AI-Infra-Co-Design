"""Checks of the warp model against ground truth it was not built from (B27_plus.md steps 3-4): B27's streaming ILP/TLP
grid and B29's generated-kernel loops at the bad corners. Prints the tables; returns the summaries.

  python -m agentic.sim.check_warp
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import load
from .warp import WarpModel, LDS32, FFMA, ALU

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "calibration" / "spark-6951" / "raw"
WARP_TO_NCU = {"issued": "not_selected", "not_selected": "not_selected", "scoreboard": "short_scoreboard", "lsu": "mio_throttle"}
NCU_FAMILY = {"not_selected": "issue", "selected": "issue", "math_pipe_throttle": "issue", "wait": "issue", "short_scoreboard": "scoreboard", "mio_throttle": "lsu", "lg_throttle": "lsu",
              "long_scoreboard": "copy", "barrier": "barrier", "no_instruction": "issue"}
WARP_FAMILY = {"issued": "issue", "not_selected": "issue", "scoreboard": "scoreboard", "lsu": "lsu"}


def b27_stream_grid(m: WarpModel):
    """B27 stream variant: per iteration a streaming ld.shared.32 then a dependent FADD per chain, L=16 unrolled, 4w warps per SM."""
    rows = []
    for f in sorted(RAW.iterdir()):
        if not (f.name.startswith("dependency_cycles_per_iter[variant=stream_") and f.name.endswith("L=16].json")):
            continue
        r = json.loads(f.read_text()); d = r["detail"]; a, w = d["a"], d["w"]
        # one block of 4w warps; emulate the iteration as a k-step with a loads then a dependent ALU-class consumers, 16 steps per body
        cyc, hist = m.phase(4 * w, 1, 16, a, a, 0, 0, 0)
        sim = cyc / 16
        rows.append({"a": a, "w": w, "measured": r["value"], "simulated": sim, "error": sim / r["value"] - 1})
    return rows


def b28_lds_curve(m: WarpModel, contract: dict):
    """B28's lds mix (4 x ld.shared.128 + 8 x ld.shared.32 per thread per step, one consumer, 8 steps per phase) at 1-6 blocks per SM."""
    from ..templates.smem_tiling.events import wavefronts
    # B28's addresses: the four 128-bit loads broadcast one address to the warp (lanes all equal), the eight 32-bit loads are lane-consecutive
    a_wf = tuple(wavefronts([0] * 32, 16) for _ in range(4)); b_wf = tuple(wavefronts([l * 4 for l in range(32)], 4) for _ in range(8))
    rows = []
    for b, d in sorted(((int(k), v) for k, v in (contract.get("lds_rate_by_residency") or {}).items())):
        cyc, hist = m.phase(8, b, 8, 1, 8, 4, 0, 0, 1, a_wf, b_wf)
        sim = cyc / 8
        rows.append({"b": b, "measured": d["cycles_per_step_per_block"], "simulated": sim, "error": sim / d["cycles_per_step_per_block"] - 1})
    return rows


def b29(m: WarpModel):
    rows = []
    # register-limited residency (ptxas: T1 255 registers -> 1 block, T5 ~100 -> 2); beyond it the harness's extra blocks ran serialized
    LIMIT = {"T1_128x256x16_S2_P1": 1, "T5_128x128x32_S3_P1": 2}
    for f in sorted(RAW.iterdir()):
        if not (f.name.startswith("inner_loop_ground_truth[") and "variant=stripped" in f.name):
            continue
        r = json.loads(f.read_text()); d = r["detail"]
        if not d.get("measured") or d["blocks_per_sm"] > LIMIT.get(d["kernel"], 6):
            continue
        v = d["design"]; rm, rn = v["Tm"] // 16, v["Tn"] // 16; b = d["blocks_per_sm"]
        ffma, l32, l128 = m.counts(rm, rn)
        from ..templates.smem_tiling.events import inner_loop_wavefronts
        from ..templates.smem_tiling import TilingDesign
        from ..templates.smem_tiling.sass import sass_counts
        from ..chips import load_chip, scope_chip
        from ..examples.gemm_tiling import tiling_contract
        from ..agents.evaluator.cuda import kernel_artifact
        from ..templates.smem_tiling import Workload
        WL = {"square_4096": (4096, 4096, 4096), "skinny_64x4096x4096": (64, 4096, 4096), "ragged_1000x1000x1000": (1000, 1000, 1000)}
        des = TilingDesign(v["Tm"], v["Tn"], v["Tk"], v["S"], v["P"], bool(v["beta"])); wf = inner_loop_wavefronts(des)
        T = tiling_contract(Workload(d["workload"], *WL[d["workload"]]), hw=scope_chip(load_chip("spark-6951-gb10")), precondition="exact"); T.setdefault("id", "t"); T.setdefault("template", "smem_tiling")
        src = kernel_artifact(T, {"id": "k", "values": v})["source"]; cnt = sass_counts(src, v["Tm"], v["Tn"], v["Tk"], None)["counts"]
        loop = v["Tk"] * (ffma + wf["instructions_per_step"] + 1); other = int(round(max(0.0, cnt["per_warp_per_phase"] - loop))) if cnt else 0
        cyc, hist = m.phase(8, b, v["Tk"], ffma, l32, l128, other, 1, 1, tuple(wf["a_wavefronts_per_load"]), tuple(wf["b_wavefronts_per_load"]))
        sim = cyc / v["Tk"]
        tot = sum(hist[k] for k in WARP_FAMILY) or 1
        sim_f = max(WARP_FAMILY, key=lambda k: hist[k]) if tot else None
        ncu = d["ncu"]; dom = ncu.get("dominant_stall"); ncu_f = NCU_FAMILY.get(dom)
        rows.append({"kernel": d["kernel"], "b": b, "measured": r["value"], "simulated": sim, "error": sim / r["value"] - 1, "sim_family": WARP_FAMILY.get(sim_f), "ncu_stall": dom, "ncu_family": ncu_f,
                     "agree": (WARP_FAMILY.get(sim_f) == ncu_f) if (sim_f and ncu_f) else None, "issue_active_measured": ncu.get("issue_active_fraction"), "issue_active_sim": hist["issued"] / (cyc * m.schedulers) if cyc else None})
    return rows


def main():
    cfg = load("spark-6951-gb10"); m = WarpModel(cfg["contract"], cfg["structural"])
    g = b27_stream_grid(m)
    print("B27 stream grid (cycles per iteration per warp), simulated vs measured:")
    for r in g:
        print(f"  a={r['a']:2d} w={r['w']}: sim {r['simulated']:6.1f}  meas {r['measured']:6.1f}  {r['error']:+6.0%}")
    e = sorted(abs(r["error"]) for r in g); print(f"  B27: {sum(1 for x in e if x <= 0.10)}/{len(e)} within 10 %, median {e[len(e)//2]:.0%}, worst {e[-1]:.0%}")
    c28 = b28_lds_curve(m, cfg["contract"])
    print("B28 lds mix, cycles per step per block vs resident blocks:")
    for r in c28:
        print(f"  b={r['b']}: sim {r['simulated']:6.0f}  meas {r['measured']:6.0f}  {r['error']:+6.0%}")
    rows = b29(m)
    print("B29 stripped kernels (cycles per k-step per block):")
    for r in rows:
        print(f"  {r['kernel']:<24} b={r['b']}: sim {r['simulated']:6.1f}  meas {r['measured']:6.1f}  {r['error']:+6.0%}   sim {r['sim_family']} / ncu {r['ncu_stall']} ({r['ncu_family']}) {'agree' if r['agree'] else 'DISAGREE'}   issue-active sim {r['issue_active_sim']:.2f} meas {r['issue_active_measured']:.2f}")
    e = sorted(abs(r["error"]) for r in rows); ag = [r["agree"] for r in rows if r["agree"] is not None]
    print(f"  B29: {sum(1 for x in e if x <= 0.10)}/{len(e)} within 10 %, median {e[len(e)//2]:.0%}, worst {e[-1]:.0%}; family agreement {sum(1 for x in ag if x)}/{len(ag)}")
    return g, c28, rows


if __name__ == "__main__":
    main()
