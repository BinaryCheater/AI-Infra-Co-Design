"""G4 validation set (notes/mechanistic_simulator_design.md §7): designs sampled from the feasible menu on the three
workloads, measured on the Spark through the runner (clock sampled in-run, ncu traffic and stall reasons), simulated
with the mechanistic model, compared.  Writes calibration/<host>/validation.json.

  python -m agentic.sim.validate --host dgx-spark --per-workload 10 [--seed 7] [--reps 30]

The sample is deterministic (seed) over the feasible designs of each workload under the exact precondition; it keeps
the spark-001 kernels' designs out so the set is independent of what G1-G4a were tuned against (nothing is tuned,
but the acceptance should not re-use its own evidence).
"""
from __future__ import annotations

import argparse, json, random, time
from itertools import product
from pathlib import Path

from ..chips import load_chip, scope_chip
from ..examples.gemm_tiling import tiling_contract
from ..agents.evaluator.cuda import kernel_artifact
from ..agents.verifier import decode
from ..chips import hardware_contract
from ..templates.smem_tiling import TilingDesign, Workload, cost_model
from ..templates.smem_tiling.remote import run_remote
from ..templates.smem_tiling.smem_template import K_TILE_MENU, PREFETCH_MENU, STAGE_MENU, TILE_MENU, search_face_violations
from .config import load
from .warp import WarpModel
from .memory import Hierarchy
from .run import SimConfig
from .tiling import simulate_design

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = [Workload("square_4096", 4096, 4096, 4096), Workload("skinny_64x4096x4096", 64, 4096, 4096), Workload("ragged_1000x1000x1000", 1000, 1000, 1000)]
# the simulator's wait reasons vs the profiler's stall reasons, by family: the dominant family of warp-time in the simulation
# (copy waits, barrier waits, issue-bound compute) must be the dominant family of the profiler's stalled-warps-per-issue-active
STALL_FAMILY = {"compute": {"not_selected", "mio_throttle", "short_scoreboard", "wait", "math_pipe_throttle", "selected"},
                "copy": {"long_scoreboard", "lg_throttle", "membar"}, "barrier": {"barrier"}}
SIM_FAMILY = {"compute": "compute", "compute_queue": "compute", "smem": "compute", "copy": "copy", "barrier": "barrier"}
# the warp model's histogram families map to the profiler's: issued/not_selected -> not_selected family, scoreboard -> short_scoreboard, lsu -> mio_throttle
WARP_TO_NCU = {"issued": "not_selected", "not_selected": "not_selected", "scoreboard": "short_scoreboard", "lsu": "mio_throttle", "done": None}


def dominant_families(sim_waits: dict, stalls: dict) -> tuple[str | None, str | None, dict, dict]:
    sim_f = {}
    for k, v in (sim_waits or {}).items():
        f = SIM_FAMILY.get(k)
        if f: sim_f[f] = sim_f.get(f, 0.0) + float(v)
    ncu_f = {}
    for k, v in (stalls or {}).items():
        for f, members in STALL_FAMILY.items():
            if k in members: ncu_f[f] = ncu_f.get(f, 0.0) + float(v)
    return (max(sim_f, key=sim_f.get) if sim_f else None), (max(ncu_f, key=ncu_f.get) if ncu_f else None), sim_f, ncu_f


def instr_per_phase_for(row: dict, w, chip, host: str | None) -> float | None:
    """Per-warp instructions per phase from the compiled kernel's SASS (templates/smem_tiling/sass.py), cached by source hash."""
    from ..templates.smem_tiling.sass import sass_counts
    d = row["design"]; v = {"Tm": d["tm"], "Tn": d["tn"], "Tk": d["tk"], "S": d["stages"], "P": d["prefetch"], "beta": int(d["trailing_barrier"])}
    T = tiling_contract(w, hw=chip, precondition="exact"); T.setdefault("id", "tmpl_validation"); T.setdefault("template", "smem_tiling")
    src = kernel_artifact(T, {"id": "val", "values": v})["source"]
    c = sass_counts(src, v["Tm"], v["Tn"], v["Tk"], host)
    row["sass_counts"] = c.get("counts")
    return (c.get("counts") or {}).get("per_warp_per_phase")


def simulate_row(row: dict, sc, C, S, F, chip=None, host: str | None = None) -> dict:
    """(Re)simulate one measured row with the current model; fills the sim_* fields and the family agreement."""
    w = next(x for x in WORKLOADS if x.name == row["workload"]); d = TilingDesign(**{k: v for k, v in row["design"].items() if k in TilingDesign.__dataclass_fields__})
    ipp = instr_per_phase_for(row, w, chip, host) if chip is not None else None
    sim = simulate_design(d, w, sc, Hierarchy(C, S, F), None, issue=WarpModel(C, S), instr_per_phase=ipp, chip=chip, sass_key=row.get("sass_key"))
    sim_dom, ncu_dom, sim_f, ncu_f = dominant_families(sim["waits_s"], (row.get("ncu") or {}).get("stalls_per_issue_active"))
    wf = sim.get("wait_families_s") or {}
    row.update({"simulated_us": sim["time_us"], "sim_input": sim.get("input"), "sim_error": (sim["time_us"] / row["measured_us"] - 1) if row.get("measured_us") else None,
                "sim_critical": max(sim["critical_counts"], key=sim["critical_counts"].get) if sim["critical_counts"] else None, "sim_critical_counts": sim["critical_counts"],
                "sim_dram_bytes": sim["dram_bytes"], "sim_waves": sim["waves"], "sim_wait_families_warp_s": sim_f, "ncu_stall_families": ncu_f,
                "sim_dominant_family": sim_dom, "ncu_dominant_family": ncu_dom, "critical_agrees_with_stalls": (sim_dom == ncu_dom) if (sim_dom and ncu_dom) else None,
                "sim_warp_histogram_s": wf, "sim_dominant_warp_stall": (WARP_TO_NCU.get(max((k for k in wf if k != "done" and k != "issued"), key=wf.get)) if any(k not in ("done", "issued") for k in wf) else None)})
    return row


def summarize(rows: list) -> dict:
    """B27_plus.md §3.2: per-family agreement. For each family the profiler reports as dominant, the fraction of those designs where
    the simulator's dominant family agrees; the gate needs >= 80 % overall and at least half of the non-compute designs."""
    errs = sorted(abs(r["sim_error"]) for r in rows if r["sim_error"] is not None)
    agree = [r["critical_agrees_with_stalls"] for r in rows if r["critical_agrees_with_stalls"] is not None]
    per_family = {}
    for r in rows:
        f = r.get("ncu_dominant_family")
        if f and r.get("critical_agrees_with_stalls") is not None:
            d = per_family.setdefault(f, {"designs": 0, "agree": 0}); d["designs"] += 1; d["agree"] += int(r["critical_agrees_with_stalls"])
    for f, d in per_family.items():
        d["fraction"] = round(d["agree"] / d["designs"], 3) if d["designs"] else None
    non_compute = [r for r in rows if r.get("ncu_dominant_family") and r["ncu_dominant_family"] != "compute" and r.get("critical_agrees_with_stalls") is not None]
    overall = (sum(1 for x in agree if x) / len(agree)) if agree else None
    nc_frac = (sum(1 for r in non_compute if r["critical_agrees_with_stalls"]) / len(non_compute)) if non_compute else None
    return {"designs": len(rows), "ran": sum(1 for r in rows if r["ran"]), "reference_agrees": sum(1 for r in rows if r["reference_agrees"]),
            "median_abs_error": errs[len(errs) // 2] if errs else None, "worst_abs_error": errs[-1] if errs else None,
            "within_10": sum(1 for e in errs if e <= 0.10), "within_20": sum(1 for e in errs if e <= 0.20),
            "signed_median_error": sorted(r["sim_error"] for r in rows if r["sim_error"] is not None)[len(errs) // 2] if errs else None,
            "critical_agreement": f"{sum(1 for x in agree if x)}/{len(agree)}", "agreement_overall": overall, "agreement_by_ncu_family": per_family,
            "agreement_non_compute": nc_frac, "non_compute_designs": len(non_compute),
            "gate": {"median_abs_error": 0.10, "worst_abs_error": 0.20, "agreement_overall_min": 0.80, "agreement_non_compute_min": 0.5},
            "gate_passed": bool(errs and errs[len(errs) // 2] <= 0.10 and errs[-1] <= 0.20 and overall is not None and overall >= 0.8 and (nc_frac is None or nc_frac >= 0.5))}


BUDGET_BLOCK_PHASES = 1_200_000     # the Python simulator runs about 2 us per event, 40 events per block-phase: ~100 s at this budget


def feasible(w: Workload, hw, exclude: set, budget: int = BUDGET_BLOCK_PHASES) -> tuple[list[TilingDesign], int]:
    """Feasible designs of the menu under the exact precondition, minus the excluded ones and those whose blocks x phases
    exceed the simulation budget (the tiny-tile corner of the menu: thousands of blocks times hundreds of phases)."""
    out, over = [], 0
    for m, n, k, s, p, b in product(TILE_MENU, TILE_MENU, K_TILE_MENU, STAGE_MENU, PREFETCH_MENU, (False, True)):
        d = TilingDesign(m, n, k, s, p, b)
        if search_face_violations(d, hw, "exact") or (m, n, k, s, p, b) in exclude:
            continue
        blocks = -(-w.m // m) * -(-w.n // n); phases = -(-w.k // k)
        if blocks * phases > budget:
            over += 1; continue
        out.append(d)
    return out, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True); ap.add_argument("--chip", default="spark-6951-gb10"); ap.add_argument("--config", default="spark-6951-gb10")
    ap.add_argument("--per-workload", type=int, default=10); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=None); ap.add_argument("--no-ncu", action="store_true")
    ap.add_argument("--resimulate", default=None, help="re-run only the simulation on the measured rows of this validation.json (model changes need no new measurements)")
    a = ap.parse_args()
    chip = scope_chip(load_chip(a.chip)); hw = hardware_contract(chip)
    cfg = load(a.config); C, S, F = cfg["contract"], cfg["structural"], cfg["fixed"]; sc = SimConfig(C, S, F, a.config)
    if a.resimulate:
        v = json.loads(Path(a.resimulate).read_text())
        for i, row in enumerate(v["rows"]):
            simulate_row(row, sc, C, S, F, chip, a.host)
            print(f"[{i + 1:2d}] {row['workload']:<22} {row['design']['tm']}x{row['design']['tn']}x{row['design']['tk']} S{row['design']['stages']} P{row['design']['prefetch']}  measured {row['measured_us']} us  sim {row['simulated_us']:.1f} us ({(row['sim_error'] or 0) * 100:+.1f} %)  {row['sim_dominant_family']} / {row['ncu_dominant_family']} {'agree' if row['critical_agrees_with_stalls'] else 'DISAGREE'}", flush=True)
        v["summary"] = summarize(v["rows"]); v["resimulated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        Path(a.out or a.resimulate).write_text(json.dumps(v, indent=1) + "\n"); print(json.dumps(v["summary"], indent=1)); return
    contract_path = ROOT / "calibration" / "spark-6951" / "contract.yaml"
    # the spark-001 designs are excluded from the sample
    exclude = set()
    db = ROOT / "runs" / "spark-001.db"
    if db.exists():
        from ..ledger import Ledger
        L = Ledger(str(db))
        for m in L.artifacts("measurement"):
            v = L.get(m["assignment"])["values"]; exclude.add((v["Tm"], v["Tn"], v["Tk"], v["S"], v["P"], bool(v["beta"])))
    rng = random.Random(a.seed)
    rows = []; budget_excluded = {}
    for w in WORKLOADS:
        pool, over = feasible(w, hw, exclude); budget_excluded[w.name] = {"feasible_in_budget": len(pool), "over_budget": over}
        print(f"{w.name}: {len(pool)} feasible designs within the simulation budget, {over} over it", flush=True)
        pick = rng.sample(pool, min(a.per_workload, len(pool)))
        T = tiling_contract(w, hw=chip, precondition="exact"); T.setdefault("id", f"tmpl_validation_{w.name}"); T.setdefault("template", "smem_tiling")
        for i, d in enumerate(pick):
            x = {"id": f"val_{w.name}_{i}", "values": {"Tm": d.tm, "Tn": d.tn, "Tk": d.tk, "S": d.stages, "P": d.prefetch, "beta": int(d.trailing_barrier)}, "objective": cost_model(decode(T, {"Tm": d.tm, "Tn": d.tn, "Tk": d.tk, "S": d.stages, "P": d.prefetch, "beta": int(d.trailing_barrier)}), w, hw)["time_s"] * 1e6}
            dd = decode(T, x["values"])
            k = kernel_artifact(T, x)
            t0 = time.time()
            r = run_remote(a.host, k["source"], w.m, w.n, w.k, reps=a.reps, arch=k["arch"] or "sm_121", ncu=not a.no_ncu, contract=str(contract_path))
            prof = r.get("ncu") or {}
            row = {"workload": w.name, "design": dd.to_dict(), "assignment": x["id"], "predicted_us": x["objective"],
                   "ran": bool(r.get("ran")), "reference_agrees": r.get("reference_agrees"), "measured_us": r.get("median_us"), "ci95_us": r.get("ci95_us"),
                   "sm_mhz_in_run": r.get("sm_clock_mhz_in_run"), "policy": r.get("policy"), "error": r.get("error"),
                   "ncu": {kk: vv for kk, vv in prof.items() if kk != "raw"}, "kernel_sha256": k["sha256"], "seconds": round(time.time() - t0, 1)}
            simulate_row(row, sc, C, S, F, chip, a.host)
            rows.append(row)
            print(f"[{len(rows):2d}] {w.name:<22} {x['values']}  measured {row['measured_us']} us  sim {row['simulated_us']:.1f} us  "
                  f"({(row['sim_error'] or 0) * 100:+.1f} %)  {row['sim_dominant_family']} / {row['ncu_dominant_family']} {'agree' if row['critical_agrees_with_stalls'] else 'DISAGREE'}", flush=True)
    summary = summarize(rows)
    out = {"host": a.host, "chip": a.chip, "config": a.config, "seed": a.seed, "per_workload": a.per_workload, "reps": a.reps, "excluded_spark001_designs": len(exclude),
           "simulation_budget_block_phases": BUDGET_BLOCK_PHASES, "menu": budget_excluded,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summary": summary, "rows": rows}
    path = Path(a.out) if a.out else ROOT / "calibration" / "spark-6951" / "validation.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1)); print("wrote", path)


if __name__ == "__main__":
    main()
