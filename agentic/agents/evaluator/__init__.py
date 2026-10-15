"""Evaluator service: measure(decoded design), probe(T) and the optimism record.

Two backends. `roofline` is the template's analytic cost model re-evaluated for the decoded design:
regime detection is real but "measured" equals "predicted" by construction, and no number carries a
coverage row because none was checked against the chip. `simulator` (G5) is the mechanistic GB10
simulator (`agentic/sim`): the event-driven block/phase reader with the memory hierarchy and the
warp model, every parameter a measurement in the chip's contract. Every number it returns is labeled
with the coverage rows it rests on (`calibration/<host>/coverage.md`: the mechanism, the benchmarks that
measured it, the error of the model as a prediction on the composite that depends on it), so a reader
sees which measured mechanism a time or a regime comes from and how well that mechanism predicted.

`probe(T, backend)` is the Formulator's cost_model.probe tool and the Orchestrator's regime bound. With
the roofline it sweeps the whole feasible menu; one dominant resource everywhere proves the regime as
a bound. With the simulator a sweep costs seconds per design, so it simulates a sample (the roofline's
best designs, and the best of each roofline regime) and can only observe the regime, never prove it:
`regime_holds_as_bound` is False and the note says how many designs were simulated.
"""
from __future__ import annotations

import re
from functools import lru_cache
from itertools import product
from pathlib import Path

from ...templates.smem_tiling import (K_TILE_MENU, PREFETCH_MENU, STAGE_MENU, TILE_MENU, HardwareContract,
                                      TilingDesign, Workload, cost_model, search_face_violations)
from ...chips import hardware_contract
from ..verifier import decode

RESOURCE_REGIME = {"compute": "compute-bound", "DRAM": "memory-bound", "copy": "copy-bound", "barrier": "barrier-bound",
                   "smem": "load-bound", "compute_queue": "compute-bound"}
SIMULATOR = "simulator"
ROOT = Path(__file__).resolve().parents[3]
# which coverage rows (by the first words of the mechanism cell) a simulator number rests on
RESTS_ON = {"compute": ["issue", "scoreboard", "LSU queue", "banks"], "copy": ["cp.async completion under load", "L2", "DRAM"],
            "barrier": ["cp.async completion under load", "issue"], "smem": ["banks", "LSU queue"], "compute_queue": ["issue"],
            "DRAM": ["DRAM", "L2"], "L2": ["L2"], "time": ["issue", "LSU queue", "cp.async completion under load", "L2", "DRAM"]}


def regime_of(dominant_resource: str) -> str:
    return RESOURCE_REGIME.get(dominant_resource, dominant_resource)


# ------------------------------------------------------------------------------------------- the simulator backend
@lru_cache(maxsize=None)
def coverage_rows(chip_name: str) -> dict:
    """The coverage table beside the chip's contract: mechanism -> {benchmarks, error}; empty when there is none."""
    for f in (ROOT / "calibration").glob("*/contract.yaml"):
        try:
            if re.search(r"^name:\s*" + re.escape(chip_name) + r"\s*$", f.read_text(), re.M) and (f.parent / "coverage.md").exists():
                rows = {}
                for line in (f.parent / "coverage.md").read_text().splitlines():
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    if len(cells) >= 3 and cells[0] not in ("mechanism", "") and not cells[0].startswith("---"):
                        rows[cells[0]] = {"benchmarks": cells[1], "error": cells[2]}
                return rows
        except OSError:
            pass
    return {}


def rests_on(chip_name: str, family: str) -> list[dict]:
    rows = coverage_rows(chip_name)
    out = []
    for key in RESTS_ON.get(family, []):
        hit = next((name for name in rows if name.startswith(key)), None)
        out.append({"row": hit or key, **(rows.get(hit) or {"benchmarks": None, "error": "no coverage row"})})
    return out


@lru_cache(maxsize=None)
def _sim_context(chip_name: str):
    from ...sim.config import load
    from ...sim.run import SimConfig
    cfg = load(chip_name)
    C, S, F = cfg["contract"], cfg["structural"], cfg["fixed"]
    return C, S, F, SimConfig(C, S, F, chip_name)


def simulate(d: TilingDesign, w: Workload, chip: dict) -> dict:
    """One design through the full simulator; the result with its dominant family and the coverage labels."""
    from ...sim.memory import Hierarchy
    from ...sim.tiling import simulate_design
    from ...sim.warp import WarpModel
    name = chip["name"]
    C, S, F, sc = _sim_context(name)
    r = simulate_design(d, w, sc, Hierarchy(C, S, F), issue=WarpModel(C, S), chip=chip)
    t = r["time_us"] * 1e-6
    counts = dict(r.get("critical_counts") or {})
    fam = max(counts, key=counts.get) if counts else "compute"
    hz = sc.clock_hz; sms = int(C["sms"])
    dram_s = r["dram_bytes"] / float(C["dram_bytes_per_s"]) if t else 0.0
    util = {"compute": min(1.0, r.get("compute_busy_s", 0.0) / (sms * t)) if t else 0.0, "DRAM": min(1.0, dram_s / t) if t else 0.0}
    for k, v in counts.items():                                     # the share of block-phases each family set the pace of
        util[k if k != "compute" else "compute_phases"] = v / max(1, sum(counts.values()))
    dom = "DRAM" if util["DRAM"] >= 0.8 and util["DRAM"] >= util["compute"] else fam
    labels = {"time_us": rests_on(name, "time"), "dominant_resource": rests_on(name, dom), "dram_bytes": rests_on(name, "DRAM"),
              **{f"utilizations.{k}": rests_on(name, "DRAM" if k == "DRAM" else ("compute" if k.startswith("compute") else k)) for k in util}}
    return {"time_us": r["time_us"], "utilizations": util, "dominant_resource": dom, "critical_counts": counts, "dram_bytes": r["dram_bytes"],
            "l2_bytes": r.get("l2_bytes"), "waves": r.get("waves"), "blocks": r.get("blocks"), "coverage": labels,
            "input": r.get("input"),
            "simulator": f"mechanistic simulator ({name}: contract + warp model ({r.get('input')}) + memory hierarchy; every number labeled by its coverage row)"}


def _evaluate(d: TilingDesign, w: Workload, hw: HardwareContract) -> dict:
    c = cost_model(d, w, hw)
    t = c["time_s"]
    util = {"compute": c["compute_s"] / t if t else 0.0, "DRAM": c["memory_s"] / t if t else 0.0}
    dom = max(util, key=util.get)
    return {"time_us": t * 1e6, "utilizations": util, "dominant_resource": dom, "detail": c}


def _evaluate(d: TilingDesign, w: Workload, hw: HardwareContract) -> dict:
    c = cost_model(d, w, hw)
    t = c["time_s"]
    util = {"compute": c["compute_s"] / t if t else 0.0, "DRAM": c["memory_s"] / t if t else 0.0}
    dom = max(util, key=util.get)
    return {"time_us": t * 1e6, "utilizations": util, "dominant_resource": dom, "detail": c}


def measure(T: dict, x: dict, backend: str = "roofline") -> dict:
    if backend not in ("roofline", SIMULATOR):
        raise ValueError(f"no evaluator backend {backend!r} (roofline, simulator, or cuda:<alias> through cuda_evaluator)")
    w = Workload(**T["scope"]["workload"])
    d = decode(T, x["values"])
    if backend == SIMULATOR:
        if T["scope"]["chip"].get("kind") != "measured":
            raise ValueError("the simulator backend needs a measured chip (a calibration contract), not the illustrative one")
        e = simulate(d, w, T["scope"]["chip"])
        measured, predicted = e["time_us"], x["objective"]
        return {"contract": T["id"], "assignment": x["id"], "design": d.to_dict(),
                "predicted_cost": predicted, "measured_cost": measured,
                "discrepancy": (measured - predicted) / predicted if predicted else 0.0,
                "utilizations": e["utilizations"], "dominant_resource": e["dominant_resource"],
                "simulator": e["simulator"], "coverage": e["coverage"],
                "seed": None, "detail": {k: e[k] for k in ("critical_counts", "dram_bytes", "l2_bytes", "waves", "blocks")},
                "provenance": {"service": "agentic.agents.evaluator", "backend": backend, "chip": T["scope"]["chip"]["name"]}}
    hw = hardware_contract(T["scope"]["chip"])
    e = _evaluate(d, w, hw)
    measured, predicted = e["time_us"], x["objective"]
    return {"contract": T["id"], "assignment": x["id"], "design": d.to_dict(),
            "predicted_cost": predicted, "measured_cost": measured,
            "discrepancy": (measured - predicted) / predicted if predicted else 0.0,
            "utilizations": e["utilizations"], "dominant_resource": e["dominant_resource"],
            "simulator": "analytic-roofline (template cost model re-evaluated; not a measurement)", "coverage": None,
            "seed": None, "detail": {k: v for k, v in e["detail"].items() if k != "time_s"},
            "provenance": {"service": "agentic.agents.evaluator", "backend": backend}}


def probe(T: dict, backend: str = "roofline", sample: int = 12, budget_block_phases: int = 150_000) -> dict:
    """Regime over the branch. Roofline: the dominant resource on every feasible design of the menu (a proof when uniform).
    Simulator: the roofline sweep ranks the menu, then the simulator runs the `sample` fastest designs plus the fastest of
    every roofline regime, each within `budget_block_phases` (blocks x k-phases; about 10 s per 150 000); the regime is the
    simulator's majority over that sample and is observed, never proved. Every simulated number carries its coverage rows."""
    w = Workload(**T["scope"]["workload"])
    hw = hardware_contract(T["scope"]["chip"])
    mode = next(p["mode"] for p in T["preconditions"] if p["id"] == "lifetime")
    s = T["scope"]["structure"]
    counts: dict[str, int] = {}
    best = None
    n = 0
    ranked = []
    for m, nn, k, S, P, b in product(TILE_MENU, TILE_MENU, K_TILE_MENU, STAGE_MENU, PREFETCH_MENU, (False, True)):
        d = TilingDesign(m, nn, k, S, P, b, leading_barrier=s["leading_barrier"], grid_rounding=s["grid_rounding"],
                         guard=s["guard"], stride_offset=s["stride_offset"])
        if search_face_violations(d, hw, mode):
            continue
        n += 1
        e = _evaluate(d, w, hw)
        counts[e["dominant_resource"]] = counts.get(e["dominant_resource"], 0) + 1
        ranked.append((e["time_us"], e["dominant_resource"], d))
        if best is None or e["time_us"] < best["time_us"]:
            best = {"time_us": e["time_us"], "dominant_resource": e["dominant_resource"],
                    "utilizations": e["utilizations"], "design": d.to_dict()}
    uniform = len(counts) == 1
    dom = max(counts, key=counts.get) if counts else "none"
    if backend != SIMULATOR:
        return {"feasible_designs": n, "dominant_counts": counts, "uniform": uniform,
                "regime": regime_of(dom), "regime_holds_as_bound": uniform, "best_predicted": best, "backend": backend, "coverage": None,
                "note": "bound proved by exhaustive evaluation of the feasible menu" if uniform else
                        "regime varies across the feasible menu: it can only be observed, not proved"}
    # the simulator on a sample: the roofline's fastest designs and the fastest of each roofline regime, within the budget
    ranked.sort(key=lambda r: r[0])
    chosen, seen = [], set()
    for t_us, res, d in ranked:
        blocks = -(-w.m // d.tm) * -(-w.n // d.tn); phases = -(-w.k // d.tk)
        if blocks * phases > budget_block_phases:
            continue
        key = (d.tm, d.tn, d.tk, d.stages, d.prefetch, d.trailing_barrier)
        if key in seen:
            continue
        covered = {rr for _, rr, _ in chosen}
        if len(chosen) < sample or res not in covered:
            chosen.append((t_us, res, d)); seen.add(key); covered.add(res)
        if len(chosen) >= sample and all(rr in covered for rr in counts):
            break
    sim_counts: dict[str, int] = {}; sim_best = None; rows = []
    chip = T["scope"]["chip"]
    for t_us, res, d in chosen:
        e = simulate(d, w, chip)
        sim_counts[e["dominant_resource"]] = sim_counts.get(e["dominant_resource"], 0) + 1
        rows.append({"design": d.to_dict(), "roofline_us": t_us, "roofline_resource": res, "simulated_us": e["time_us"],
                     "dominant_resource": e["dominant_resource"], "critical_counts": e["critical_counts"]})
        if sim_best is None or e["time_us"] < sim_best["time_us"]:
            sim_best = {"time_us": e["time_us"], "dominant_resource": e["dominant_resource"], "utilizations": e["utilizations"],
                        "design": d.to_dict(), "coverage": e["coverage"]}
    sdom = max(sim_counts, key=sim_counts.get) if sim_counts else "none"
    name = chip["name"]
    return {"feasible_designs": n, "dominant_counts": counts, "uniform": uniform, "backend": backend,
            "simulated_designs": len(rows), "simulated_dominant_counts": sim_counts, "simulated": rows,
            "regime": regime_of(sdom), "regime_holds_as_bound": False, "best_predicted": sim_best, "roofline_best": best,
            "coverage": {"regime": rests_on(name, sdom), "best_predicted.time_us": rests_on(name, "time")},
            "note": f"simulator on {len(rows)} of {n} feasible designs (the roofline's fastest and the fastest of each roofline regime, "
                    f"within {budget_block_phases} block-phases each): the regime is observed on the sample, not proved over the menu; "
                    f"roofline counts over the menu: {counts}"}


class OptimismRecord:
    """Whether every measured design has run no faster than its prediction (one-sided cost model).

    While clean, ceilings from the model may prune. The first measurement that beats its prediction
    flips the record to dirty; every decision that depended on it is listed so it can be demoted.
    """

    def __init__(self, tol: float = 1e-9):
        self.clean = True
        self.tol = tol
        self.dependents: list[str] = []
        self.history: list[dict] = []

    def update(self, predicted: float, measured: float, artifact: str | None = None) -> bool:
        beat = measured < predicted * (1 - self.tol)
        self.history.append({"predicted": predicted, "measured": measured, "beat_prediction": beat,
                             "artifact": artifact})
        if beat:
            self.clean = False
        return self.clean

    def depend(self, decision_artifact: str):
        self.dependents.append(decision_artifact)

    def to_dict(self) -> dict:
        return {"clean": self.clean, "dependents": list(self.dependents), "samples": len(self.history)}
