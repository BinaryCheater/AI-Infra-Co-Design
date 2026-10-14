"""Evaluator service: measure(decoded design) and the optimism record.

v1 has one backend, the analytic roofline model of the template (`roofline`). It re-evaluates the
template's own cost model for the decoded design and reports per-resource utilization, so regime
detection is real but "measured" equals "predicted" by construction. It is labelled as such in every
measurement. A Ventus timing-simulation adapter is the S4 follow-up and plugs in as a second backend.

`probe(T)` is the Formulator's cost_model.probe tool and the Orchestrator's regime bound: it sweeps
the feasible menu of the template and reports which resource dominates where. If one resource
dominates on every feasible design, the regime holds as a bound over the whole branch.
"""
from __future__ import annotations

from itertools import product

from ...templates.smem_tiling import (K_TILE_MENU, PREFETCH_MENU, STAGE_MENU, TILE_MENU, HardwareContract,
                                      TilingDesign, Workload, cost_model, search_face_violations)
from ...chips import hardware_contract
from ..verifier import decode

RESOURCE_REGIME = {"compute": "compute-bound", "DRAM": "memory-bound"}


def regime_of(dominant_resource: str) -> str:
    return RESOURCE_REGIME.get(dominant_resource, dominant_resource)


def _evaluate(d: TilingDesign, w: Workload, hw: HardwareContract) -> dict:
    c = cost_model(d, w, hw)
    t = c["time_s"]
    util = {"compute": c["compute_s"] / t if t else 0.0, "DRAM": c["memory_s"] / t if t else 0.0}
    dom = max(util, key=util.get)
    return {"time_us": t * 1e6, "utilizations": util, "dominant_resource": dom, "detail": c}


def measure(T: dict, x: dict, backend: str = "roofline") -> dict:
    if backend != "roofline":
        raise ValueError(f"no evaluator backend {backend!r} (Ventus adapter is S4 follow-up)")
    w = Workload(**T["scope"]["workload"])
    hw = hardware_contract(T["scope"]["chip"])
    d = decode(T, x["values"])
    e = _evaluate(d, w, hw)
    measured, predicted = e["time_us"], x["objective"]
    return {"contract": T["id"], "assignment": x["id"], "design": d.to_dict(),
            "predicted_cost": predicted, "measured_cost": measured,
            "discrepancy": (measured - predicted) / predicted if predicted else 0.0,
            "utilizations": e["utilizations"], "dominant_resource": e["dominant_resource"],
            "simulator": "analytic-roofline (template cost model re-evaluated; not a measurement)",
            "seed": None, "detail": {k: v for k, v in e["detail"].items() if k != "time_s"},
            "provenance": {"service": "agentic.agents.evaluator", "backend": backend}}


def probe(T: dict) -> dict:
    """Regime over the branch: dominant resource on every feasible design of the menu."""
    w = Workload(**T["scope"]["workload"])
    hw = hardware_contract(T["scope"]["chip"])
    mode = next(p["mode"] for p in T["preconditions"] if p["id"] == "lifetime")
    s = T["scope"]["structure"]
    counts: dict[str, int] = {}
    best = None
    n = 0
    for m, nn, k, S, P, b in product(TILE_MENU, TILE_MENU, K_TILE_MENU, STAGE_MENU, PREFETCH_MENU, (False, True)):
        d = TilingDesign(m, nn, k, S, P, b, leading_barrier=s["leading_barrier"], grid_rounding=s["grid_rounding"],
                         guard=s["guard"], stride_offset=s["stride_offset"])
        if search_face_violations(d, hw, mode):
            continue
        n += 1
        e = _evaluate(d, w, hw)
        counts[e["dominant_resource"]] = counts.get(e["dominant_resource"], 0) + 1
        if best is None or e["time_us"] < best["time_us"]:
            best = {"time_us": e["time_us"], "dominant_resource": e["dominant_resource"],
                    "utilizations": e["utilizations"], "design": d.to_dict()}
    uniform = len(counts) == 1
    dom = max(counts, key=counts.get) if counts else "none"
    return {"feasible_designs": n, "dominant_counts": counts, "uniform": uniform,
            "regime": regime_of(dom), "regime_holds_as_bound": uniform, "best_predicted": best,
            "note": "bound proved by exhaustive evaluation of the feasible menu" if uniform else
                    "regime varies across the feasible menu: it can only be observed, not proved"}


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
