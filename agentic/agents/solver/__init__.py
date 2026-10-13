"""Solver service: solve(contract) -> assignment. Deterministic; reads only the search face.

v1 dispatches on `contract.template`. The only backend is the smem tiling MILP (HiGHS via SciPy),
the 93-variable program of paper §6.2. The solver never sees the correctness face.
"""
from __future__ import annotations

import math

from ...templates.smem_tiling import HardwareContract, Workload, solve_milp

BACKENDS = {}


def backend(name):
    def reg(fn):
        BACKENDS[name] = fn
        return fn
    return reg


@backend("smem_tiling")
def _solve_smem_tiling(T: dict) -> dict:
    w = Workload(**T["scope"]["workload"])
    hw = HardwareContract(**T["scope"]["chip"])
    mode = next(p["mode"] for p in T["preconditions"] if p["id"] == "lifetime")
    sol = solve_milp(w, hw, precondition=mode)
    d = sol["design"]
    bound = sol["dual_bound_us"]
    if bound is None or (isinstance(bound, float) and math.isnan(bound)):
        bound = sol["objective_us"]
    gap = sol["gap"]
    if gap is None or (isinstance(gap, float) and math.isnan(gap)):
        gap = 0.0
    return {
        "contract": T["id"],
        "values": {"Tm": d.tm, "Tn": d.tn, "Tk": d.tk, "S": d.stages, "P": d.prefetch,
                   "beta": 1 if d.trailing_barrier else 0},
        "objective": float(sol["objective_us"]), "bound": float(bound), "gap": float(gap),
        "solver_time": float(sol["solver_seconds"]), "status": sol["status"],
        "diagnostics": {"nodes": sol["nodes"], "variables": sol["variables"],
                        "binary_variables": sol["binary_variables"], "constraints": sol["constraints"],
                        "backend": "scipy.optimize.milp (HiGHS)", "precondition_mode": mode},
        "provenance": {"service": "agentic.agents.solver", "template": T["template"]},
    }


def solve(T: dict) -> dict:
    try:
        fn = BACKENDS[T["template"]]
    except KeyError:
        raise ValueError(f"no solver backend for template {T['template']!r}")
    return fn(T)
