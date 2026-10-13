"""Search face: mixed-integer linear program over the tiling template.

Decision variables (all binary unless noted)
  xm[a], xn[b]  one-hot Tm, Tn in TILE_MENU
  xk[c]         one-hot Tk in K_TILE_MENU
  ys[s], yp[p]  one-hot stages S, prefetch distance P
  bb            trailing barrier present
  z_mn, z_mk, z_kn, z_kb   exact products of one-hot pairs (binary McCormick)
  smem, comp, T (continuous, microseconds / bytes)

Constraints
  capacity     S * 4*(Tm*Tk + Tk*Tn) <= smem budget       (big-M per stage choice)
  registers    Tm*Tn within [threads, threads*max_acc]    (pairs fixed to 0)
  lifetime     P <= S - 2 + bb                            (template precondition)
  objective    T >= comp, T >= mem, T >= comp + mem - BIG*[P>=1]   (epigraph of max/sum)

Times are scaled to microseconds so coefficients stay within a few orders of magnitude.
The objective is the template's cost model, NOT a measurement.
"""
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .smem_template import (K_TILE_MENU, PREFETCH_MENU, STAGE_MENU, TILE_MENU, TilingDesign,
                           ceil_div, cost_model, search_face_violations)

US = 1e6


class VarIndex:
    def __init__(self):
        self.names, self.lb, self.ub, self.integer = [], [], [], []

    def add(self, name, lb=0, ub=1, integer=True):
        self.names.append(name); self.lb.append(lb); self.ub.append(ub); self.integer.append(1 if integer else 0)
        return len(self.names) - 1


def solve_milp(w, hw, time_limit=60, precondition="exact"):
    V = VarIndex()
    xm = {m: V.add(f"Tm={m}") for m in TILE_MENU}
    xn = {n: V.add(f"Tn={n}") for n in TILE_MENU}
    xk = {k: V.add(f"Tk={k}") for k in K_TILE_MENU}
    ys = {s: V.add(f"S={s}") for s in STAGE_MENU}
    yp = {p: V.add(f"P={p}") for p in PREFETCH_MENU}
    bb = V.add("trailing_barrier")
    budget_hi = hw.threads_per_block * hw.max_accumulators_per_thread
    zmn = {}
    for m, n in product(TILE_MENU, TILE_MENU):
        ub = 1 if hw.threads_per_block <= m * n <= budget_hi else 0
        zmn[m, n] = V.add(f"z[Tm={m},Tn={n}]", ub=ub)
    zmk = {(m, k): V.add(f"z[Tm={m},Tk={k}]") for m, k in product(TILE_MENU, K_TILE_MENU)}
    zkn = {(k, n): V.add(f"z[Tk={k},Tn={n}]") for k, n in product(K_TILE_MENU, TILE_MENU)}
    zkb = {k: V.add(f"z[Tk={k},bb]") for k in K_TILE_MENU}
    smem = V.add("smem_stage_bytes", 0, np.inf, False)
    comp = V.add("compute_us", 0, np.inf, False)
    T = V.add("time_us", 0, np.inf, False)

    rows, lo, hi = [], [], []
    def row(coefs, l, h):
        r = np.zeros(len(V.names))
        for i, c in coefs:
            r[i] += c
        rows.append(r); lo.append(l); hi.append(h)

    for group in (xm, xn, xk, ys, yp):
        row([(i, 1) for i in group.values()], 1, 1)
    for prod_vars, (a_map, b_map) in ((zmn, (xm, xn)), (zmk, (xm, xk)), (zkn, (xk, xn))):
        for (a, b), z in prod_vars.items():
            row([(z, 1), (a_map[a], -1)], -np.inf, 0)
            row([(z, 1), (b_map[b], -1)], -np.inf, 0)
            row([(z, 1), (a_map[a], -1), (b_map[b], -1)], -1, np.inf)
        row([(z, 1) for z in prod_vars.values()], 1, 1)
    for k, z in zkb.items():
        row([(z, 1), (xk[k], -1)], -np.inf, 0)
        row([(z, 1), (bb, -1)], -np.inf, 0)
        row([(z, 1), (xk[k], -1), (bb, -1)], -1, np.inf)

    bpe = hw.bytes_per_element
    # bytes per stage, linear in the pair products
    row([(smem, 1)] + [(z, -bpe * m * k) for (m, k), z in zmk.items()]
        + [(z, -bpe * k * n) for (k, n), z in zkn.items()], 0, 0)
    big_smem = 3 * bpe * 2 * max(TILE_MENU) * max(K_TILE_MENU)
    for s, y in ys.items():   # s*smem <= cap + BIG*(1-y)
        row([(smem, s), (y, big_smem)], -np.inf, hw.smem_bytes_per_block + big_smem)
    # lifetime precondition (the template's claim is "exact": P - S - bb <= -2)
    if precondition == "exact":
        row([(yp[p], p) for p in PREFETCH_MENU] + [(ys[s], -s) for s in STAGE_MENU] + [(bb, -1)], -np.inf, -2)
    elif precondition == "conservative":
        row([(yp[p], p) for p in PREFETCH_MENU] + [(ys[s], -s) for s in STAGE_MENU], -np.inf, -1)
        row([(bb, 1)], 1, 1)
    elif precondition == "unsound":
        row([(yp[p], p) for p in PREFETCH_MENU] + [(ys[s], -s) for s in STAGE_MENU], -np.inf, -1)
    else:
        raise ValueError(precondition)

    # memory time, linear in z_mn
    def traffic(m, n):
        gm, gn = ceil_div(w.m, m), ceil_div(w.n, n)
        return bpe * (gm * m * w.k * gn + w.k * gn * n * gm + w.m * w.n)
    mem_terms = [(z, traffic(m, n) / hw.dram_bytes_per_s * US) for (m, n), z in zmn.items()]
    # barrier time per block: barrier_s * (K/Tk) * (1 + bb)
    bar_terms = [(xk[k], hw.barrier_s * ceil_div(w.k, k) * US) for k in K_TILE_MENU] + \
                [(zkb[k], hw.barrier_s * ceil_div(w.k, k) * US) for k in K_TILE_MENU]
    worst = 0.0
    for (m, n), z in zmn.items():
        waves = ceil_div(ceil_div(w.m, m) * ceil_div(w.n, n), hw.sms)
        blk = 2 * m * n * w.k / hw.fp32_flops_per_s_per_sm * US
        bar_max = 2 * hw.barrier_s * ceil_div(w.k, min(K_TILE_MENU)) * US
        worst = max(worst, waves * (blk + bar_max))
    big_t = 4 * worst + 4 * max(c for _, c in mem_terms)
    for (m, n), z in zmn.items():
        waves = ceil_div(ceil_div(w.m, m) * ceil_div(w.n, n), hw.sms)
        blk = 2 * m * n * w.k / hw.fp32_flops_per_s_per_sm * US
        # comp >= waves*(blk + bar) - BIG*(1-z)
        row([(comp, 1), (z, -waves * blk - big_t)] + [(i, -waves * c) for i, c in bar_terms], -big_t, np.inf)
    row([(T, 1), (comp, -1)], 0, np.inf)
    row([(T, 1)] + [(i, -c) for i, c in mem_terms], 0, np.inf)
    # no prefetch => no overlap: T >= comp + mem - BIG*(1 - yp[0])
    row([(T, 1), (comp, -1), (yp[0], -big_t)] + [(i, -c) for i, c in mem_terms], -big_t, np.inf)

    c = np.zeros(len(V.names)); c[T] = 1
    t0 = perf_counter()
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.array(V.integer), bounds=Bounds(V.lb, V.ub),
               options={"time_limit": time_limit, "mip_rel_gap": 0})
    dt = perf_counter() - t0
    pick = lambda d: next(k for k, i in d.items() if res.x[i] > 0.5)
    design = TilingDesign(pick(xm), pick(xn), pick(xk), pick(ys), pick(yp), bool(res.x[bb] > 0.5))
    return {"precondition": precondition, "status": res.message, "solver_seconds": dt, "objective_us": float(res.fun),
            "dual_bound_us": float(getattr(res, "mip_dual_bound", np.nan)),
            "gap": float(getattr(res, "mip_gap", np.nan)), "nodes": int(getattr(res, "mip_node_count", -1)),
            "variables": len(V.names), "binary_variables": int(sum(V.integer)), "constraints": len(rows),
            "design": design}


def enumerate_menu(w, hw, precondition="exact"):
    """Independent cross-check: brute force over the same menu and constraints."""
    best, feasible = None, 0
    for m, n, k, s, p, b in product(TILE_MENU, TILE_MENU, K_TILE_MENU, STAGE_MENU, PREFETCH_MENU, (False, True)):
        d = TilingDesign(m, n, k, s, p, b)
        if search_face_violations(d, hw, precondition):
            continue
        feasible += 1
        t = cost_model(d, w, hw)["time_s"]
        if best is None or t < best[0] - 1e-15:
            best = (t, d)
    return {"feasible_designs": feasible, "best_time_us": best[0] * US, "best_design": best[1]}
