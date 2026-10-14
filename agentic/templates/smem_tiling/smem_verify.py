"""Correctness face of the shared-memory tiling template, checked with Z3.

Two kinds of obligations:

1. Index obligations (coverage, exact_once, no_phantom) for each of the M, N, K
   dimensions.  The tile size is fixed per query, the extent E is an unbounded symbolic
   positive integer, so every UNSAT result covers all extents E >= 1 for that tile size.
   Coverage is checked against the template's proposed witness (the backward map
   bi = i div stride, ii = i - bi*stride); a failed witness is "unresolved" unless the
   independent replay exhibits a missing reference term.

2. Schedule obligations (readiness, lifetime) on an explicit event model: W warps,
   KT k-phases, S slots, prefetch distance P, leading/trailing barriers.  Event times are
   integers constrained only by program order and barrier semantics, so a SAT model is an
   admissible interleaving.  Every SAT model is replayed by an independent interpreter
   that (a) re-checks admissibility and (b) recomputes C as formal tokens and compares
   with the GEMM reference.

The condition builder and the replay interpreter are trusted Python; Z3 is the trusted
backend.  No floating-point values are involved: tokens A[i,k]*B[k,j] are compared
structurally.
"""
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter

import z3

from ...sim.explore import admissibility, event_vars, linearize
from .smem_template import TilingDesign

TIMEOUT_MS = 5000


@dataclass
class Outcome:
    obligation: str
    scope: str
    status: str                 # proved | counterexample | unresolved
    seconds: float
    query_sha256: str
    model: dict = field(default_factory=dict)
    replay: dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__


def solve(formula, decls):
    s = z3.Solver()
    s.set("timeout", TIMEOUT_MS)
    s.set("random_seed", 0)
    s.add(formula)
    text = s.to_smt2()
    t0 = perf_counter()
    r = s.check()
    dt = perf_counter() - t0
    model = {}
    if r == z3.sat:
        m = s.model()
        model = {name: m.eval(v, model_completion=True).as_long() for name, v in decls.items()}
    status = "proved" if r == z3.unsat else ("counterexample" if r == z3.sat else "unresolved")
    return status, dt, sha256(text.encode()).hexdigest(), model


# ======================================================== index obligations ===

def _dim(design: TilingDesign, tile: int, prefix: str):
    """Symbolic view of one dimension of the tiled iteration space."""
    E = z3.Int(f"{prefix}_extent")
    stride = tile + design.stride_offset
    G = (E + tile - 1) / tile if design.grid_rounding == "ceil" else E / tile   # z3 int div
    def admitted(b, o):
        c = [b >= 0, b < G, o >= 0, o < tile]
        if design.guard == "bounded":
            c.append(b * stride + o < E)
        return z3.And(*c)
    def index(b, o):
        return b * stride + o
    return E, stride, admitted, index


def index_obligations(design: TilingDesign):
    out = []
    for dim, tile in (("M", design.tm), ("N", design.tn), ("K", design.tk)):
        if tile + design.stride_offset < 1:
            continue
        E, stride, admitted, index = _dim(design, tile, dim)
        i, b, o, b2, o2 = (z3.Int(f"{dim}_{s}") for s in ("i", "b", "o", "b2", "o2"))
        base = E >= 1
        scope = f"{dim}: tile={tile}, all extents E>=1"
        # coverage, against the proposed witness (backward map)
        wb = i / stride
        wo = i - wb * stride
        cov = z3.And(base, i >= 0, i < E, z3.Not(admitted(wb, wo)))
        once = z3.And(base, admitted(b, o), admitted(b2, o2), z3.Or(b != b2, o != o2),
                      index(b, o) == index(b2, o2), index(b, o) < E)
        phantom = z3.And(base, admitted(b, o), index(b, o) >= E)
        for name, f, decls in (
                ("coverage", cov, {"E": E, "i": i}),
                ("exact_once", once, {"E": E, "b": b, "o": o, "b2": b2, "o2": o2}),
                ("no_phantom", phantom, {"E": E, "b": b, "o": o})):
            status, dt, h, model = solve(f, decls)
            oc = Outcome(f"{name}[{dim}]", scope, status, dt, h, model)
            if status == "counterexample":
                oc.replay = replay_index_counterexample(design, dim, model["E"])
                if not oc.replay["agrees_with_reference"]:
                    oc.status = "counterexample (replayed workload defect)"
                else:
                    oc.status = "unresolved (certificate failed, no defect replayed)"
            out.append(oc)
    return out


def _admitted_indices(design, extent, tile):
    stride = tile + design.stride_offset
    grid = -(-extent // tile) if design.grid_rounding == "ceil" else extent // tile
    for b in range(grid):
        for o in range(tile):
            idx = b * stride + o
            if design.guard == "bounded" and idx >= extent:
                continue
            yield idx


def replay_gemm_indices(design, m, n, k):
    """Independent token replay of the tiled iteration space (no SMT)."""
    rows = list(_admitted_indices(design, m, design.tm))
    cols = list(_admitted_indices(design, n, design.tn))
    reds = list(_admitted_indices(design, k, design.tk))
    actual = {(i, j): Counter() for i in range(m) for j in range(n)}
    phantom = Counter()
    for i in rows:
        for j in cols:
            for kk in reds:
                tok = (i, kk, kk, j)
                if i < m and j < n:
                    actual[i, j][tok] += 1
                else:
                    phantom[tok] += 1
    diffs = []
    for i in range(m):
        for j in range(n):
            exp = Counter({(i, kk, kk, j): 1 for kk in range(k)})
            for tok in sorted(set(exp) | set(actual[i, j])):
                if exp[tok] != actual[i, j][tok]:
                    diffs.append({"output": [i, j], "token": list(tok),
                                  "expected": exp[tok], "actual": actual[i, j][tok],
                                  "reason": "out-of-range reduction term (padding read)" if tok[1] >= k else
                                            ("missing term" if actual[i, j][tok] < exp[tok] else "duplicated term")})
    if phantom:
        diffs.append({"phantom_writes": len(phantom), "example": list(next(iter(phantom))),
                      "reason": "write outside C (unguarded tail)"})
    return {"shape": [m, n, k], "agrees_with_reference": not diffs,
            "differences_total": len(diffs), "differences_sample": diffs[:4]}


def replay_index_counterexample(design, dim, extent):
    extent = max(1, extent)
    # Keep the other dimensions at one full tile so only the failing dimension varies.
    m = extent if dim == "M" else design.tm
    n = extent if dim == "N" else design.tn
    k = extent if dim == "K" else design.tk
    return replay_gemm_indices(design, m, n, k)


# ===================================================== schedule obligations ===

@dataclass(frozen=True)
class EventModel:
    """Small explicit block: Tm=W rows (one per warp), Tn=2, Tk=1, KT k-phases."""
    warps: int = 2
    phases: int = 5
    tn: int = 2

    def elements(self):
        # slot layout: A column (one element per warp row) then B row (tn elements)
        return [("A", r) for r in range(self.warps)] + [("B", c) for c in range(self.tn)]

    def owner(self, e):
        return e % self.warps


def schedule_obligations(design: TilingDesign, em: EventModel = EventModel()):
    from .events import verifier_projection, warp_programs     # the shared event model, projected
    progs = verifier_projection(warp_programs(design, em.warps, em.phases))
    ev, bars = event_vars(progs)
    adm = admissibility(progs, ev, bars)
    S = design.stages
    elems = em.elements()
    reads = []        # (comp_event, element, defining_write, clobbering_writes)
    for w in range(em.warps):
        for p in range(em.phases):
            needed = [i for i, (mat, idx) in enumerate(elems) if mat == "B" or idx == w]
            for e in needed:
                d = ("LOAD", em.owner(e), p)
                clobber = [("LOAD", em.owner(e), u) for u in range(em.phases)
                           if u != p and u % S == p % S]
                reads.append((("COMP", w, p), e, d, clobber))
    ready_bad = z3.Or(*[ev[c] < ev[d] for c, _, d, _ in reads])
    life_bad = z3.Or(*[z3.And(ev[d] < ev[x], ev[x] < ev[c]) for c, _, d, cl in reads for x in cl] or [z3.BoolVal(False)])
    scope = (f"event model W={em.warps}, KT={em.phases}, S={design.stages}, P={design.prefetch}, "
             f"leading={design.leading_barrier}, trailing={design.trailing_barrier}; all interleavings")
    decls = {str(v): v for v in list(ev.values()) + list(bars.values())}
    out = []
    for name, bad in (("readiness", ready_bad), ("lifetime", life_bad)):
        status, dt, h, model = solve(z3.And(adm, bad), decls)
        oc = Outcome(name, scope, status, dt, h)
        if status == "counterexample":
            oc.model = {"interleaving": linearize(model, progs)}
            oc.replay = replay_schedule(design, em, progs, oc.model["interleaving"])
            oc.status = ("counterexample (replayed workload defect)" if not oc.replay["agrees_with_reference"]
                         else "unresolved (no defect replayed)")
        out.append(oc)
    # terminal agreement on one admissible interleaving (witness of nonempty behavior)
    status, dt, h, model = solve(adm, decls)
    if status == "counterexample":           # SAT: an admissible execution exists
        order = linearize(model, progs)
        rep = replay_schedule(design, em, progs, order)
        out.append(Outcome("terminal_agreement(witness run)", scope,
                           "agrees" if rep["agrees_with_reference"] else "disagrees", dt, h,
                           {"events": len(order)}, rep))
    else:
        out.append(Outcome("terminal_agreement(witness run)", scope, "no admissible execution", dt, h))
    return out


def replay_schedule(design, em, progs, order):
    """Independent interpreter: validate admissibility, then execute formal tokens."""
    order = [tuple(e) for e in order]
    pos = {e: t for t, e in enumerate(order)}
    # admissibility, re-derived from the program text (not from the SMT encoding)
    for prog in progs:
        for a, b in zip(prog, prog[1:]):
            if pos[a] >= pos[b]:
                return {"admissible": False, "reason": f"program order {a} !< {b}"}
    elems = em.elements()
    slot = {}                       # (slot, element) -> token ("A", row, k) / ("B", k, col)
    acc = {(w, j): Counter() for w in range(em.warps) for j in range(em.tn)}
    trace = []
    for e in order:
        if e[0] == "LOAD":
            _, w, u = e
            for i, (mat, idx) in enumerate(elems):
                if em.owner(i) == w:
                    slot[u % design.stages, i] = (mat, idx, u) if mat == "A" else (mat, u, idx)
            trace.append(f"LOAD warp{w} k-tile{u} -> slot{u % design.stages}")
        elif e[0] == "COMP":
            _, w, p = e
            s = p % design.stages
            a = slot.get((s, w))
            for j in range(em.tn):
                b = slot.get((s, em.warps + j))
                if a is None or b is None:
                    acc[w, j][("UNINITIALIZED",)] += 1
                    continue
                # A token (A,row,k) and B token (B,k,col): product token (row, kA, kB, col)
                acc[w, j][(a[1], a[2], b[1], b[2])] += 1
            bs = [slot.get((s, em.warps + j)) for j in range(em.tn)]
            stale = [t for t in [a] + bs if t is not None and (t[2] if t[0] == "A" else t[1]) != p]
            trace.append(f"COMP warp{w} k-tile{p} reads slot{s}: A={a} B={bs}"
                         + (f"   <-- STALE/CLOBBERED {stale}" if stale else ""))
        else:
            trace.append(f"BARRIER {e[1]}{e[2]}")
    diffs = []
    for (w, j), got in acc.items():
        exp = Counter({(w, k, k, j): 1 for k in range(em.phases)})
        for tok in sorted(set(exp) | set(got), key=str):
            if exp[tok] != got[tok]:
                diffs.append({"output": [w, j], "token": list(tok), "expected": exp[tok], "actual": got[tok]})
    return {"admissible": True, "agrees_with_reference": not diffs, "differences_total": len(diffs),
            "differences_sample": diffs[:4], "trace": trace}
