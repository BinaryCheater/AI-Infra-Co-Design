"""Paper Listing 1, the shared-memory tiling template, as a contract.

`tiling_contract` builds the contract for one workload and one precondition mode. The JSON fixture
`contract.json` next to this file is the exact-precondition instance for the 1000^3 ragged GEMM and
is the S0 fixture: it must validate, and the stores must be rebuildable from a ledger that holds it.
"""
from __future__ import annotations

from ...templates.smem_tiling import (K_TILE_MENU, PRECONDITIONS, PREFETCH_MENU, STAGE_MENU, TILE_MENU,
                                      HardwareContract, Workload)

LEVEL = 4            # tensor execution mapping: operator level -> tensor-mapping level
TEMPLATE = "smem_tiling"
DEFAULT_STRUCTURE = {"leading_barrier": True, "grid_rounding": "ceil", "guard": "bounded", "stride_offset": 0}

PROGRAM = """for each block (i, j):
  for u in 0 .. P-1:   load k-tile u -> slot u mod S
  for p in 0 .. KT-1:                     # KT = ceil(K/Tk)
    if p+P < KT:       load k-tile p+P -> slot (p+P) mod S
    barrier                               # leading
    compute C[i,j] += slot (p mod S)
    if beta: barrier                      # trailing"""

OBLIGATIONS = [
    {"name": "coverage", "kind": "index", "scope": "any extent",
     "meaning": "every output index of each dimension lies in some admitted tile"},
    {"name": "exact_once", "kind": "index", "scope": "any extent",
     "meaning": "no two admitted (tile, offset) pairs denote the same in-range index"},
    {"name": "no_phantom", "kind": "index", "scope": "any extent",
     "meaning": "no admitted (tile, offset) consumes or writes an index >= extent"},
    {"name": "readiness", "kind": "schedule", "scope": "all interleavings of the event model",
     "meaning": "a warp reads a slot only after every write of the tile it consumes"},
    {"name": "lifetime", "kind": "schedule", "scope": "all interleavings of the event model",
     "meaning": "no slot element is overwritten between its defining write and its last read"},
    {"name": "agreement", "kind": "replay", "scope": "one admissible interleaving",
     "meaning": "a replayed execution yields exactly the reference tokens A[i,k]*B[k,j]"},
]


def tiling_contract(workload: Workload, hw: HardwareContract | None = None, precondition: str = "exact",
                    structure: dict | None = None, rationale: str = "", parent: str | None = None,
                    structural: dict | None = None, formulator: str = "scripted", **provenance) -> dict:
    hw = hw or HardwareContract()
    structural = {**DEFAULT_STRUCTURE, **(structural or {})}
    pre_desc, _ = PRECONDITIONS[precondition]
    c3 = {"exact": "P <= S - 2 + beta", "conservative": "P <= S - 1 and beta = 1",
          "unsound": "P <= S - 1"}[precondition]
    structure = structure or {
        "name": "bottleneck_dominance", "bucket": "objective",
        "regime": "compute-bound: util(compute) >= util(DRAM)",
        "move": "epigraph cost T >= max(T_comp, T_mem); search the tile menu with prefetch overlap",
        "obligation": "the dominant resource stays dominant across the feasible region of the menu",
        "strength": "observed", "evidence": []}
    return {
        "parent": parent, "level": LEVEL, "object": "shared-memory slot", "template": TEMPLATE,
        "scope": {"workload": workload.to_dict(), "chip": hw.to_dict(), "structure": structural},
        "signature": f"{TEMPLATE}:{precondition}:{structural['grid_rounding']}:{structural['guard']}:"
                     f"{structural['stride_offset']}:{int(structural['leading_barrier'])}",
        "decisions": {
            "Tm": {"domain": list(TILE_MENU), "kind": "software", "meaning": "output tile rows"},
            "Tn": {"domain": list(TILE_MENU), "kind": "software", "meaning": "output tile columns"},
            "Tk": {"domain": list(K_TILE_MENU), "kind": "software", "meaning": "reduction tile"},
            "S": {"domain": list(STAGE_MENU), "kind": "software", "meaning": "slots per block"},
            "P": {"domain": list(PREFETCH_MENU), "kind": "software", "meaning": "prefetch distance"},
            "beta": {"domain": [0, 1], "kind": "software", "meaning": "trailing barrier"}},
        "constraints": [
            {"id": "C1", "expr": "S * e * (Tm*Tk + Tk*Tn) <= M_s", "meaning": "shared-memory capacity"},
            {"id": "C2", "expr": "256 <= Tm*Tn <= R_acc", "meaning": "accumulator budget"},
            {"id": "C3", "expr": c3, "meaning": "== Pre.lifetime (K1)"}],
        "cost": {"model": "T >= T_comp, T >= T_mem, T >= T_comp + T_mem if P = 0", "objective": "time_us",
                 "terms": {"T_mem": "DRAM traffic / bandwidth with A re-read per column block and B per row block",
                           "T_comp": "waves * (2*Tm*Tn*K / FLOPs_per_SM + barriers * t_barrier)"}},
        "decomposition": None,
        "preconditions": [{"id": "lifetime", "expr": c3, "mode": precondition, "meaning": pre_desc}],
        "program": PROGRAM,
        "backward_map": "forget the grid, the slots and the schedule => C = A B over (M, N, K)",
        "certificate_schema": {
            "relation": "slot (u mod S) holds A[:,u], B[u,:] from the completion of its load until the last "
                        "read of k-tile u",
            "rank": "number of k-tiles not yet consumed",
            "obligations": OBLIGATIONS},
        "structure": structure,
        "rationale": rationale,
        "status": "proposed",
        "certificate": None,
        "provenance": {"formulator": formulator, "model": None, "prompt_hash": None, **provenance},
    }


FIXTURE_WORKLOAD = Workload("ragged_1000x1000x1000", 1000, 1000, 1000)


def fixture() -> dict:
    return tiling_contract(FIXTURE_WORKLOAD, rationale=(
        "Listing 1 of the paper. Shared-memory staging with S rotating slots and prefetch distance P; "
        "the lifetime precondition P <= S - 2 + beta is the weakest one on the checked family (24/24)."))
