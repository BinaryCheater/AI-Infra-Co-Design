"""The tiling template's program (paper Listing 1) compiled into the shared event vocabulary.

  for each block:
    for u in 0..P-1:        issue copy of k-tile u -> slot u mod S
    for p in 0..KT-1:
      if p+P < KT:          issue copy of k-tile p+P -> slot (p+P) mod S
      wait for k-tile p's copy group
      leading barrier
      compute k-tile p from slot p mod S
      if beta:              trailing barrier

`warp_programs` builds it for any warp count and phase count (the verifier's two-warp event model or
a real block's eight warps). `verifier_projection` is what the untimed reader checks: the copy is
placed at its issue point (the earliest a write can land, which is the worst case for lifetime), the
wait is implied by the leading barrier, and each phase's barriers are shared events. This replaces
the former `build_program` in smem_verify.py and must agree with it event for event.
"""
from __future__ import annotations

from ...sim.events import BarrierArrive, Compute, IssueCopy, Program, Store, Wait
from .smem_template import TilingDesign


def warp_programs(design: TilingDesign, warps: int, phases: int, tile_bytes: int = 0, flops_per_phase: float = 0.0,
                  store_bytes: int = 0) -> Program:
    S, P = design.stages, design.prefetch
    progs = []
    for w in range(warps):
        seq = []
        for u in range(min(P, phases)):
            seq.append(IssueCopy(w, ("tile", u), u % S, tile_bytes, group=u, tile=u))
        for p in range(phases):
            if P == 0:
                seq.append(IssueCopy(w, ("tile", p), p % S, tile_bytes, group=p, tile=p))
            elif p + P < phases:
                seq.append(IssueCopy(w, ("tile", p + P), (p + P) % S, tile_bytes, group=p + P, tile=p + P))
            seq.append(Wait(w, group=p))
            if design.leading_barrier:
                seq.append(BarrierArrive(w, f"A{p}", p, "leading"))
            seq.append(Compute(w, (p % S,), flops_per_phase, "fp32", tile=p, phase=p))
            if design.trailing_barrier:
                seq.append(BarrierArrive(w, f"B{p}", p, "trailing"))
        if store_bytes:
            seq.append(Store(w, 0, store_bytes))          # the warp's share of the C tile, written once at the end
        progs.append(seq)
    return Program(progs, S, phases, {"P": P, "beta": int(bool(design.trailing_barrier)), "leading": design.leading_barrier})


def verifier_projection(program: Program) -> list[list[tuple]]:
    """The untimed reader's view: ("LOAD", w, tile) at the issue point, ("BAR", "A"/"B", phase) shared,
    ("COMP", w, tile); waits are implied by the leading barrier that follows them."""
    out = []
    for w, seq in enumerate(program.warps):
        prog = []
        for e in seq:
            if isinstance(e, IssueCopy):
                prog.append(("LOAD", w, e.tile))
            elif isinstance(e, BarrierArrive):
                prog.append(("BAR", "A" if e.kind == "leading" else "B", e.phase))
            elif isinstance(e, Compute):
                prog.append(("COMP", w, e.tile))
        out.append(prog)
    return out
