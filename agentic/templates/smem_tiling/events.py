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


BANKS, BANK_BYTES = 32, 4
GROUPS_BY_WIDTH = {4: 1, 8: 2, 16: 4}           # a 32-bit access is served per full warp, 64-bit per half, 128-bit per quarter (B26.6)


def wavefronts(addresses: list[int], width: int) -> int:
    """Shared-memory wavefronts of one warp-level load from its per-lane byte addresses (B30_plus round 3, check 2).

    distinct addresses move bytes in 128 B passes; a wide access with several distinct addresses also needs one pass per
    distinct address up to its group count; a bank serving two different words serializes (degree). A broadcast is one."""
    D = sorted(set(addresses))
    if not D:
        return 0
    by_bank = {}
    for a in D:
        for w in range(a // BANK_BYTES, (a + width) // BANK_BYTES):
            by_bank.setdefault(w % BANKS, set()).add(w)
    degree = max(len(v) for v in by_bank.values())
    return max(-(-len(D) * width // 128), min(len(D), GROUPS_BY_WIDTH.get(width, 1)), degree)


def inner_loop_wavefronts(design: TilingDesign, warp: int = 0) -> dict:
    """Per k-step, per warp: the loads of the generated inner loop with their wavefronts. Thread (tx, ty) = (t % 16, t / 16);
    warp w holds ty = 2w and 2w + 1. A: As[(ty + i*16)*TK + kk] read as 128-bit over four k-steps (one per i per 4 steps);
    B: Bs[kk*TN + tx + j*16], 32-bit, one per j per step."""
    tm, tn, tk = design.tm, design.tn, design.tk
    rm, rn = max(1, tm // 16), max(1, tn // 16)
    lanes = [(l % 16, 2 * warp + l // 16) for l in range(32)]
    a_loads = []                                  # per 4 k-steps: one 128-bit load per i, at k0 = 0 (addresses shift by 16 B per 4 steps: same banks)
    for i in range(rm):
        addrs = [((ty + i * 16) * tk + 0) * 4 for tx, ty in lanes]
        a_loads.append(wavefronts(addrs, 16))
    b_loads = []                                  # per k-step: one 32-bit load per j
    for j in range(rn):
        addrs = [(0 * tn + tx + j * 16) * 4 for tx, ty in lanes]
        b_loads.append(wavefronts(addrs, 4))
    per_step_lds128 = rm / 4.0
    return {"a_wavefronts_per_load": a_loads, "b_wavefronts_per_load": b_loads,
            "lds128_per_step": per_step_lds128, "lds32_per_step": rn,
            "wavefronts_per_step": sum(a_loads) / 4.0 + sum(b_loads), "instructions_per_step": per_step_lds128 + rn}


def warp_programs(design: TilingDesign, warps: int, phases: int, tile_bytes: int = 0, flops_per_phase: float = 0.0,
                  store_bytes: int = 0, instr_per_phase: float | None = None) -> Program:
    S, P = design.stages, design.prefetch
    # the generated kernel (kernel.cu.j2): a 16 x 16 thread grid, RM x RN outputs per thread; per k-step each thread
    # loads RM values of A (broadcast across the warp's 16 columns: one wavefront each) and RN values of B
    # (16 consecutive words: one wavefront each), so (RM + RN) * Tk ld.shared wavefronts per warp per phase
    rm, rn = max(1, design.tm // 16), max(1, design.tn // 16)
    smem_wf = (rm + rn) * design.tk if tile_bytes else 0
    # G4 issue model: per warp per k-step RM x RN FFMA, RN ld.shared.32 (the B row) and ceil(RM / 4) ld.shared.128
    # (the A column, four k-steps per instruction under the compiler's unroll); B21's SASS mix confirms the counts
    ffma, lds32, lds128 = rm * rn, rn, -(-rm // 4)
    wf = inner_loop_wavefronts(design)
    # outside the k-loop: the copy issue, guards, addresses, wait and barriers. Read from the compiled kernel's SASS
    # (templates/smem_tiling/sass.py) when the caller passes instr_per_phase; no estimate otherwise (0)
    loop_instr = design.tk * (ffma + wf["instructions_per_step"] + 1)
    copies = int(round(tile_bytes * warps / (warps * 32 * 16))) if tile_bytes else 0      # chunks per thread per phase (tile_bytes is one warp's share)
    other = int(round(max(0.0, (instr_per_phase or 0.0) - loop_instr - copies))) if instr_per_phase else 0
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
            seq.append(Compute(w, (p % S,), flops_per_phase, "fp32", tile=p, phase=p, smem_wavefronts=smem_wf, smem_stride=1,
                               k_steps=design.tk if tile_bytes else 0, ffma_per_step=ffma, lds32_per_step=lds32, lds128_per_step=lds128, other_per_phase=other,
                               a_wavefronts=tuple(wf["a_wavefronts_per_load"]), b_wavefronts=tuple(wf["b_wavefronts_per_load"]),
                               copies_per_phase=copies if (P == 0 or p + P < phases) else 0))
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
