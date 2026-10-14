"""Event programs for the calibration microbenchmarks (§4): each benchmark's own kernel as events, so
the simulator must reproduce the measured number from the mechanism it has."""
from __future__ import annotations

from .events import BarrierArrive, Compute, Program


def fma_kernel(sms: int, blocks_per_sm: int = 8, threads: int = 256, iters: int = 200000, ilp: int = 8) -> list[Program]:
    """B2: every warp runs iters x ilp dependent FMAs (2 flops each) with no memory traffic."""
    warps = threads // 32
    flops = 2.0 * ilp * iters * 32
    progs = []
    for _ in range(sms * blocks_per_sm):
        progs.append(Program([[Compute(w, (), flops, "fp32", phase=0)] for w in range(warps)], 0, 1))
    return progs


def barrier_kernel(sms: int, threads: int, iters: int) -> list[Program]:
    """B7: one block per SM, a loop of __syncthreads with a trivial dependent update."""
    warps = threads // 32
    progs = []
    for _ in range(sms):
        seqs = [[BarrierArrive(w, f"A{i}", i, "leading") for i in range(iters)] for w in range(warps)]
        progs.append(Program(seqs, 0, iters))
    return progs


def compute_only_kernel(sms: int, threads: int, phases: int, flops_per_phase_per_warp: float, trailing: bool = False) -> list[Program]:
    """A tiling-shaped kernel with the copies removed: leading barrier + compute per phase."""
    warps = threads // 32
    progs = []
    for _ in range(sms):
        seqs = []
        for w in range(warps):
            seq = []
            for p in range(phases):
                seq.append(BarrierArrive(w, f"A{p}", p, "leading"))
                seq.append(Compute(w, (), flops_per_phase_per_warp, "fp32", tile=p, phase=p))
                if trailing:
                    seq.append(BarrierArrive(w, f"B{p}", p, "trailing"))
            seqs.append(seq)
        progs.append(Program(seqs, 0, phases))
    return progs
