"""The event vocabulary shared by the verifier (untimed) and the timed simulator (§3.1).

A template's program compiles into one event sequence per warp. Events name the resources they
touch (copy groups, slots, barriers, compute units) and nothing about time; the timed reader adds
time, the untimed reader adds non-determinism.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    warp: int | None            # None for events that belong to the block (barrier release, copy completion)


@dataclass(frozen=True)
class IssueCopy(Event):
    """cp.async / TMA issue: a region of `bytes` from `src` into shared-memory slot `dst_slot`, in `group`."""
    src: tuple                  # ("A", tile) / ("B", tile) or an address range
    dst_slot: int
    bytes: int
    group: int                  # copy group id; completion and waits refer to it
    tile: int | None = None     # the k-tile this copy carries (for the tiling template)


@dataclass(frozen=True)
class CopyDone(Event):
    group: int


@dataclass(frozen=True)
class Wait(Event):
    group: int                  # cp.async.wait_group: block until this group (and all earlier ones) completed


@dataclass(frozen=True)
class BarrierArrive(Event):
    barrier: str                # e.g. "A3" (leading barrier of phase 3) or "B3" (trailing)
    phase: int
    kind: str                   # "leading" | "trailing"


@dataclass(frozen=True)
class BarrierRelease(Event):
    barrier: str


@dataclass(frozen=True)
class Load(Event):
    addr: int
    bytes: int


@dataclass(frozen=True)
class Store(Event):
    addr: int
    bytes: int


@dataclass(frozen=True)
class Compute(Event):
    slot_reads: tuple           # slots this step reads, e.g. (slot,)
    flops: float
    unit: str = "fp32"
    tile: int | None = None     # the k-tile consumed (for the tiling template)
    phase: int | None = None
    smem_wavefronts: int = 0    # warp-level ld.shared instructions this step issues (each a wavefront when conflict-free)
    smem_stride: int = 1        # word stride of those accesses across the warp (bank-conflict degree, M3)
    k_steps: int = 0            # G4 issue model (M7): inner-loop k-steps in this phase; 0 = use the rate model
    ffma_per_step: int = 0      # per warp per k-step: FFMA, ld.shared.32 and ld.shared.128 warp-instructions
    lds32_per_step: int = 0
    lds128_per_step: int = 0
    other_per_phase: int = 0    # per warp per phase: copy-issue, guard, address and loop instructions outside the k-loop (from SASS)
    a_wavefronts: tuple = ()    # wavefronts of each A load (128-bit, one per RM row per four k-steps) from its addresses
    b_wavefronts: tuple = ()    # wavefronts of each B load (32-bit, one per RN column per k-step)


@dataclass
class Program:
    """Per-warp event sequences plus the block-level facts both readers need."""
    warps: list[list[Event]]
    slots: int
    phases: int
    meta: dict = field(default_factory=dict)

    def flat(self):
        for w, seq in enumerate(self.warps):
            for i, e in enumerate(seq):
                yield w, i, e
