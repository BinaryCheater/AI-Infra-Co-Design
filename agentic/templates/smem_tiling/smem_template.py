"""Dual-purpose template: GEMM -> shared-memory-tiled GEMM.

One object, two faces:
  * search face   -- the design variables, their menus and the linear constraints the
                     MILP optimizes under (capacity, register budget, lifetime precondition);
  * correctness face -- the refinement obligations the checker must discharge before a
                     design admitted by the search face is accepted.

The refinement introduces one new noun: a shared-memory tile slot.  A block stages
A[Tm x Tk] and B[Tk x Tn] into one of S slots, synchronizes, computes from the slot and
(optionally) synchronizes again before the slot may be refilled.  Loads for k-tile p+P are
issued while k-tile p is being consumed (prefetch distance P).

The backward map is the identity on the GEMM: discard the block grid, the slots and the
schedule and the remaining computation is C = A B over the declared shape.
"""
from dataclasses import asdict, dataclass, replace

# ---------------------------------------------------------------- design ---

TILE_MENU = (16, 32, 64, 128, 256)     # Tm, Tn
K_TILE_MENU = (8, 16, 32, 64)         # Tk
STAGE_MENU = (1, 2, 3)                 # S
PREFETCH_MENU = (0, 1, 2)              # P


@dataclass(frozen=True)
class TilingDesign:
    """One point admitted (or rejected) by the template."""
    tm: int
    tn: int
    tk: int
    stages: int = 1                    # S: number of shared-memory slots
    prefetch: int = 0                  # P: k-tiles loaded ahead of the one consumed
    trailing_barrier: bool = True      # barrier after compute, before the slot is refilled
    leading_barrier: bool = True       # barrier after loads, before compute
    grid_rounding: str = "ceil"        # number of tiles per dimension: ceil or floor
    guard: str = "bounded"             # tail elements guarded by  index < extent
    stride_offset: int = 0             # tile origin stride = T + stride_offset (0 = disjoint tiles)

    def to_dict(self):
        return asdict(self)


def lifetime_precondition(stages, prefetch, trailing_barrier):
    """The template's declared (claimed weakest) precondition for slot reuse.

    With a trailing barrier the slot refilled in phase p held k-tile p+P-S, consumed in
    phase <= p-1, which every warp finished before that barrier: need P <= S-1.
    Without it, only the leading barrier of phase p-1 orders a slow warp's reads, so
    the overwritten tile must be consumed in phase <= p-2: need P <= S-2.
    """
    return prefetch <= stages - 2 + (1 if trailing_barrier else 0)


# --------------------------------------------------------- hardware contract ---

@dataclass(frozen=True)
class HardwareContract:
    """Illustrative cost-model parameters; NOT measured hardware."""
    name: str = "illustrative-fp32-gpu"
    sms: int = 128
    smem_bytes_per_block: int = 96 * 1024
    threads_per_block: int = 256
    max_accumulators_per_thread: int = 128
    dram_bytes_per_s: float = 1.8e12
    fp32_flops_per_s_per_sm: float = 80e12 / 128
    barrier_s: float = 40 / 1.8e9        # ~40 cycles at 1.8 GHz per block barrier
    bytes_per_element: int = 4

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Workload:
    name: str
    m: int
    n: int
    k: int

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------- search face ---

def ceil_div(a, b):
    return -(-a // b)


PRECONDITIONS = {
    # name: (description, predicate(S, P, trailing_barrier))
    "exact": ("P <= S - 2 + trailing_barrier (template claim)", lifetime_precondition),
    "conservative": ("P <= S - 1 and trailing barrier always present",
                     lambda s, p, tb: tb and p <= s - 1),
    "unsound": ("P <= S - 1 regardless of barriers (plausible but wrong)",
                lambda s, p, tb: p <= s - 1),
}


def search_face_violations(d: TilingDesign, hw: HardwareContract, precondition="exact"):
    """Linear (after one-hot encoding) constraints that the MILP enforces."""
    v = []
    if d.tm not in TILE_MENU or d.tn not in TILE_MENU or d.tk not in K_TILE_MENU:
        v.append("tile size outside menu")
    if d.stages not in STAGE_MENU or d.prefetch not in PREFETCH_MENU:
        v.append("stage/prefetch outside menu")
    smem = d.stages * (d.tm * d.tk + d.tk * d.tn) * hw.bytes_per_element
    if smem > hw.smem_bytes_per_block:
        v.append(f"shared-memory capacity {smem} > {hw.smem_bytes_per_block}")
    if d.tm * d.tn > hw.threads_per_block * hw.max_accumulators_per_thread:
        v.append("accumulator register budget exceeded")
    if d.tm * d.tn < hw.threads_per_block:
        v.append("fewer outputs than threads")
    if not PRECONDITIONS[precondition][1](d.stages, d.prefetch, d.trailing_barrier):
        v.append(f"lifetime precondition violated: {PRECONDITIONS[precondition][0]}")
    # Structural choices fixed by the template (the search never relaxes them):
    if (d.grid_rounding, d.guard, d.stride_offset, d.leading_barrier) != ("ceil", "bounded", 0, True):
        v.append("structural template choice altered")
    return v


def cost_model(d: TilingDesign, w: Workload, hw: HardwareContract):
    """Predicted time in seconds; the MILP objective uses exactly these terms."""
    bpe = hw.bytes_per_element
    gm, gn = ceil_div(w.m, d.tm), ceil_div(w.n, d.tn)
    traffic = bpe * (gm * d.tm * w.k * gn       # A re-read once per column block
                     + w.k * gn * d.tn * gm     # B re-read once per row block
                     + w.m * w.n)               # C written once
    mem = traffic / hw.dram_bytes_per_s
    waves = ceil_div(gm * gn, hw.sms)
    block_flops = 2 * d.tm * d.tn * w.k
    barriers = ceil_div(w.k, d.tk) * (1 + (1 if d.trailing_barrier else 0))
    comp = waves * (block_flops / hw.fp32_flops_per_s_per_sm + barriers * hw.barrier_s)
    total = max(comp, mem) if d.prefetch >= 1 else comp + mem
    return {"time_s": total, "compute_s": comp, "memory_s": mem,
            "dram_bytes": traffic, "waves": waves, "blocks": gm * gn,
            "smem_bytes": d.stages * (d.tm * d.tk + d.tk * d.tn) * bpe}


def baseline_cost(w: Workload, hw: HardwareContract):
    """Coarse level: every thread streams its A row and B column from DRAM into registers."""
    traffic = hw.bytes_per_element * (2 * w.m * w.n * w.k + w.m * w.n)
    mem = traffic / hw.dram_bytes_per_s
    comp = 2 * w.m * w.n * w.k / (hw.fp32_flops_per_s_per_sm * hw.sms)
    return {"time_s": max(mem, comp), "compute_s": comp, "memory_s": mem, "dram_bytes": traffic}


# --------------------------------------------------------- correctness face ---

CORRECTNESS_FACE = (
    ("coverage", "every output index of each dimension lies in some admitted tile"),
    ("exact_once", "no two admitted (tile, offset) pairs denote the same in-range index"),
    ("no_phantom", "no admitted (tile, offset) consumes or writes an index >= extent"),
    ("readiness", "a warp reads a slot only after every write of the tile it consumes"),
    ("lifetime", "no slot element is overwritten between its defining write and its last read"),
    ("terminal_agreement", "a replayed execution yields exactly the reference GEMM tokens"),
)


BROKEN_VARIANTS = {
    "floor_grid_uncovered_tail": dict(grid_rounding="floor"),
    "unguarded_tail_phantom_terms": dict(guard="unguarded"),
    "overlapping_tiles_duplicate_terms": dict(stride_offset=-1),
    "single_buffer_prefetch_overwrite": dict(stages=1, prefetch=1),
    "double_buffer_no_trailing_barrier_prefetch1": dict(stages=2, prefetch=1, trailing_barrier=False),
    "missing_leading_barrier": dict(leading_barrier=False),
}


def broken(design: TilingDesign, name: str) -> TilingDesign:
    return replace(design, **BROKEN_VARIANTS[name])
