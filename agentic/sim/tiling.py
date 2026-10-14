"""The tiling template on the timed reader: a solved design on a workload and a chip -> block
programs (8 warps, KT phases, real bytes and FLOPs) -> the simulator -> a measurement-shaped result."""
from __future__ import annotations

from ..templates.smem_tiling import TilingDesign, Workload, ceil_div
from ..templates.smem_tiling.events import warp_programs
from .events import Program
from .run import Memory, SimConfig, Simulator


def design_programs(d: TilingDesign, w: Workload, threads: int = 256) -> tuple[list[Program], dict]:
    KT = ceil_div(w.k, d.tk)
    gm = ceil_div(w.m, d.tm) if d.grid_rounding == "ceil" else w.m // d.tm
    gn = ceil_div(w.n, d.tn) if d.grid_rounding == "ceil" else w.n // d.tn
    tile_bytes = 4 * (d.tm * d.tk + d.tk * d.tn)
    warps = threads // 32
    flops_per_phase = 2.0 * d.tm * d.tn * d.tk
    base = warp_programs(d, warps, KT, tile_bytes // warps, flops_per_phase / warps, store_bytes=4 * d.tm * d.tn // warps)   # each warp copies its share
    progs = []
    for i in range(gm):
        for j in range(gn):
            # each block's copies read its own A row-panel tile (i, u) and B column-panel tile (u, j)
            seqs = []
            for wseq in base.warps:
                seq = []
                for e in wseq:
                    if type(e).__name__ == "IssueCopy":
                        seq.append(type(e)(e.warp, ("block", i, j, e.tile), e.dst_slot, e.bytes, e.group, e.tile))
                    else:
                        seq.append(e)
                seqs.append(seq)
            progs.append(Program(seqs, base.slots, base.phases, {**base.meta, "block": (i, j)}))
    smem = d.stages * tile_bytes
    return progs, {"KT": KT, "gm": gm, "gn": gn, "blocks": gm * gn, "tile_bytes": tile_bytes, "smem_bytes": smem,
                   "flops": 2.0 * w.m * w.n * w.k, "threads": threads, "a_tile_bytes": 4 * d.tm * d.tk, "b_tile_bytes": 4 * d.tk * d.tn}


def simulate_design(d: TilingDesign, w: Workload, cfg: SimConfig, memory: Memory | None = None, smem=None,
                    record_trace: bool = False, amplify: tuple | None = None, issue=None) -> dict:
    progs, meta = design_programs(d, w)
    sim = Simulator(cfg, memory, smem, amplify, issue)
    if memory is not None and hasattr(memory, "bind"):
        memory.bind(d, w, meta)
    r = sim.launch(progs, meta["threads"], meta["smem_bytes"], record_trace)
    r.update(meta)
    r["waves"] = -(-meta["blocks"] // int(cfg.contract["sms"]))
    return r
