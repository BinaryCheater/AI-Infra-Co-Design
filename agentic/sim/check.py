"""The consistency condition (§3.4): the obligations the verifier proves over all admissible
interleavings must hold on the timed run's trace. Checks readiness (a compute of tile u starts after
u's copy completed) and lifetime (no copy of another tile lands in u's slot between u's completion and
the end of its last read) on a simulator trace; a violation is a bug in one of the two readers."""
from __future__ import annotations

from collections import defaultdict

from .events import IssueCopy, Program


def check_trace(trace: list, programs: list[Program]) -> dict:
    """trace: (t, sm, block, warp, kind, info). Returns {readiness: [...], lifetime: [...]} violations."""
    group_meta = {}                       # block -> group -> (slot, tile)
    for bi, prog in enumerate(programs):
        for seq in prog.warps:
            for e in seq:
                if isinstance(e, IssueCopy):
                    group_meta.setdefault(bi, {})[e.group] = (e.dst_slot, e.tile)
    done = defaultdict(dict)              # block -> group -> t
    computes = defaultdict(list)          # block -> [(t_start, t_end, warp, tile, slot)]
    starts = {}
    for t, sm, b, w, kind, info in sorted(trace, key=lambda r: r[0]):
        if kind == "copy_done":
            done[b][info["group"]] = t
        elif kind == "compute" and w is not None:
            # the trace records the end; the start is the previous advance of this warp
            key = (b, w)
            t0 = starts.get(key, None)
            computes[b].append((t0, t, w, info["tile"], info["slot"]))
        if w is not None and kind in ("wait", "barrier", "compute", "issue_copy"):
            starts[(b, w)] = t
    readiness, lifetime = [], []
    for b, cs in computes.items():
        for t0, t1, w, tile, slot in cs:
            td = done[b].get(tile)
            if td is None or (t0 is not None and td > t0):
                readiness.append({"block": b, "warp": w, "tile": tile, "copy_done": td, "compute_start": t0})
            # any other tile's copy into this slot landing in (td, t1] clobbers the read
            for g, tg in done[b].items():
                s2, tile2 = group_meta[b][g]
                if s2 == slot and tile2 != tile and td is not None and td < tg <= t1:
                    lifetime.append({"block": b, "warp": w, "tile": tile, "slot": slot, "clobbered_by": tile2, "at": tg, "read_end": t1})
    return {"readiness": readiness, "lifetime": lifetime}
