"""Program -> schedule events for the shared-memory tiling template (console round 4, §2–§3).

The template's program fixes the order of loads, barriers and computes; the design fixes Tm, Tn, Tk,
S, P and beta; the cost model fixes the modeled durations. The pipeline below lays one thread block's
k-phases out in program order with those durations, so it is structurally exact (ordering, overlap,
slot lifetimes, barriers) and only its durations are modeled:

  compute bar  = block_flops / KT / fp32_flops_per_s_per_sm
  load bar     = tile_bytes / (dram_bytes_per_s / blocks_per_wave)
  barrier tick = barrier_s

A load into slot u mod S may start only after the last read of the tile that slot held (the lifetime
precondition, which the verifier certifies), and a compute may start only after its tile's load has
completed (readiness). Both therefore hold in the picture by construction.
"""
from __future__ import annotations

from .smem_template import HardwareContract, TilingDesign, Workload, ceil_div, cost_model


def block_events(d: TilingDesign, w: Workload, hw: HardwareContract) -> dict:
    KT = ceil_div(w.k, d.tk)
    S, P, beta = d.stages, d.prefetch, 1 if d.trailing_barrier else 0
    gm = ceil_div(w.m, d.tm) if d.grid_rounding == "ceil" else w.m // d.tm
    gn = ceil_div(w.n, d.tn) if d.grid_rounding == "ceil" else w.n // d.tn
    blocks = gm * gn
    waves = ceil_div(blocks, hw.sms) if blocks else 0
    blocks_per_wave = min(hw.sms, blocks) if blocks else 1
    block_flops = 2 * d.tm * d.tn * w.k
    dc = block_flops / KT / hw.fp32_flops_per_s_per_sm
    tile_bytes = hw.bytes_per_element * (d.tm * d.tk + d.tk * d.tn)
    dl = tile_bytes / (hw.dram_bytes_per_s / blocks_per_wave)
    bt = hw.barrier_s

    loads, computes, barriers, slots = [], [], [], []
    load_end = {}           # tile -> completion time
    slot_free = [0.0] * S   # slot -> time its previous tile was last read
    mem_free = 0.0          # one DMA path per block: loads are sequential
    t = 0.0

    def issue(u, phase_issued, now):
        nonlocal mem_free
        slot = u % S
        t0 = max(now, mem_free, slot_free[slot])
        t1 = t0 + dl
        loads.append({"tile": u, "slot": slot, "issue_phase": phase_issued, "t0": t0, "t1": t1, "bytes": tile_bytes})
        load_end[u] = t1
        mem_free = t1

    for u in range(min(P, KT)):
        issue(u, -1, 0.0)
    for p in range(KT):
        if P == 0:
            issue(p, p, t)
        elif p + P < KT:
            issue(p + P, p, t)
        tb = max(t, load_end[p])          # leading barrier: every warp's loads of tile p are complete
        barriers.append({"phase": p, "kind": "leading", "t": tb})
        c0 = tb + bt
        c1 = c0 + dc
        computes.append({"tile": p, "slot": p % S, "t0": c0, "t1": c1, "flops": block_flops / KT})
        t = c1
        if beta:
            barriers.append({"phase": p, "kind": "trailing", "t": t})
            t += bt
        slot_free[p % S] = t              # the slot may be refilled once this tile's reads are done
        slots.append({"slot": p % S, "tile": p, "t_ready": load_end[p], "t_last_read": c1,
                      "t_free": t})
    block_time = t
    cm = cost_model(d, w, hw)
    return {
        "phases": KT, "S": S, "P": P, "beta": beta, "waves": waves, "blocks": blocks, "blocks_per_wave": blocks_per_wave,
        "durations": {"compute": dc, "load": dl, "barrier": bt, "tile_bytes": tile_bytes, "block_flops": block_flops},
        "loads": loads, "computes": computes, "barriers": barriers, "slots": slots, "block_time": block_time,
        "cost_model": cm,
        "grid": {"gm": gm, "gn": gn, "tm": d.tm, "tn": d.tn, "m": w.m, "n": w.n, "k": w.k, "tk": d.tk,
                 "waves": waves, "sms": hw.sms, "rounding": d.grid_rounding},
    }


def proved_windows(ev: dict) -> dict:
    """The windows the schedule obligations protect, from the events: readiness per compute, lifetime per
    slot occupancy (last read of a tile -> next load into its slot)."""
    next_load_into_slot = {}
    for ld in sorted(ev["loads"], key=lambda x: x["t0"]):
        next_load_into_slot.setdefault((ld["slot"], ld["tile"]), ld["t0"])
    by_tile = {ld["tile"]: ld for ld in ev["loads"]}
    readiness = [{"tile": c["tile"], "slot": c["slot"], "t0": by_tile[c["tile"]]["t1"], "t1": c["t0"]} for c in ev["computes"]]
    lifetime = []
    for s in ev["slots"]:
        nxt = next_load_into_slot.get((s["slot"], s["tile"] + ev["S"]))
        lifetime.append({"tile": s["tile"], "slot": s["slot"], "t0": s["t_last_read"], "t1": nxt if nxt is not None else s["t_free"],
                         "next_tile": s["tile"] + ev["S"] if nxt is not None else None})
    return {"readiness": readiness, "lifetime": lifetime}


def counterexample_events(cex: dict) -> dict | None:
    """The verifier's two-warp interleaving as per-warp event rows plus the violating pair."""
    if not cex or not cex.get("replay") or not cex["replay"].get("trace"):
        return None
    import re
    trace = cex["replay"]["trace"]
    design = cex.get("design", {})
    S = design.get("stages", 1)
    rows: dict[int, list] = {}
    steps = []
    for i, line in enumerate(trace):
        m = re.match(r"LOAD warp(\d+) k-tile(\d+) -> slot(\d+)", line)
        if m:
            e = {"step": i, "kind": "load", "warp": int(m[1]), "tile": int(m[2]), "slot": int(m[3])}
        else:
            m = re.match(r"COMP warp(\d+) k-tile(\d+) reads slot(\d+)", line)
            if m:
                e = {"step": i, "kind": "compute", "warp": int(m[1]), "tile": int(m[2]), "slot": int(m[3]),
                     "stale": "STALE" in line}
            else:
                m = re.match(r"BARRIER (\w)(\d+)", line)
                e = {"step": i, "kind": "barrier", "warp": None, "name": (m[1] + m[2]) if m else line, "phase": int(m[2]) if m else None}
        steps.append(e)
        if e["warp"] is not None:
            rows.setdefault(e["warp"], []).append(e)
    violation = None
    stale = next((e for e in steps if e["kind"] == "compute" and e.get("stale")), None)
    if stale:
        # the write that clobbered the read: the latest load into the same slot, of a different tile, before the read
        writes = [e for e in steps if e["kind"] == "load" and e["slot"] == stale["slot"] and e["tile"] != stale["tile"]
                  and e["step"] < stale["step"]]
        clobber = writes[-1] if writes else None
        violation = {"obligation": cex.get("obligation", "lifetime").split("[")[0], "slot": stale["slot"], "tile": stale["tile"],
                     "pair": [clobber, stale] if clobber else [stale]}
    ob = cex.get("obligation", "").split("[")[0]
    return {"warps": [rows[w] for w in sorted(rows)], "steps": steps, "event_model": {"warps": len(rows), "phases": 5, "S": S},
            "violation": violation or {"obligation": ob, "slot": None, "tile": None, "pair": []}}
