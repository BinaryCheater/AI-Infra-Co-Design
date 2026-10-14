"""The untimed reader (§3.2): admissible interleavings of a projected program as Z3 constraints.

Program order holds within a warp; a barrier instance is one point every warp crosses; all other
events are distinct points in time. These three functions were the encoding inside smem_verify.py
and now live beside the shared event model; the obligations themselves stay in the template's
verifier, which is where their meaning is.
"""
from __future__ import annotations

import z3


def event_vars(progs: list[list[tuple]]):
    """Per-warp event variables; barriers share one variable per instance."""
    ev, bars = {}, {}
    for prog in progs:
        for e in prog:
            if e[0] == "BAR":
                bars.setdefault(e, z3.Int(f"t_bar_{e[1]}_{e[2]}"))
            else:
                ev.setdefault(e, z3.Int(f"t_{e[0]}_{e[1]}_{e[2]}"))
    return ev, bars


def admissibility(progs: list[list[tuple]], ev: dict, bars: dict):
    """Program order per warp; a barrier instance is one point every warp crosses."""
    c = []
    for prog in progs:
        times = [bars[e] if e[0] == "BAR" else ev[e] for e in prog]
        c += [a < b for a, b in zip(times, times[1:])]
    c.append(z3.Distinct(*ev.values()))
    return z3.And(*c)


def linearize(model: dict, progs: list[list[tuple]]) -> list[list]:
    """One concrete interleaving from a Z3 model: events sorted by their assigned times."""
    items, seen = [], set()
    for prog in progs:
        for e in prog:
            name = f"t_bar_{e[1]}_{e[2]}" if e[0] == "BAR" else f"t_{e[0]}_{e[1]}_{e[2]}"
            if name in seen:
                continue
            seen.add(name)
            items.append((model[name], name, e))
    items.sort()
    return [list(e) for _, _, e in items]
