"""The schedule view's data (notes/schedule_view_design.md §3): a pure function over the ledger.

What the solver chose and the verifier certified, laid out in time for one thread block: loads,
slot lifetimes, barriers and computes; the output grid and its waves; the resource strip; the
windows the obligations protect; and, for a refuted template, the verifier's interleaving drawn in
the same lanes. The per-template part (program -> events) is `templates/<name>/schedule.py`.
Nothing here is written to the ledger.
"""
from __future__ import annotations

import importlib

from ..chips import hardware_contract
from .describe import _dedupe, describe, display_name, phrasing
from .ledger import Ledger
from .timeline import template_view
from .views import rebuild

ACCEPTED = ("proved", "agrees")


def _backend(template: str):
    return importlib.import_module(f"agentic.templates.{template}.schedule")


def _waves(n: int) -> str:
    return "one wave" if n == 1 else f"{n} waves (rounds of up to one block per SM slot)"


def schedule(ledger: Ledger, contract_id: str, run: str | None = None) -> dict:
    from .story import short_names
    names = short_names(ledger, run) if run else {}
    v = template_view(ledger, contract_id, run)
    T = v["contract"]
    try:
        tpl = importlib.import_module(f"agentic.templates.{T['template']}")
        be = _backend(T["template"])
    except ModuleNotFoundError:
        # a level without a timeline (the graph level): say what the steps are instead
        return {"id": contract_id, "short": names.get(contract_id, "T?"), "block": None, "counterexample": None,
                "note": ("There is no timeline at this level. A timeline shows how one block of threads runs on the chip, and "
                         "blocks only exist once a template one level down (the tiling) has picked a design. At this level the "
                         "work is a short list of steps: an optional transposed copy of B, the multiplication itself (possibly "
                         "cut into pieces along K), and, if it is cut, a final step that adds the pieces up.")}
    x = _dedupe(v["assignments"])[-1] if v["assignments"] else None
    meas = _dedupe(v["measurements"])[-1] if v["measurements"] else None
    measured = None
    if meas and str(meas.get("simulator", "")).startswith("cuda:"):
        ci = meas.get("ci95") or [None, None]
        half = ((ci[1] - ci[0]) / 2 / meas["measured_cost"]) if ci[0] is not None and meas["measured_cost"] else None
        measured = {"time_us": meas["measured_cost"], "predicted_us": meas["predicted_cost"], "discrepancy": meas.get("discrepancy"),
                    "ci95_us": ci, "half_width_rel": half, "host": meas.get("host") or meas["simulator"][5:],
                    "reps": (meas.get("detail") or {}).get("reps"), "reference_agrees": meas.get("reference_agrees"),
                    "utilizations": meas.get("utilizations"), "kernel": meas.get("kernel"), "clocks": meas.get("clocks")}
    verdicts = _dedupe(v["verdicts"])
    fam = next((w for w in verdicts if w["check"] == "family"), None)
    inst = next((w for w in verdicts if w["check"] == "instance"), None)
    w = tpl.Workload(**T["scope"]["workload"])
    hw = hardware_contract(T["scope"]["chip"])
    s = T["scope"]["structure"]
    ph = phrasing(T["template"])
    out = {"id": contract_id, "short": names.get(contract_id, "T?"), "workload": display_name(w.name),
           "chip": display_name(hw.name, "chip"), "parent": T.get("parent"), "parent_short": names.get(T.get("parent") or ""),
           "objective_us": x["objective"] if x else None, "design": None, "axis": None, "block": None, "grid": None,
           "resources": None, "proved": None, "counterexample": None, "lanes": [], "measured": measured}

    cex = (inst and inst["counterexample"]) or (fam and fam["counterexample"]) or None
    if cex:
        ce = be.counterexample_events(cex)
        ob = cex["obligation"].split("[")[0]
        words = (ph.get("failure_in_words") or {}).get(ob, ob)
        if ce:
            ce["words"] = words
            ce["kind"] = "schedule"
            ce["source"] = "family sweep" if (fam and fam["counterexample"] is cex) or (fam and fam["verdict"] != "proved") else "instance check"
            ce["design"] = cex.get("design")
            out["counterexample"] = ce
        else:
            rp = cex.get("replay") or {}
            out["counterexample"] = {"kind": "index", "warps": [], "steps": [], "words": words,
                                     "violation": {"obligation": ob, "slot": None, "tile": None, "pair": []},
                                     "replay": {"shape": rp.get("shape"), "differences_total": rp.get("differences_total")},
                                     "design": cex.get("design")}

    if x is None:
        out["note"] = "There is no timeline, because no design was chosen: the checker rejected the template before the solver ran."
        return out
    vals = x["values"]
    d = tpl.TilingDesign(vals["Tm"], vals["Tn"], vals["Tk"], vals["S"], vals["P"], bool(vals["beta"]),
                         leading_barrier=s["leading_barrier"], grid_rounding=s["grid_rounding"], guard=s["guard"],
                         stride_offset=s["stride_offset"])
    ev = be.block_events(d, w, hw)
    cm = ev["cost_model"]
    KT, S, P = ev["phases"], ev["S"], ev["P"]
    shown_n = min(KT, max(4, P + S + 2))
    shown = list(range(shown_n)) + ([KT - 1] if KT - 1 >= shown_n else [])
    dur = ev["durations"]
    out["design"] = {"Tm": d.tm, "Tn": d.tn, "Tk": d.tk, "S": S, "P": P, "beta": ev["beta"], "grid_rounding": d.grid_rounding}
    formulas = (f"Each dark bar is one step of arithmetic ({dur['compute'] * 1e6:.3f} µs). Each colored bar on the top line is one "
                f"tile arriving from memory ({dur['load'] * 1e6:.3f} µs, sharing the memory bandwidth with the other blocks running "
                f"at the same time, {ev['blocks_per_wave']} in all). Each red tick is a barrier, where all threads wait ({dur['barrier'] * 1e9:.1f} ns). "
                f"The chip runs the blocks in {_waves(ev['waves'])}.")
    if measured:
        scale = (measured["time_us"] * 1e-6 / ev["waves"]) / ev["block_time"] if ev["block_time"] and ev["waves"] else 1.0
        caption = (f"The shape of this timeline comes from the time formula. The total time was measured on {measured['host']} "
                   f"over {measured['reps']} runs: {measured['time_us']:.1f} µs, against {measured['predicted_us']:.1f} µs predicted "
                   f"({measured['discrepancy']:+.0%}), so the timeline is stretched by ×{scale:.2f} to match. {formulas}")
    else:
        scale = 1.0
        caption = "This timeline comes from the time formula, not from a measurement. " + formulas
    out["axis"] = {
        "unit": "s", "phases": KT, "shown": shown, "block_time": ev["block_time"], "waves": ev["waves"], "modeled": measured is None,
        "measured_scale": scale, "caption": caption, "durations": dur}
    out["block"] = {"loads": ev["loads"], "slots": ev["slots"], "barriers": ev["barriers"], "computes": ev["computes"],
                    "block_time": ev["block_time"]}
    g = ev["grid"]
    partial, uncovered = [], []
    for i in range(g["gm"]):
        rows = min(d.tm, w.m - i * d.tm)
        for j in range(g["gn"]):
            cols = min(d.tn, w.n - j * d.tn)
            if rows < d.tm or cols < d.tn:
                partial.append({"i": i, "j": j, "rows": rows, "cols": cols})
    if g["gm"] * d.tm < w.m:
        uncovered.append({"dim": "M", "from": g["gm"] * d.tm, "to": w.m, "count": w.m - g["gm"] * d.tm})
    if g["gn"] * d.tn < w.n:
        uncovered.append({"dim": "N", "from": g["gn"] * d.tn, "to": w.n, "count": w.n - g["gn"] * d.tn})
    if g["rounding"] == "floor" and (w.k - (w.k // d.tk) * d.tk) and False:
        pass
    out["grid"] = {**g, "partial": partial, "uncovered": uncovered, "blocks": ev["blocks"]}
    t_total = cm["time_s"]
    out["resources"] = {
        "dram": {"util": cm["memory_s"] / t_total if t_total else 0, "bytes": cm["dram_bytes"], "s": cm["memory_s"]},
        "compute": {"util": cm["compute_s"] / t_total if t_total else 0, "s": cm["compute_s"]},
        "smem": {"bytes": cm["smem_bytes"], "cap": hw.smem_bytes_per_block, "util": cm["smem_bytes"] / hw.smem_bytes_per_block},
        "dominant": "compute" if cm["compute_s"] >= cm["memory_s"] else "DRAM", "time_s": t_total,
        "baseline_s": tpl.baseline_cost(w, hw)["time_s"],
        "measured": ({"dram": {"util": measured["utilizations"].get("DRAM")}, "compute": {"util": measured["utilizations"].get("compute")},
                      "dominant": max(measured["utilizations"], key=measured["utilizations"].get), "time_s": measured["time_us"] * 1e-6}
                     if measured and measured.get("utilizations") else None)}
    windows = be.proved_windows(ev)
    status = {}
    if inst:
        for o in inst["obligations"]:
            status[o["obligation"].split("[")[0]] = o["status"]
    def st(name):
        return "proved" if status.get(name, "").startswith(ACCEPTED) else (status.get(name) or "not checked")
    out["proved"] = {"readiness": windows["readiness"], "lifetime": windows["lifetime"],
                     "status": {"readiness": st("readiness"), "lifetime": st("lifetime"),
                                "coverage": st("coverage"), "exact_once": st("exact_once"), "no_phantom": st("no_phantom")},
                     "index": {k: st(k) for k in ("coverage", "exact_once", "no_phantom")},
                     "instance_verdict": inst["verdict"] if inst else None}
    out["lanes"] = (["loads from memory"] + [f"buffer {i}" for i in range(S)] + ["barriers", "arithmetic"])
    return out
