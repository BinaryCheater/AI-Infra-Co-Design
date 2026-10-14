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


def schedule(ledger: Ledger, contract_id: str, run: str | None = None) -> dict:
    from .story import short_names
    names = short_names(ledger, run) if run else {}
    v = template_view(ledger, contract_id, run)
    T = v["contract"]
    tpl = importlib.import_module(f"agentic.templates.{T['template']}")
    be = _backend(T["template"])
    x = _dedupe(v["assignments"])[-1] if v["assignments"] else None
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
           "resources": None, "proved": None, "counterexample": None, "lanes": []}

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
        out["note"] = "no design was chosen (the template was rejected before the solver ran)"
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
    out["axis"] = {
        "unit": "s", "phases": KT, "shown": shown, "block_time": ev["block_time"], "waves": ev["waves"], "modeled": True,
        "caption": (f"Modeled, not measured: compute bar = block FLOPs / KT / per-SM rate = {dur['compute'] * 1e6:.3f} µs; "
                    f"load bar = tile bytes / (DRAM bandwidth / {ev['blocks_per_wave']} blocks per wave) = {dur['load'] * 1e6:.3f} µs; "
                    f"barrier tick = {dur['barrier'] * 1e9:.1f} ns. One block of {ev['waves']} wave{'s' if ev['waves'] != 1 else ''}."),
        "durations": dur}
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
        "baseline_s": tpl.baseline_cost(w, hw)["time_s"]}
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
    out["lanes"] = (["memory"] + [f"slot {i}" for i in range(S)] + ["sync", "compute"])
    return out
