"""The describer: three descriptions of one template from the same structured fields, plus one
shared headline (notes/describing_templates.md §4). Pure functions over the ledger, like story.py.

Sources, in priority order: the contract's structured fields (decisions, constraints, cost,
preconditions, certificate schema, backward map); the linked artifacts (assignment, verdicts,
measurement, diagnosis); the per-template phrasing table (templates/<name>/describe.yaml); and the
formulator's `rationale`, quoted but never used as the description. `refines` is not used at all.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from .ledger import Ledger
from .timeline import template_view
from .views import rebuild

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"

# --------------------------------------------------------------------------- display names
WORKLOAD_NAMES = {
    "square_4096": "the 4096³ GEMM",
    "skinny_64x4096x4096": "the skinny 64×4096×4096 GEMM",
    "ragged_1000x1000x1000": "the ragged 1000³ GEMM",
}
CHIP_NAMES = {
    "illustrative-fp32-gpu": "the illustrative FP32 GPU (128 SMs, 96 KiB smem/block, 1.8 TB/s)",
    "spark-6951-gb10": "the GB10 chip in the DGX Spark (host spark-6951)",
}
REGIME_WORDS = {"compute": "compute-bound", "DRAM": "memory-bound"}
CHIP_SHORT = {"illustrative-fp32-gpu": "the illustrative FP32 GPU", "spark-6951-gb10": "spark-6951 (GB10)"}
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def display_name(raw, kind: str = "workload") -> str:
    """Display name for a workload or chip; unknown names fall back to a readable form of the raw
    string, never to an error, so a new workload is never invisible."""
    if raw is None:
        return "?"
    if isinstance(raw, dict):
        raw = raw.get("name", str(raw))
    table = WORKLOAD_NAMES if kind == "workload" else CHIP_NAMES
    if raw in table:
        return table[raw]
    if kind == "workload":
        m = re.fullmatch(r"([a-z]+)_(\d+)x(\d+)x(\d+)", raw)
        if m:
            a, m_, n, k = m.groups()
            dims = f"{m_}³" if m_ == n == k else f"{m_}×{n}×{k}"
            return f"the {a} {dims} GEMM"
        m = re.fullmatch(r"([a-z]+)_(\d+)", raw)
        if m:
            return f"the {m.group(2)}³ GEMM"
    return raw.replace("_", " ")


@lru_cache(maxsize=None)
def phrasing(template: str) -> dict:
    path = TEMPLATES / template / "describe.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()).get(template, {})


def _t(seconds) -> str:
    if seconds is None:
        return "?"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.2f} s"


def _get(obj, path: str):
    """`a.b[0].c` lookups on nested dicts/lists."""
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            cur = cur[int(part[1:-1])]
        else:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def _math(expr: str) -> str:
    return (expr.replace("<=", "≤").replace(">=", "≥").replace("beta", "β").replace("*", "·")
            .replace(" if ", " when "))


def _point(fp: dict) -> str:
    return f"(S = {fp['S']}, P = {fp['P']}, β = {fp['beta']})"


def _decisions(T: dict, ph: dict) -> list[tuple[str, dict]]:
    """Decisions in the template's canonical display order (the ledger stores keys sorted)."""
    order = ph.get("decision_order") or []
    keys = sorted(T["decisions"], key=lambda k: (order.index(k) if k in order else len(order), k))
    return [(k, T["decisions"][k]) for k in keys]


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for x in items:
        if x["id"] not in seen:
            seen.add(x["id"])
            out.append(x)
    return out


# --------------------------------------------------------------------------- the describer
def describe(ledger: Ledger, contract_id: str, run: str | None = None, _ctx: dict | None = None) -> dict:
    from .story import short_names
    ctx = _ctx or {"names": short_names(ledger, run) if run else {}, "views": rebuild(ledger, run)}
    names, views = ctx["names"], ctx["views"]
    v = template_view(ledger, contract_id, run)
    T = v["contract"]
    ph = phrasing(T["template"])
    short = names.get(contract_id, "T?")
    wl, chip = T["scope"]["workload"], T["scope"]["chip"]
    wl_name, chip_name = display_name(wl, "workload"), display_name(chip, "chip")
    x = _dedupe(v["assignments"])[-1] if v["assignments"] else None
    verdicts = _dedupe(v["verdicts"])
    fam = next((w for w in verdicts if w["check"] == "family"), None)
    inst = next((w for w in verdicts if w["check"] == "instance"), None)
    m = _dedupe(v["measurements"])[-1] if v["measurements"] else None
    diag = v["diagnoses"][0] if v["diagnoses"] else None
    mode = next((p["mode"] for p in T["preconditions"] if p["id"] == "lifetime"), None) or (T["preconditions"][0].get("mode") if T["preconditions"] else None)
    pre = next((p for p in T["preconditions"] if p["id"] == "lifetime"), T["preconditions"][0])
    c3 = next((c for c in T["constraints"] if c["id"] == "C3"), None)

    # parent and incumbent
    parent = None
    if T.get("parent"):
        try:
            parent = template_view(ledger, T["parent"], run)
        except KeyError:
            parent = None
    parent_short = names.get(T.get("parent") or "", None)
    key = f"{wl['name']}|{chip.get('name')}|{T['cost'].get('objective', 'time')}"
    inc = views["incumbents"].get(key)
    is_best = bool(inc and inc["contract"] == contract_id)

    # status and failure
    if v["admitted"]:
        status = "accepted"
    elif fam and fam["verdict"] != "proved":
        status = "rejected"
    elif inst and inst["verdict"] != "proved":
        status = "rejected"
    elif fam:
        status = "proved safe"
    else:
        status = "proposed"
    cex = (inst and inst["counterexample"]) or (fam and fam["counterexample"]) or None
    failed_ob = cex["obligation"].split("[")[0] if cex else None
    failure_words = (ph.get("failure_in_words") or {}).get(failed_ob, failed_ob)
    reused_owner = None
    if fam and fam["provenance"].get("reused_from"):
        cert = views["proofs"].get(fam["provenance"]["reused_from"])
        reused_owner = names.get(cert["contract"]) if cert else None

    # ---- headline
    what = (ph.get("what_it_decides") or "refines {workload}").format(workload=wl_name)
    if status == "accepted" and x:
        if is_best:
            outcome = f"best so far, {x['objective']:.1f} µs"
        elif inc and abs(x["objective"] / inc["cost"] - 1) < 5e-4:
            outcome = f"{x['objective']:.1f} µs, same as the best so far"
        elif inc:
            outcome = f"{x['objective']:.1f} µs, {(x['objective'] / inc['cost'] - 1) * 100:.1f} % slower than the best"
        else:
            outcome = f"{x['objective']:.1f} µs"
    elif status == "rejected" and fam and fam["verdict"] != "proved":
        outcome = "rejected before any design was chosen"
    elif status == "rejected":
        outcome = f"design rejected: {failure_words}"
    elif x:
        outcome = f"{x['objective']:.1f} µs predicted"
    else:
        outcome = "not checked yet"
    chip_short = CHIP_SHORT.get(chip.get("name"), chip.get("name", "?"))
    headline = f"{short} · {what[0].upper() + what[1:]} on {chip_short} · {status} · {outcome}"

    # ---- delta from parent
    delta = _delta(T, parent["contract"] if parent else None, parent_short, ph, mode, wl_name)

    # ---- systems lens
    systems = _systems(T, x, m, parent, parent_short, ph, mode, status, fam, inst, failure_words, what, chip_name, inc)
    optimization = _optimization(T, x, parent_short, ph, mode, status, fam, inst, c3, pre, failure_words, v)
    proof = _proof(T, fam, inst, cex, ph, mode, pre, reused_owner, failure_words, diag)
    # the plain-language sections every tab shows (ledger/plain.py); the formal statements stay in each section's details
    from . import plain
    px = _dedupe(parent["assignments"])[-1] if parent and parent["assignments"] else None
    vals = (x or {}).get("values") or {}
    feas = next((e["payload"].get("feasible_designs") for e in v["events"] if e["kind"] == "regime_bound"
                 and e["payload"].get("feasible_designs")), None)
    ctx = {"T": T, "ph": ph, "short": short, "x": x, "px": px, "parent_short": parent_short, "m": m, "status": status,
           "fam": fam, "inst": inst, "cex": cex, "failure_words": failure_words, "diag": diag, "chip_name": chip_name,
           "what": what, "inc": inc, "is_best": is_best, "reused_owner": reused_owner, "feasible": feas,
           "decision_order": [k for k, _ in _decisions(T, ph)], "changes": _design_changes(x, px, ph), "mode": mode,
           "on_boundary": c3 is not None and bool(vals) and vals.get("P") == vals.get("S", 0) - 2 + vals.get("beta", 0),
           "size": optimization.get("size"), "solver": optimization.get("solver"), "cost_model": _math(T["cost"]["model"]),
           "math": _math}
    for lens_dict, make in ((systems, plain.systems), (optimization, plain.optimization), (proof, plain.proof)):
        lens_dict["sections"] = make(ctx)
        lens_dict["text"] = plain.text_of(lens_dict["sections"])
    return {
        "id": contract_id, "short": short, "status": status, "headline": headline, "delta_from_parent": delta,
        "workload": wl_name, "chip": chip_name, "chip_short": chip_short, "chip_kind": chip.get("kind", "illustrative"),
        "parent": T.get("parent"), "parent_short": parent_short,
        "is_best": is_best, "objective": x["objective"] if x else None,
        "rationale": T.get("rationale", ""),
        "systems": systems, "optimization": optimization, "proof": proof,
    }


def describe_all(ledger: Ledger, run: str) -> dict[str, dict]:
    from .story import short_names
    ctx = {"names": short_names(ledger, run), "views": rebuild(ledger, run)}
    out = {}
    for cid in ctx["names"]:
        try:
            out[cid] = describe(ledger, cid, run, ctx)
        except KeyError:
            continue
    return out


# --------------------------------------------------------------------------- pieces
def _delta(T, P, parent_short, ph, mode, wl_name) -> str:
    pre_words = (ph.get("precondition_in_words") or {})
    if P is None:
        return "First template of the run."
    changes = []
    for field in ph.get("delta_fields", []):
        a, b = _get(P, field), _get(T, field)
        if a == b:
            continue
        if field.endswith(".mode"):
            label = (ph.get("precondition_label") or {}).get(b, b)
            changes.append(f"the safety rule is {label}: {pre_words.get(b, b)}")
        elif ".structure." in field:
            name = field.rsplit(".", 1)[1]
            words = ((ph.get("structure_in_words") or {}).get(name) or {}).get(str(b).lower(), f"{name} = {b}")
            changes.append(words)
        elif field.endswith("workload.name"):
            changes.append(f"now on {wl_name}")
        else:
            changes.append(f"{field} = {b}")
    if not changes:
        return f"Same as {parent_short}."
    return f"Same as {parent_short}, but " + "; and ".join(changes) + "."


def _design_words(T, x, ph) -> str:
    vals = x["values"]
    beta = vals.get("beta")
    parts = [f"{vals.get('Tm')} × {vals.get('Tn')} × {vals.get('Tk')} tiles", f"{vals.get('S')} slots",
             f"prefetch {vals.get('P')}", "no trailing barrier" if beta == 0 else "a trailing barrier"]
    return ", ".join(parts)


def _design_changes(x, px, ph) -> list[str]:
    if not x or not px:
        return []
    out = []
    words = ph.get("decision_change_in_words") or {}
    for k in (ph.get("decision_order") or list(x["values"])) + [k for k in x["values"] if k not in (ph.get("decision_order") or [])]:
        if k not in x["values"]:
            continue
        new, old = x["values"][k], px["values"].get(k)
        if old == new:
            continue
        w = words.get(k)
        if isinstance(w, dict):
            out.append(w.get(f"{old}->{new}", f"{k} {new} instead of {old}"))
        elif isinstance(w, str):
            out.append(w.format(old=old, new=new))
        else:
            out.append(f"{k} {new} instead of {old}")
    return out


def _systems(T, x, m, parent, parent_short, ph, mode, status, fam, inst, failure_words, what, chip_name, inc) -> dict:
    decides = ph.get("decides_in_words") or ", ".join(f"{k} ({d.get('meaning', '')})" for k, d in _decisions(T, ph))
    pre_words = (ph.get("precondition_in_words") or {})
    px = _dedupe(parent["assignments"])[-1] if parent and parent["assignments"] else None
    paras = []
    family_refuted = fam is not None and fam["verdict"] != "proved"
    if family_refuted and parent is not None:
        strict = ph.get("precondition_strictness") or {}
        pmode = next((p["mode"] for p in parent["contract"]["preconditions"] if p["id"] == "lifetime"), None)
        rel = "looser" if strict.get(mode, 0) > strict.get(pmode, 0) else ("stricter" if strict.get(mode, 0) < strict.get(pmode, 0) else "different")
        same_wl = parent["contract"]["scope"]["workload"]["name"] == T["scope"]["workload"]["name"]
        paras.append(f"Same template as {parent_short} on {'the same workload' if same_wl else 'a different workload'}, "
                     f"but with a {rel} prefetch rule: {pre_words.get(mode, mode)}."
                     + (" That would let the solver save shared memory." if rel == "looser" else ""))
    else:
        paras.append(f"This template {what}. It decides {decides}.")
    if x:
        chosen = f"On {chip_name}, the solver chose {_design_words(T, x, ph)}."
        pred = f"predicted {x['objective']:.1f} µs"
        if px and parent_short and px["objective"]:
            ratio = px["objective"] / x["objective"] - 1
            if abs(ratio) > 1e-4:
                ch = _design_changes(x, px, ph)
                pred += f", {abs(ratio) * 100:.1f} % {'faster' if ratio > 0 else 'slower'} than {parent_short}" + (
                    f" because {', '.join(ch)}" if ch else "")
            else:
                pred += f", the same design as {parent_short}"
        line = chosen + " " + pred[0].upper() + pred[1:] + "."
        if m:
            util = ", ".join(f"{k} {u * 100:.0f} %" for k, u in m["utilizations"].items())
            line += f" The design is {REGIME_WORDS.get(m['dominant_resource'], m['dominant_resource'])} ({util})."
        paras.append(line)
    if status == "rejected":
        if family_refuted:
            paras.append(f"Rejected before any design was chosen. With this rule, {failure_words}. "
                         "The checker replayed the exact interleaving.")
        elif inst is not None:
            rp = (inst.get("counterexample") or {}).get("replay") or {}
            paras.append(f"Rejected: {failure_words}. The checker replayed the design on a small shape"
                         + (f" and found {rp['differences_total']} output terms wrong." if rp.get("differences_total") else "."))
    elif status == "accepted" and inc:
        pass
    units = ph.get("decision_units") or {}
    table = [{"name": k, "meaning": d.get("meaning", ""), "value": (x or {}).get("values", {}).get(k),
              "parent_value": (px or {}).get("values", {}).get(k), "unit": units.get(k, "")}
             for k, d in _decisions(T, ph)]
    return {"text": "\n\n".join(paras), "design_table": table,
            "parent_values": (px or {}).get("values"), "parent_short": parent_short,
            "chip": chip_name, "structural": T["scope"]["structure"],
            "measurement": {"predicted": m["predicted_cost"], "measured": m["measured_cost"],
                            "utilizations": m["utilizations"], "dominant": m["dominant_resource"]} if m else None}


def _optimization(T, x, parent_short, ph, mode, status, fam, inst, c3, pre, failure_words, v) -> dict:
    decs = dict(_decisions(T, ph))
    groups: list[tuple[list[str], list]] = []
    for k, d in decs.items():
        if groups and groups[-1][1] == d["domain"]:
            groups[-1][0].append(k)
        else:
            groups.append(([k], d["domain"]))
    doms = ", ".join(f"{', '.join(ks)} ∈ {{{', '.join(str(z) for z in dom)}}}" for ks, dom in groups)
    objective = "minimize T with " + _math(T["cost"]["model"]).replace(", T ≥ T_comp + T_mem when", ", and T ≥ T_comp + T_mem when")
    cons = "; ".join(f"{c['id']} {c.get('meaning', '')}, {_math(c['expr'])}" for c in T["constraints"])
    family_refuted = fam is not None and fam["verdict"] != "proved"
    paras = []
    if family_refuted and parent_short and c3:
        paras.append(f"Identical program to {parent_short} except C3 is relaxed to {_math(c3['expr'])}. The feasible region grows.")
        paras.append("The solver never ran: the template failed family certification, so by K1 the enlarged region "
                     "admits designs that are not correct, and solving over it would optimize over wrong designs.")
    else:
        paras.append(f"A mixed-integer linear program over {len(decs)} decisions: {doms}. Objective: {objective}. "
                     f"Constraints: {cons}.")
    size, solver = None, None
    if x:
        d = x.get("diagnostics", {})
        size = {"variables": d.get("variables"), "binary": d.get("binary_variables"), "constraints": d.get("constraints")}
        solver = {"backend": d.get("backend"), "status": "optimal" if "Optimal" in x.get("status", "") else x.get("status"),
                  "gap": x.get("gap"), "bound": x.get("bound"), "objective": x.get("objective"),
                  "time": x.get("solver_time"), "nodes": d.get("nodes")}
        feas = next((e["payload"].get("feasible_designs") for e in v["events"] if e["kind"] == "regime_bound"
                     and e["payload"].get("feasible_designs")), None)
        vals = x["values"]
        on_boundary = c3 is not None and vals.get("P") == vals.get("S") - 2 + vals.get("beta", 0)
        line = (f"Encoded as {size['variables']} variables ({size['binary']} binary) and {size['constraints']} constraints. "
                f"HiGHS: {solver['status']}, gap {x['gap'] * 100:.0f} %, dual bound {x['bound']:.2f}, {_t(x['solver_time'])}.")
        if feas:
            line += f" Of {feas} feasible designs, the optimum " + ("lies on the boundary of C3." if on_boundary else "is interior to C3.")
        elif on_boundary:
            line += " The optimum lies on the boundary of C3."
        paras.append(line)
        if status == "rejected" and inst is not None and inst["verdict"] != "proved":
            paras.append("The design is optimal for this program, but the program's structural choice is outside what the "
                         f"verifier accepts: {failure_words}. The program, not the solution, is wrong.")
    return {"text": "\n\n".join(paras),
            "variables": [{"name": k, "domain": d["domain"], "meaning": d.get("meaning", ""), "kind": d.get("kind")} for k, d in decs.items()],
            "objective": {"model": _math(T["cost"]["model"]), "terms": T["cost"].get("terms", {}), "objective": T["cost"].get("objective")},
            "constraints": [{"id": c["id"], "expr": _math(c["expr"]), "meaning": c.get("meaning", "")} for c in T["constraints"]],
            "precondition": {"expr": _math(pre["expr"]), "mode": mode, "meaning": pre.get("meaning", "")},
            "size": size, "solver": solver, "values": (x or {}).get("values")}


def _proof(T, fam, inst, cex, ph, mode, pre, reused_owner, failure_words, diag) -> dict:
    sch = T["certificate_schema"]
    obl_meaning = {o["name"]: o for o in sch.get("obligations", [])}
    sweep = ph.get("family_sweep_in_words") or "swept over the family"
    paras = []
    obligations = []
    family = None
    if fam:
        prov = fam["provenance"]
        claimed_pts = {tuple(o["family_point"].values()) for o in fam["obligations"] if o.get("family_point") and o.get("claimed_by_precondition")}
        safe_pts = {tuple(p) for p in prov.get("checker_admits", [])}
        family = {"verdict": fam["verdict"], "weakest": prov.get("precondition_is_weakest"), "reused_from": reused_owner,
                  "time": fam["checker_time"], "points_claimed": len(claimed_pts), "points_safe": len(safe_pts) if safe_pts else None}
        head = "Family proof (once per template): C3 is literally the lifetime precondition, so K1 (C ⇒ Pre) holds by construction."
        if fam["verdict"] == "proved":
            if prov.get("reused_from"):
                body = (f" The precondition {_math(pre['expr'])} was certified earlier on this family; this certificate was reused"
                        + (f" from {reused_owner}" if reused_owner else "") + f" (same family key), {_t(fam['checker_time'])}.")
            else:
                weak = prov.get("precondition_is_weakest")
                verdict = ("the checker's safe set equals the precondition's admitted set, so the precondition is the weakest sound one on this family"
                           if weak else f"the precondition is sound but stronger than needed: it admits {len(claimed_pts)} of the {len(safe_pts)} safe points")
                body = f" The precondition {_math(pre['expr'])} was {sweep}; {verdict}. Certified in {_t(fam['checker_time'])}."
            paras.append(head + body)
        else:
            bad = [o for o in fam["obligations"] if o.get("family_point") and o.get("claimed_by_precondition")
                   and not o["status"].startswith(("proved", "agrees"))]
            pts = ", ".join(_point(o["family_point"]) for o in bad)
            ob = bad[0]["obligation"] if bad else (cex or {}).get("obligation", "?")
            paras.append(f"Family proof refuted on the {ob} obligation at {len(bad)} family point{'s' if len(bad) != 1 else ''} "
                         f"the precondition {_math(pre['expr'])} admits, {pts}: the checker finds an interleaving where "
                         f"{failure_words}. Counterexample replayed by the independent interpreter ({_t(fam['checker_time'])}). "
                         "The precondition is not sound; the diagnosis says to strengthen Pre (and C3 with it, to keep K1).")
        for o in fam["obligations"]:
            if o.get("family_point") and not o.get("claimed_by_precondition"):
                continue
            obligations.append({"name": o["obligation"], "kind": "family", "scope": o.get("scope", ""), "status": o["status"],
                                "time": o.get("seconds"), "reused": o.get("reused", False), "point": o.get("family_point")})
    instance = None
    if inst:
        n = len(inst["obligations"])
        ok = sum(1 for o in inst["obligations"] if o["status"].startswith(("proved", "agrees")))
        instance = {"verdict": inst["verdict"], "n": n, "proved": ok, "time": inst["checker_time"]}
        idx = [o["name"].replace("_", "-") for o in sch.get("obligations", []) if o.get("kind") == "index"]
        schd = [o for o in sch.get("obligations", []) if o.get("kind") == "schedule"]
        if inst["verdict"] == "proved":
            paras.append(f"Instance check (this design): {n} obligations proved in {_t(inst['checker_time'])}. "
                         f"Index: {', '.join(idx)} over all extents. Schedule: "
                         + " and ".join(f"{o['name']} ({o['meaning']})" for o in schd)
                         + " over all interleavings; terminal agreement with the reference.")
        else:
            rp = (inst.get("counterexample") or {}).get("replay") or {}
            ob = (inst.get("counterexample") or {}).get("obligation", "?")
            meaning = obl_meaning.get(ob.split("[")[0], {}).get("meaning", "")
            paras.append(f"Instance check refuted on {ob}" + (f" ({meaning})" if meaning else "") + f": {failure_words}."
                         + (f" Replayed on shape {rp['shape']}: {rp['differences_total']} output terms differ from the reference."
                            if rp.get("shape") else "")
                         + f" {ok} of {n} obligations passed, {_t(inst['checker_time'])}.")
        for o in inst["obligations"]:
            obligations.append({"name": o["obligation"], "kind": "instance", "scope": o.get("scope", ""), "status": o["status"],
                                "time": o.get("seconds"), "reused": o.get("reused", False), "point": None})
    if not fam and not inst:
        paras.append("Not certified yet.")
    paras.append(f"Backward map: {T['backward_map']}. Rank: {sch.get('rank', '?')}.")
    return {"text": "\n\n".join(paras), "family": family, "instance": instance, "obligations": obligations,
            "counterexample": cex, "relation": sch.get("relation"), "rank": sch.get("rank"),
            "backward_map": T["backward_map"], "diagnosis": diag}
