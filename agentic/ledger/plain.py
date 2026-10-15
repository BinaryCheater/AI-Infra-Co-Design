"""Plain-language sections for the inspector's tabs (Systems, Optimization, Proof).

Each tab is a list of sections; each section answers one question a person would ask, in short sentences and plain words:
  {"title": str, "paras": [str], "items": [str], "table": {"cols": [...], "rows": [[...]]} | None,
   "bars": [{"label": str, "value": float, "main": bool}] | None, "details": {"summary": str, "items": [str]} | None}
The words come from the template's phrasing file (templates/<name>/describe.yaml: decision_words, meaning_in_words,
cost_in_words, constraints_in_words, obligations_in_words, ...) and from the ledger's records. Formal expressions and solver
statistics are kept, but only inside "details", collapsed, for specialists.
"""
from __future__ import annotations

from math import prod

CHECKED_TEMPLATES = {"smem_tiling"}       # the templates the checker and the solver handle (agents/orchestrator.py)

RESOURCE_WORDS = {"compute": "the arithmetic units", "DRAM": "memory bandwidth", "copy": "waiting for data copies to arrive",
                  "barrier": "threads waiting for each other", "smem": "shared-memory loads", "compute_queue": "the arithmetic units"}
BAR_LABELS = {"compute": "arithmetic units", "DRAM": "memory bandwidth"}


def _t(seconds) -> str:
    if seconds is None:
        return "an unknown time"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.1f} s"


def _plural(n, word, plural=None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _or(words: list[str]) -> str:
    words = [str(w) for w in words]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def label(ph, name) -> str:
    dw = (ph.get("decision_words") or {}).get(name) or {}
    return dw.get("label", name)


def value_words(ph, name, value) -> str:
    dw = (ph.get("decision_words") or {}).get(name) or {}
    vals = dw.get("values") or {}
    for k in (value, str(value)):
        if k in vals:
            return vals[k]
    if isinstance(value, bool) and int(value) in vals:
        return vals[int(value)]
    return str(value)


def unit(ph, name) -> str:
    return ((ph.get("decision_words") or {}).get(name) or {}).get("unit", "")


def meaning(ph, name, T) -> str:
    return (ph.get("meaning_in_words") or {}).get(name) or T["decisions"][name].get("meaning", "")


def point_words(fp: dict) -> str:
    s, p, b = fp.get("S"), fp.get("P"), fp.get("beta")
    return f"{_plural(s, 'buffer')}, loading {_plural(p, 'step')} ahead, barrier after each step {'kept' if b else 'dropped'}"


def _fill(text: str, T: dict) -> str:
    nums = (T.get("cost") or {}).get("numbers") or {}
    try:
        return text.format(**{k: (f"{v:,.0f}" if isinstance(v, (int, float)) else v) for k, v in nums.items()})
    except (KeyError, IndexError, ValueError):
        return text


def section(title, paras=None, items=None, table=None, bars=None, details=None) -> dict:
    return {"title": title, "paras": [p for p in (paras or []) if p], "items": items or [], "table": table, "bars": bars,
            "details": details}


def text_of(sections: list[dict]) -> str:
    out = []
    for s in sections:
        out += s["paras"]
        out += s["items"]
    return "\n\n".join(out)


# ------------------------------------------------------------------------------------------------ Systems: what it does
def systems(c: dict) -> list[dict]:
    T, ph = c["T"], c["ph"]
    out = []
    names = list(c["decision_order"])
    out.append(section("What this template does",
                       [ph.get("intro_in_words", "") or f"{c['short']} {c['what']}.", "It decides:"],
                       [f"{label(ph, k)}: {meaning(ph, k, T)}." for k in names]))
    x, px, ps = c["x"], c["px"], c["parent_short"]
    if x:
        rows = []
        for k in names:
            v = x["values"].get(k)
            row = [label(ph, k) + (f" ({unit(ph, k)})" if unit(ph, k) else ""), value_words(ph, k, v)]
            if px:
                row.append(value_words(ph, k, px["values"].get(k)))
            rows.append(row)
        paras = [f"The solver picked this design. By the time formula it takes {x['objective']:.1f} µs."]
        if px and ps and px.get("objective"):
            ratio = px["objective"] / x["objective"] - 1
            if abs(ratio) > 1e-4:
                ch = c["changes"]
                paras.append(f"That is {abs(ratio) * 100:.1f} % {'faster' if ratio > 0 else 'slower'} than {ps}"
                             + (f", because {', and '.join(ch)}." if ch else "."))
            elif any(x["values"].get(k) != px["values"].get(k) for k in names):
                paras.append(f"By the formula it takes the same time as {ps}, with a different shape (see the table).")
            else:
                paras.append(f"It is the same design as {ps}.")
        out.append(section("What was chosen", paras, table={"cols": ["Choice", "Chosen"] + ([f"In {ps}"] if px else []), "rows": rows}))
    else:
        if c["status"] == "rejected":
            why = "Nothing was chosen: the checker found a bug in the template before the solver ran."
        elif T["template"] not in CHECKED_TEMPLATES:
            why = "Nothing has been chosen yet. There is no solver for this level yet."
        else:
            why = "Nothing has been chosen yet. The solver picks a design after the checker has proved the template correct."
        out.append(section("What was chosen", [why]))
    m = c["m"]
    if m:
        sim = str(m.get("simulator", ""))
        how = ("measured on the Spark" if sim.startswith("cuda:") else
               "simulated on the GB10 model" if sim.startswith("mechanistic") else "estimated again with the same formula (not a real measurement)")
        diff = m["measured_cost"] / m["predicted_cost"] - 1 if m.get("predicted_cost") else None
        paras = [f"The time formula predicted {m['predicted_cost']:.1f} µs.",
                 f"{how[0].upper() + how[1:]}, it takes {m['measured_cost']:.1f} µs"
                 + (f", {abs(diff) * 100:.0f} % {'more' if diff > 0 else 'less'} than predicted." if diff is not None and abs(diff) > 0.005 else ".")]
        dom = m.get("dominant_resource")
        u = m.get("utilizations") or {}
        if dom:
            busy = u.get(dom)
            if isinstance(busy, (int, float)) and busy > 1.0:
                paras.append(f"What limits it: {RESOURCE_WORDS.get(dom, dom)}. The formula's count of this work comes to "
                             f"{busy * 100:.0f} % of what the chip could do in the measured time, which is impossible, so the "
                             "formula counts more of it than really happened.")
            else:
                paras.append(f"What limits it: {RESOURCE_WORDS.get(dom, dom)}" + (f", busy {busy * 100:.0f} % of the time." if isinstance(busy, (int, float)) else "."))
        bars = [{"label": BAR_LABELS[k], "value": float(u[k]), "main": k == dom} for k in ("compute", "DRAM") if k in u]
        out.append(section("How fast it runs", paras, bars=bars))
    else:
        out.append(section("How fast it runs", ["Not measured yet." + (" The simulator runs after the solver." if T["template"] in CHECKED_TEMPLATES else "")]))
    s_words = []
    for k, v in (T["scope"].get("structure") or {}).items():
        w = (((ph.get("structure_in_words") or {}).get(k)) or {}).get(str(v).lower())
        if w:
            s_words.append(w[0].upper() + w[1:] + ".")
    if s_words:
        out.append(section("Fixed parts of this template", ["These are not choices; every design of this template has them:"], s_words))
    out.append(section("Where it runs", [f"On {c['chip_name']}. The hardware is fixed in this study."]))
    out.append(section("Status", [status_words(c)]))
    return out


def status_words(c: dict) -> str:
    T, st, inc = c["T"], c["status"], c["inc"]
    if st == "accepted":
        if c["is_best"]:
            return "Accepted. It is the fastest design found so far for this workload."
        if inc and c["x"]:
            return f"Accepted. It is {(c['x']['objective'] / inc['cost'] - 1) * 100:.1f} % slower than the best so far."
        return "Accepted."
    if st == "rejected":
        if c["fam"] is not None and c["fam"]["verdict"] != "proved":
            return f"Rejected before any design was chosen: {c['failure_words']}. The Proof tab shows how."
        return f"Rejected, because the design the solver chose is wrong: {c['failure_words']}. The Proof tab shows how."
    if st == "proved safe":
        return "Proved correct for every design it allows. The solver picks a design next."
    if T["template"] not in CHECKED_TEMPLATES:
        return ("Proposed. Nothing checks this level yet, so it stays a proposal: it is neither accepted nor rejected. "
                "The next step is either a checker and a solver for this level, or a template one level down.")
    return "Proposed. Next, the checker tries to prove it correct."


# ------------------------------------------------------------------------------------------------ Optimization: the problem
def optimization(c: dict) -> list[dict]:
    T, ph = c["T"], c["ph"]
    names = list(c["decision_order"])
    total = prod(len(T["decisions"][k]["domain"]) for k in names)
    out = []
    rows = [[label(ph, k), _or([value_words(ph, k, v) for v in T["decisions"][k]["domain"]]) + (f" {unit(ph, k)}" if unit(ph, k) else ""),
             meaning(ph, k, T)] for k in names]
    out.append(section("The choices", [f"{_plural(len(names), 'choice')}, {total:,} combinations in all."],
                       table={"cols": ["Choice", "Options", "What it means"], "rows": rows}))
    out.append(section("The goal", [ph.get("goal_in_words") or "Make it finish as fast as possible.", "The time is estimated like this:"],
                       [_fill(w, T) for w in (ph.get("cost_in_words") or [])]))
    cw = dict(ph.get("constraints_in_words") or {})
    pre_words = (ph.get("precondition_in_words") or {}).get(c.get("mode"))
    if pre_words and "C3" in cw:
        cw["C3"] = f"The safety rule: {pre_words}."
    feas = c["feasible"]
    out.append(section("The rules", ["A design is allowed only if:"],
                       [cw.get(r["id"]) or (r.get("meaning", r["expr"])[0].upper() + r.get("meaning", r["expr"])[1:] + ".") for r in T["constraints"]]
                       + ([f"{feas:,} of the {total:,} combinations pass the rules."] if feas else [])))
    x = c["x"]
    if x:
        paras = [f"The solver compared every allowed combination by the time formula and picked the fastest, in {_t(x.get('solver_time'))}."]
        if (x.get("gap") or 0) < 1e-9:
            paras.append("It also proved that no other allowed combination is faster by the formula.")
        if c["on_boundary"]:
            paras.append("The chosen design sits right at the edge of the safety rule: it loads as far ahead as the rule allows.")
        if c["status"] == "rejected" and c["inst"] is not None and c["inst"]["verdict"] != "proved":
            paras.append(f"But the checker then found the design wrong: {c['failure_words']}. The rules let it through, so the "
                         "template's rules, not the solver, are at fault.")
        how = section("How it was solved", paras)
    elif c["fam"] is not None and c["fam"]["verdict"] != "proved":
        how = section("How it was solved", ["The solver did not run. The checker found a bug in the template first, and "
                                            "picking the fastest among designs that may be wrong would be pointless."])
    elif T["template"] not in CHECKED_TEMPLATES:
        how = section("How it was solved", [f"Not solved yet. No solver handles this level yet. With {total:,} combinations, "
                                            "they could even be compared by hand."])
    else:
        how = section("How it was solved", ["The solver has not run yet. It runs after the checker."])
    spec = [f"Time formula: {c['cost_model']}."] + [f"Rule {r['id']}: {c['math'](r['expr'])}." for r in T["constraints"]]
    if c["size"]:
        s, sv = c["size"], c["solver"]
        spec.append(f"Solver input: {s['variables']} variables ({s['binary']} yes/no), {s['constraints']} constraints.")
        spec.append(f"Solver: {sv['backend']}, status {sv['status']}, gap {(sv['gap'] or 0) * 100:.0f} %, best possible bound "
                    f"{sv['bound']:.2f} µs, {sv['nodes']} search nodes.")
    how["details"] = {"summary": "The formula and the solver's numbers, for specialists", "items": spec}
    out.append(how)
    return out


# ------------------------------------------------------------------------------------------------ Proof: is it correct?
DIAG_WORDS = {
    "contract": "The template itself is wrong. The fix is to change the template, not the solver or the time formula.",
    "constraints": "The rules let through a design that breaks the safety rule. The fix is to tighten the rules.",
    "cost_model": "The time formula and the measurement disagree. The fix is to improve the formula, not the template.",
    "model": ("A measured time came out faster than the formula predicted, so the formula can no longer be trusted as a lower "
              "limit. Decisions that relied on it are reopened."),
    "trusted_base": ("A design the checker proved correct computed a wrong answer on the hardware. The checker's assumptions "
                     "about the hardware must be wrong somewhere, so the run was stopped."),
}


def proof(c: dict) -> list[dict]:
    T, ph = c["T"], c["ph"]
    sch = T["certificate_schema"]
    ow = ph.get("obligations_in_words") or {}
    obls = sch.get("obligations", [])
    out = [section("What must be true", ["For this template to be correct, all of these must hold:"],
                   [ow.get(o["name"]) or (o.get("meaning", o["name"])[0].upper() + o.get("meaning", o["name"])[1:] + ".") for o in obls])]
    fam, inst = c["fam"], c["inst"]
    paras, items = [], []
    if not fam and not inst:
        if T["template"] not in CHECKED_TEMPLATES:
            paras.append("Nothing has been checked yet. There is no checker for this level yet, so this proposal can be "
                         "neither accepted nor rejected until one is built.")
        else:
            paras.append("Not checked yet. The checker runs next.")
    if fam:
        prov = fam["provenance"]
        if fam["verdict"] == "proved":
            if prov.get("reused_from"):
                paras.append("Step 1, all designs at once: the safety rule was already proved for an earlier template"
                             + (f" ({c['reused_owner']})" if c["reused_owner"] else "") + f", so the proof was reused ({_t(fam['checker_time'])}).")
            else:
                paras.append("Step 1, all designs at once: " + (ph.get("family_check_in_words") or "the checker tried every combination the template allows.").strip()
                             + f" Every combination the safety rule allows came out safe ({_t(fam['checker_time'])}).")
                if prov.get("precondition_is_weakest"):
                    paras.append("The safety rule is the loosest safe one: it allows every combination that is safe, and nothing else.")
                elif prov.get("precondition_is_weakest") is False:
                    paras.append("The safety rule is safe but stricter than it needs to be: some safe combinations are ruled out.")
        else:
            bad = [o for o in fam["obligations"] if o.get("family_point") and o.get("claimed_by_precondition")
                   and not o["status"].startswith(("proved", "agrees"))]
            pts = "; ".join(point_words(o["family_point"]) for o in bad[:3])
            paras.append(f"Step 1, all designs at once: the checker found a bug. With this safety rule, {c['failure_words']}."
                         + (f" It happens with {pts}." if pts else "")
                         + " The checker replayed the exact order of events; it is shown below.")
    if inst:
        n = len(inst["obligations"])
        ok = sum(1 for o in inst["obligations"] if o["status"].startswith(("proved", "agrees")))
        if inst["verdict"] == "proved":
            paras.append(f"Step 2, the chosen design: all {n} checks passed ({_t(inst['checker_time'])}).")
        else:
            rp = ((inst.get("counterexample") or {}).get("replay")) or {}
            paras.append(f"Step 2, the chosen design: a bug was found. {c['failure_words'][0].upper() + c['failure_words'][1:]}."
                         + (f" On a small test shape, {rp['differences_total']} values of C came out wrong." if rp.get("differences_total") else "")
                         + f" {ok} of {n} checks passed.")
        dims = ph.get("obligations_by_dimension_in_words") or {}
        for o in inst["obligations"]:
            name = o["obligation"]
            base = name.split("[")[0].split("(")[0]
            dim = name[len(base):].strip("[]") if "[" in name else None
            good = o["status"].startswith(("proved", "agrees"))
            words = (dims.get(base) or {}).get(dim) if dim else None
            items.append(("✓ " if good else "✗ ") + (words or ow.get(base) or base.replace("_", " ")))
    out.append(section("What the checker did", paras, items))
    if c["diag"]:
        d = c["diag"]
        out.append(section("What the optimizer concluded", [DIAG_WORDS.get(d.get("source"), d.get("response", ""))],
                           details={"summary": "The optimizer's own record", "items": [f"Symptom: {d.get('symptom')}", f"Response: {d.get('response')}"]}))
    out.append(section("For specialists", [], details={
        "summary": "The formal statement of what is proved",
        "items": [f"Relation: {sch.get('relation')}", f"Progress measure: {sch.get('rank')}", f"Backward map: {T.get('backward_map')}"]
                 + [f"Check {o['name']} ({o.get('kind')}, over {o.get('scope')}): {o.get('meaning')}" for o in obls]}))
    return out
