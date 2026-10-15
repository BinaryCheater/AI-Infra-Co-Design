"""The story view: one plain-language entry per iteration, composed from the ledger alone
(notes/architect_console_design.md §4). Pure function over the ledger, like timeline.py.

Sentence order never changes: proposed -> verifier -> solver -> gate -> evaluator -> orchestrator.
Clauses whose event is absent are dropped; a rejection ends the paragraph. Templates get short names
T1, T2, ... in order of first `proposed` within the run; an edited or injected proposal keeps the
number of the proposal it replaced.
"""
from __future__ import annotations

from .describe import display_name
from .ledger import Ledger
from .views import rebuild

STATIONS = ("formulator", "verifier", "solver", "gate", "evaluator", "orchestrator")
REGIME_WORDS = {"compute": "compute-bound", "DRAM": "memory-bound"}


def _t(seconds) -> str:
    if seconds is None:
        return "?"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.2f} s"


def _regime(s: str | None) -> str:
    return (s or "").split(":")[0].strip() or "unknown"


def short_names(ledger: Ledger, run: str) -> dict[str, str]:
    """contract id -> T<n>, numbered by iteration of first proposal; edits and injects keep the number."""
    names: dict[str, str] = {}
    by_iteration: dict[int, str] = {}
    for ev in ledger.events(run, kinds=["proposed", "human_edit", "human_inject"]):
        it = ev["iteration"]
        if ev["kind"] == "proposed":
            if it not in by_iteration:
                by_iteration[it] = f"T{len(by_iteration) + 1}"
            names.setdefault(ev["artifact"], by_iteration[it])
        else:
            names[ev["artifact"]] = by_iteration.get(it, names.get(ev["payload"].get("from") or
                                                                   ev["payload"].get("replaced"), "T?"))
    return names


def story(ledger: Ledger, run: str) -> list[dict]:
    names = short_names(ledger, run)
    views = rebuild(ledger, run)
    incumbents = {k.split("|")[0]: v for k, v in views["incumbents"].items()}
    entries: dict[int, dict] = {}
    seen_templates: set[str] = set()
    cert_owner: dict[str, str] = {}      # family certificate id -> contract id that first earned it

    for ev in ledger.events(run):
        it, k, p, a = ev["iteration"], ev["kind"], ev["payload"], ev["artifact"]
        if it == 0:
            continue
        e = entries.setdefault(it, {
            "iteration": it, "title": f"Iteration {it}", "sentence": "", "clauses": [], "outcome": "in_progress",
            "flip": False, "level": None, "descended": False, "visits": [],
            "template": None, "pool_delta": {"added": [], "reused_from": None, "reused_from_short": None,
                                             "incumbent": False, "incumbent_cost": None},
            "human": [], "paused_at": None, "last_seq": ev["seq"]})
        e["last_seq"] = ev["seq"]
        clauses = e["clauses"]
        if k == "request":
            e["level"] = p["level"]
        elif k == "proposed":
            T = ledger.get(a)
            e["template"] = {"id": a, "short": names.get(a, "T?"), "workload": T["scope"]["workload"]["name"],
                             "refines": p.get("refines"), "regime": _regime(T["structure"]["regime"]),
                             "rationale": T.get("rationale", ""), "parent": T.get("parent"),
                             "parent_short": names.get(T.get("parent") or "", None)}
            e["title"] = f"Iteration {it} · {p.get('refines') or 'proposal'}"
            if a not in seen_templates:
                seen_templates.add(a)
                e["pool_delta"]["added"].append(a)
            e["visits"].append({"station": "formulator", "verdict_text": f"proposed {e['template']['short']}",
                                "time_s": p.get("formulator_time"), "ok": True, "artifact": a, "seq": ev["seq"]})
            clauses.append({"station": "formulator", "text":
                            f"The agent proposed {p.get('refines') or 'a template'} for "
                            f"{display_name(e['template']['workload'])}, assuming {e['template']['regime']}."})
        elif k == "note" and p.get("not_checked"):
            e["outcome"] = "not_checked"
            e["visits"].append({"station": "orchestrator", "verdict_text": "not checked yet", "time_s": None,
                                "ok": True, "artifact": a, "seq": ev["seq"]})
            clauses.append({"station": "orchestrator", "text": f"Not checked yet: there is no checker, solver or simulator "
                                                               f"step for the level-{p.get('level')} template yet, so it stays "
                                                               f"in the pool as proposed."})
        elif k in ("human_edit", "human_inject"):
            new = a
            T = ledger.get(new)
            if e["template"]:
                e["template"]["id"] = new
                e["template"]["refines"] = T["provenance"].get("refines")
                e["template"]["regime"] = _regime(T["structure"]["regime"])
            e["pool_delta"]["added"] = [x for x in e["pool_delta"]["added"] if x != p.get("from") and x != p.get("replaced")] + [new]
            e["human"].append({"kind": k, "who": p.get("who"), "reason": p.get("reason"), "seq": ev["seq"]})
            verb = "You changed the proposal" if k == "human_edit" else "You inserted a template from the pool"
            clauses.insert(0, {"station": "formulator", "text": f"{verb}" + (f" ({p['reason']})." if p.get("reason") else ".")})
        elif k in ("human_override", "human_refused"):
            e["human"].append({"kind": k, "who": p.get("who"), "reason": p.get("reason"), "seq": ev["seq"]})
            if k == "human_override":
                clauses.append({"station": "orchestrator", "text": f"You decided: {p.get('decision')}."})
            else:
                clauses.append({"station": "orchestrator", "text": "Not allowed: a verdict cannot be changed; "
                                                                    "correctness obligations gate, for humans too."})
        elif k == "rejected":
            e["outcome"] = "rejected"
            e["visits"].append({"station": "orchestrator", "verdict_text": "rejected: malformed", "time_s": None,
                                "ok": False, "artifact": a, "seq": ev["seq"]})
            clauses.append({"station": "orchestrator", "text": "Rejected: the proposal was malformed ("
                            + "; ".join(p.get("problems", [])) + ")."})
        elif k == "family_check":
            V = ledger.get(a)
            reused = V["provenance"].get("reused_from")
            if p["verdict"] == "proved":
                if reused:
                    owner = cert_owner.get(reused)
                    e["pool_delta"]["reused_from"] = owner
                    e["pool_delta"]["reused_from_short"] = names.get(owner) if owner else None
                    text = f"The verifier reused an earlier proof ({_t(p['checker_time'])})"
                    text += f", from {names.get(owner, 'T?')}." if owner else "."
                    badge = f"proof reused · {_t(p['checker_time'])}"
                else:
                    cert_owner.setdefault(a, e["template"]["id"] if e["template"] else None)
                    text = f"The verifier proved it safe for all designs in {_t(p['checker_time'])}."
                    badge = f"proved safe · {_t(p['checker_time'])}"
                e["visits"].append({"station": "verifier", "verdict_text": badge, "time_s": p["checker_time"],
                                    "ok": True, "artifact": a, "seq": ev["seq"]})
                clauses.append({"station": "verifier", "text": text})
            else:
                ob = (V.get("counterexample") or {}).get("obligation", "?")
                e["outcome"] = "rejected"
                e["visits"].append({"station": "verifier", "verdict_text": f"rejected: bug found in {ob}",
                                    "time_s": p["checker_time"], "ok": False, "artifact": a, "seq": ev["seq"]})
                clauses.append({"station": "verifier", "text": f"Rejected: the verifier found a bug in {ob}; "
                                                               "the failing case is replayed in the inspector."})
        elif k == "solved":
            gap = p.get("gap") or 0.0
            e["visits"].append({"station": "solver", "verdict_text": f"best design · {_t(p['solver_time'])} · gap {gap:.0%}",
                                "time_s": p["solver_time"], "ok": True, "artifact": a, "seq": ev["seq"]})
            clauses.append({"station": "solver", "text": f"The solver found the best design in {_t(p['solver_time'])} "
                                                         f"(gap {gap:.0%})."})
        elif k == "instance_check":
            if p["verdict"] == "proved":
                e["visits"].append({"station": "gate", "verdict_text": f"checked · {_t(p['checker_time'])}",
                                    "time_s": p["checker_time"], "ok": True, "artifact": a, "seq": ev["seq"]})
                clauses.append({"station": "gate", "text": f"The gate checked this design in {_t(p['checker_time'])}."})
            else:
                W = ledger.get(a)
                ob = (W.get("counterexample") or {}).get("obligation", "?").split("[")[0]
                e["outcome"] = "rejected"
                if p["verdict"] == "open":
                    e["visits"].append({"station": "gate", "verdict_text": "could not decide", "time_s": p["checker_time"],
                                        "ok": False, "artifact": a, "seq": ev["seq"]})
                    clauses.append({"station": "gate", "text": f"Rejected: the checker could not decide {ob}; "
                                                               "the design is not trusted."})
                else:
                    e["visits"].append({"station": "gate", "verdict_text": f"rejected: bug in {ob}",
                                        "time_s": p["checker_time"], "ok": False, "artifact": a, "seq": ev["seq"]})
                    clauses.append({"station": "gate", "text": f"Rejected: this design has a bug in {ob}."})
        elif k == "measured":
            regime = REGIME_WORDS.get(p["dominant_resource"], p["dominant_resource"])
            sim = p.get("simulator", "")
            if sim.startswith("cuda:"):
                host = p.get("host") or sim[5:]
                gap = p.get("discrepancy", 0.0)
                e["visits"].append({"station": "evaluator", "verdict_text": f"measured {p['measured']:.1f} µs · predicted {p['predicted']:.1f} ({gap:+.0%})",
                                    "time_s": p.get("evaluator_time"), "ok": p.get("reference_agrees") is not False, "artifact": a, "seq": ev["seq"]})
                if p.get("reference_agrees") is False:
                    e["outcome"] = "rejected"
                    clauses.append({"station": "evaluator", "text": f"Stopped: the design computed a wrong answer on {host}; "
                                                                    "the verifier's hardware assumptions need review."})
                else:
                    word = "slower" if gap > 0 else "faster"
                    clauses.append({"station": "evaluator", "text": f"Measured on {host}: {p['measured']:.1f} µs, {abs(gap):.0%} {word} than "
                                    f"predicted, {regime}." + (" The model is now known to be optimistic in this regime." if gap > 0.05 else
                                                              (" The model is now known to be pessimistic here." if gap < -0.05 else ""))})
            else:
                e["visits"].append({"station": "evaluator", "verdict_text": f"{regime} · {p['measured']:.0f} µs",
                                    "time_s": p.get("evaluator_time"), "ok": True, "artifact": a, "seq": ev["seq"]})
                clauses.append({"station": "evaluator", "text": f"Measured {p['measured']:.2f} µs, {regime}."})
        elif k == "extrapolating":
            changes = ", ".join(f"{f} {v['measured']} → {v['template']}" for f, v in p.get("fields", {}).items())
            clauses.append({"station": "formulator", "text": f"This template changes the chip ({changes}), which {p.get('measured_contract', 'the machine')} "
                                                             "cannot run; costs are modeled, and the regime is only assumed."})
        elif k == "stopped":
            e["outcome"] = "rejected"
            e["visits"].append({"station": "orchestrator", "verdict_text": "stopped: trusted base", "time_s": None, "ok": False,
                                "artifact": a, "seq": ev["seq"]})
        elif k == "regime_flip":
            e["flip"] = True
            clauses.append({"station": "evaluator", "text": f"The bottleneck moved from {p['assumed']} to {p['actual']}."})
        elif k == "regime_bound":
            if p.get("holds_as_bound"):
                clauses.append({"station": "orchestrator", "text": "The assumption is now proven on every feasible design."})
        elif k == "split":
            if p.get("children") and p["children"][1]:
                e["descended"] = True
        elif k == "skip_regime":
            clauses.append({"station": "orchestrator", "text": f"The other regime cannot beat the best design by more "
                                                               f"than {p['eps']:.0%}, so we stayed."})
        elif k == "probe_result":
            bound = p.get("bound_on_missed")
            clauses.append({"station": "orchestrator", "text": f"Sampled the other regime {p['n']} times, {p['wins']} wins"
                            + (f"; chance of a missed win below {bound:.0%}." if bound is not None else ".")
                            + (" We go one level deeper." if p.get("decision") == "descend" else " We stayed.")})
        elif k == "diagnosis":
            # a rejection stops the token at the red station; the orchestrator's conclusion is narrated only
            clauses.append({"station": "orchestrator", "text": f"The orchestrator concluded: {p['symptom']}. "
                                                               f"Cause: {p['source']}. Response: {p['response']}."})
        elif k == "admitted":
            e["outcome"] = "accepted"
            wl = e["template"]["workload"] if e["template"] else None
            inc = incumbents.get(wl)
            is_inc = bool(inc and inc["contract"] == a)
            e["pool_delta"]["incumbent"] = is_inc
            e["pool_delta"]["incumbent_cost"] = inc["cost"] if is_inc else None
            badge = "accepted" + (" · bottleneck moved → stayed" if e["flip"] and not e["descended"] else "")
            e["visits"].append({"station": "orchestrator", "verdict_text": badge, "time_s": None, "ok": True,
                                "artifact": a, "seq": ev["seq"]})
            clauses.append({"station": "orchestrator", "text": "Accepted into the pool."
                            + (f" New best so far for {display_name(wl)}." if is_inc else "")})
        elif k == "rewind":
            e["human"].append({"kind": k, "who": p.get("who"), "reason": None, "seq": ev["seq"]})
            e["outcome"] = "rewound"
            clauses.append({"station": "orchestrator", "text": f"You rewound to the {p.get('checkpoint')} pause point."})
        elif k == "checkpoint":
            e["paused_at"] = p["name"]

    out = []
    for it in sorted(entries):
        e = entries[it]
        e["sentence"] = " ".join(c["text"] for c in e["clauses"])
        out.append(e)
    return out


def story_entry(ledger: Ledger, run: str, iteration: int) -> dict | None:
    return next((e for e in story(ledger, run) if e["iteration"] == iteration), None)
