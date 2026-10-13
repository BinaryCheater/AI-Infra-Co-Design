"""The timeline view: one row per iteration, built from the ledger alone (design §6.1)."""
from __future__ import annotations

from .ledger import Ledger

ACCEPTED = ("proved", "agrees")


def timeline(ledger: Ledger, run: str) -> list[dict]:
    rows: dict[int, dict] = {}
    for ev in ledger.events(run):
        it, k, p, a = ev["iteration"], ev["kind"], ev["payload"], ev["artifact"]
        if it == 0:
            continue
        row = rows.setdefault(it, {
            "iteration": it, "step": None, "level": None, "refines": None, "workload": None, "contract": None,
            "structure": None, "bucket": None, "regime_assumed": None, "regime_actual": None, "strength": None,
            "strength_final": None, "bound": None, "gap": None, "solver_time": None, "objective": None,
            "family": None, "family_verdict": None, "instance": None, "instance_verdict": None,
            "predicted": None, "measured": None, "discrepancy": None, "flip": False, "regime_decision": None,
            "outcome": "in progress", "diagnosis": None, "checkpoints": [], "human": [], "replayed": False,
            "first_seq": ev["seq"], "last_seq": ev["seq"]})
        row["last_seq"] = ev["seq"]
        if k == "request":
            row["step"], row["level"] = p["step"], p["level"]
        elif k == "proposed":
            row["contract"] = a
            row["refines"] = p.get("refines")
            row["workload"] = (p.get("workload") or {}).get("name")
            s = p["structure"]
            row["structure"], row["bucket"], row["strength"] = s["name"], s["bucket"], s["strength"]
            row["regime_assumed"] = s["regime"].split(":")[0].strip()
        elif k in ("human_edit", "human_inject"):
            row["contract"] = a
            row["human"].append({"kind": k, "seq": ev["seq"], **p})
        elif k in ("human_override", "human_refused"):
            row["human"].append({"kind": k, "seq": ev["seq"], **p})
        elif k == "replay":
            row["replayed"] = True
        elif k == "rejected":
            row["outcome"] = "rejected (malformed)"
            row["diagnosis"] = "; ".join(p.get("problems", []))
        elif k == "family_check":
            row["family"], row["family_verdict"] = a, p["verdict"]
            if p["verdict"] != "proved":
                row["outcome"] = "refuted at family certification"
        elif k == "solved":
            row["bound"], row["gap"], row["solver_time"], row["objective"] = p["bound"], p["gap"], p["solver_time"], p["objective"]
        elif k == "instance_check":
            row["instance"], row["instance_verdict"] = a, p["verdict"]
            if p["verdict"] != "proved":
                row["outcome"] = f"{p['verdict']} at instance gate"
        elif k == "measured":
            row["predicted"], row["measured"], row["discrepancy"] = p["predicted"], p["measured"], p["discrepancy"]
            row["regime_actual"] = {"compute": "compute-bound", "DRAM": "memory-bound"}.get(
                p["dominant_resource"], p["dominant_resource"])
        elif k == "regime_flip":
            row["flip"] = True
        elif k in ("skip_regime", "probe_result"):
            row["regime_decision"] = p.get("decision") or ("keep: " + p.get("reason", ""))
        elif k == "regime_bound":
            row["strength_final"] = p.get("strength_now")
        elif k == "diagnosis":
            row["diagnosis"] = f"{p['source']}: {p['symptom']}"
        elif k == "admitted":
            row["outcome"] = "admitted"
            row["strength_final"] = p.get("strength")
        elif k == "checkpoint":
            row["checkpoints"].append({"seq": ev["seq"], "name": p["name"], "pending": p.get("pending")})
        elif k == "rewind":
            row["outcome"] = f"rewound to checkpoint {p['to']}"
            row["human"].append({"kind": k, "seq": ev["seq"], **p})
    return [rows[i] for i in sorted(rows)]


def template_view(ledger: Ledger, contract_id: str, run: str | None = None) -> dict:
    """Everything the template view shows for one contract: the why, the three faces, the evidence."""
    T = ledger.get(contract_id)
    out = {"contract": T, "assignments": [], "verdicts": [], "measurements": [], "diagnoses": [],
           "events": [], "lineage": [], "admitted": None}
    cur = T.get("parent")
    while cur:
        try:
            parent = ledger.get(cur)
        except KeyError:
            break
        out["lineage"].append({"id": cur, "refines": parent["provenance"].get("refines"),
                               "workload": parent["scope"]["workload"].get("name")})
        cur = parent.get("parent")
    for ev in ledger.events(run):
        a = ev["artifact"]
        if not a:
            continue
        if a == contract_id:
            out["events"].append(ev)
            if ev["kind"] == "admitted":
                out["admitted"] = ev["payload"]
            continue
        try:
            kind = ledger.kind_of(a)
        except KeyError:
            continue
        body = ledger.get(a)
        if any(body["id"] == y["id"] for y in out["assignments"] + out["verdicts"] + out["measurements"] + out["diagnoses"]):
            continue                      # several events (e.g. a checkpoint) may point at one artifact
        if kind == "assignment" and body["contract"] == contract_id:
            out["assignments"].append(body)
        elif kind == "verdict" and body["contract"] == contract_id:
            out["verdicts"].append(body)
        elif kind == "measurement" and body["contract"] == contract_id:
            out["measurements"].append(body)
        elif kind == "diagnosis" and body.get("target_artifact") == contract_id:
            out["diagnoses"].append(body)
    return out
