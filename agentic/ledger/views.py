"""The three shared stores, materialized from the ledger.

  population  certified contracts with scope, signature, evidence and lineage
  archive     assignments and decoded designs; the incumbent per (workload, chip, objective)
  proofs      certificates and the obligations they discharged (query hashes for reuse)

Nothing here is stored separately: `rebuild` folds the event sequence of one run (or of all runs)
into plain dicts. The UI reads these, the orchestrator reads these, and the test that the ledger is
complete is that these can be rebuilt from it alone.
"""
from __future__ import annotations

from .ledger import Ledger


def _design_key(contract: dict) -> str:
    w = contract["scope"]["workload"]
    c = contract["scope"]["chip"]
    return f"{w.get('name', w)}|{c.get('name', c)}|{contract['cost'].get('objective', 'time')}"


def rebuild(ledger: Ledger, run: str | None = None) -> dict:
    population: dict[str, dict] = {}
    archive: dict[str, dict] = {}
    incumbents: dict[str, dict] = {}
    proofs: dict[str, dict] = {}
    flips: list[dict] = []
    overrides: list[dict] = []

    for ev in ledger.events(run):
        k, p, a = ev["kind"], ev["payload"], ev["artifact"]
        if k == "family_check" and a:
            v = ledger.get(a)
            if v["verdict"] == "proved":
                proofs[v["id"]] = {
                    "certificate": v["id"], "contract": v["contract"], "run": ev["run"],
                    "family_key": v.get("provenance", {}).get("family_key"),
                    "obligations": [{"obligation": o["obligation"], "scope": o.get("scope", ""),
                                     "status": o["status"], "seconds": o.get("seconds", 0.0),
                                     "query_sha256": o.get("query_sha256")} for o in v["obligations"]],
                    "checker_time": v["checker_time"]}
        elif k == "solved" and a:
            x = ledger.get(a)
            archive[x["id"]] = {"assignment": x, "contract": x["contract"], "measurement": None,
                                "run": ev["run"], "iteration": ev["iteration"]}
        elif k == "measured" and a:
            m = ledger.get(a)
            entry = archive.get(m["assignment"])
            if entry is not None:
                entry["measurement"] = m
        elif k == "admitted" and a:
            T = ledger.get(a)
            x = ledger.get(p["assignment"]) if p.get("assignment") else None
            m = ledger.get(p["measurement"]) if p.get("measurement") else None
            population[T["id"]] = {
                "contract": T, "certificate": p.get("certificate"), "run": ev["run"],
                "iteration": ev["iteration"], "evidence": m, "assignment": x,
                "lineage": _lineage(population, T)}
            if x is not None:
                key = _design_key(T)
                cost = (m or {}).get("measured_cost", x["objective"])
                cur = incumbents.get(key)
                if cur is None or cost < cur["cost"]:
                    incumbents[key] = {"key": key, "cost": cost, "contract": T["id"], "assignment": x["id"],
                                       "design": x["values"], "iteration": ev["iteration"], "run": ev["run"]}
        elif k == "regime_flip":
            flips.append({"iteration": ev["iteration"], "run": ev["run"], **p})
        elif k in ("human_edit", "human_inject", "human_override", "human_refused", "rewind"):
            overrides.append({"iteration": ev["iteration"], "run": ev["run"], "kind": k, "seq": ev["seq"], **p})

    return {"population": population, "archive": archive, "incumbents": incumbents,
            "proofs": proofs, "flips": flips, "overrides": overrides}


def _lineage(population: dict, T: dict) -> list[str]:
    chain, cur = [], T.get("parent")
    while cur and cur in population and cur not in chain:
        chain.append(cur)
        cur = population[cur]["contract"].get("parent")
    if cur and cur not in chain:
        chain.append(cur)      # a parent outside the population (e.g. the coarse root)
    return chain


def search(population: dict, signature: str | None = None, scope: dict | None = None) -> list[dict]:
    """population.search(signature, scope): certified templates by structural signature, then scope."""
    hits = []
    for entry in population.values():
        T = entry["contract"]
        if signature and T.get("signature") != signature:
            continue
        if scope:
            chip = scope.get("chip")
            if chip and T["scope"]["chip"].get("name") != chip.get("name"):
                continue
        hits.append(entry)
    return sorted(hits, key=lambda e: e["iteration"])


def reuse(proofs: dict, query_sha256: str) -> dict | None:
    """proofs.reuse(obligation): an unsatisfiable query already discharged with this exact hash."""
    for cert in proofs.values():
        for o in cert["obligations"]:
            if o.get("query_sha256") == query_sha256:
                return {"certificate": cert["certificate"], **o}
    return None
