"""Verifier service: certify_family(T) and check_instance(T, x). Deterministic; reads the
correctness face only. Wraps the bounded checker of the RQ1 prototype (Z3 + independent replay).

certify_family  the K-conditions once per template: K1 (C implies Pre, by construction here, since
                C3 and the lifetime precondition are the same predicate) and the family soundness
                sweep: every schedule point (S, P, beta) the precondition admits must pass the
                schedule obligations on the event model. A family certificate is reused when a proved
                one with the same family key is already in the proof library.
check_instance  the per-design gate: index obligations (all extents) and schedule obligations (all
                interleavings) for the selected design, each SAT result replayed independently.
"""
from __future__ import annotations

import hashlib
import json
from time import perf_counter

from ...templates.smem_tiling import (PRECONDITIONS, EventModel, TilingDesign, index_obligations,
                                      schedule_obligations)

# Determinism note: verdicts (proved / refuted / open) are reproducible. The *witness* Z3 picks for a
# refuted obligation is reproducible across fresh processes (random_seed = 0) but can differ between
# calls inside one process, so a counterexample artifact may get a different id on an in-process
# rewind. Replay identity (tests, `agentic replay`) is therefore checked across processes.

FAMILY_SWEEP = [(s, p, tb) for s in (1, 2, 3) for p in (0, 1, 2, 3) for tb in (False, True)]
ACCEPTED = ("proved", "agrees")


def decode(T: dict, values: dict) -> TilingDesign:
    s = T["scope"]["structure"]
    return TilingDesign(values["Tm"], values["Tn"], values["Tk"], values["S"], values["P"],
                        bool(values["beta"]), leading_barrier=s["leading_barrier"],
                        grid_rounding=s["grid_rounding"], guard=s["guard"], stride_offset=s["stride_offset"])


def family_key(T: dict) -> str:
    mode = next(p["mode"] for p in T["preconditions"] if p["id"] == "lifetime")
    body = {"template": T["template"], "mode": mode, "structure": T["scope"]["structure"],
            "event_model": EventModel().__dict__, "program": T["program"]}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _outcome(o, extra=None) -> dict:
    d = {"obligation": o.obligation, "scope": o.scope, "status": o.status, "seconds": o.seconds,
         "query_sha256": o.query_sha256, "reused": False}
    if extra:
        d.update(extra)
    return d


def _counterexample(o, design: TilingDesign) -> dict:
    return {"obligation": o.obligation, "design": design.to_dict(), "model": o.model, "replay": o.replay,
            "status": o.status}


def certify_family(T: dict, proofs: dict | None = None) -> dict:
    t0 = perf_counter()
    key = family_key(T)
    mode = next(p["mode"] for p in T["preconditions"] if p["id"] == "lifetime")
    _, pred = PRECONDITIONS[mode]
    obligations = [{"obligation": "K1: C => Pre", "scope": "C3 and Pre.lifetime are the same predicate",
                    "status": "proved (by construction)", "seconds": 0.0, "query_sha256": key[:16], "reused": False}]
    # incremental reuse: a proved family certificate with this key
    for cert in (proofs or {}).values():
        if cert.get("family_key") == key:
            return {"contract": T["id"], "assignment": None, "check": "family", "verdict": "proved",
                    "obligations": obligations + [{**o, "reused": True} for o in cert["obligations"]
                                                  if not o["obligation"].startswith("K1")],
                    "counterexample": None, "certificate": cert["certificate"],
                    "checker_time": perf_counter() - t0,
                    "provenance": {"service": "agentic.agents.verifier", "family_key": key,
                                   "reused_from": cert["certificate"]}}
    s = T["scope"]["structure"]
    counterexample = None
    admitted_by_checker = set()
    for S, P, tb in FAMILY_SWEEP:
        d = TilingDesign(16, 16, 8, S, P, tb, leading_barrier=s["leading_barrier"], grid_rounding=s["grid_rounding"],
                         guard=s["guard"], stride_offset=s["stride_offset"])
        claimed = bool(pred(S, P, tb))
        # readiness and lifetime decide safety (unsat over all interleavings). The checker's extra
        # "witness run" replays one SAT interleaving and, on an already unsafe point, may or may not hit
        # the race depending on the model Z3 picks; it is a per-design sanity check and is recorded by
        # check_instance, not here, so that a family certificate is deterministic.
        outs = [o for o in schedule_obligations(d) if not o.obligation.startswith("terminal_agreement")]
        safe = all(o.status in ACCEPTED for o in outs)
        if safe:
            admitted_by_checker.add((S, P, tb))
        for o in outs:
            obligations.append(_outcome(o, {"family_point": {"S": S, "P": P, "beta": int(tb)},
                                            "claimed_by_precondition": claimed}))
        if claimed and not safe and counterexample is None:
            bad = next(o for o in outs if o.status not in ACCEPTED)
            counterexample = _counterexample(bad, d)
    verdict = "refuted" if counterexample else "proved"
    claimed_set = {pt for pt in FAMILY_SWEEP if pred(*pt)}
    exact = claimed_set == admitted_by_checker
    out = {"contract": T["id"], "assignment": None, "check": "family", "verdict": verdict,
           "obligations": obligations, "counterexample": counterexample, "certificate": None,
           "checker_time": perf_counter() - t0,
           "provenance": {"service": "agentic.agents.verifier", "family_key": key,
                          "precondition_mode": mode, "precondition_is_weakest": exact,
                          "checker_admits": sorted([list(p) for p in admitted_by_checker]),
                          "precondition_admits": sorted([list(p) for p in claimed_set])}}
    return out


def check_instance(T: dict, x: dict) -> dict:
    t0 = perf_counter()
    d = decode(T, x["values"])
    outs = index_obligations(d) + schedule_obligations(d)
    obligations = [_outcome(o) for o in outs]
    bad = [o for o in outs if o.status not in ACCEPTED]
    if not bad:
        verdict, cex = "proved", None
    elif any(o.status.startswith("unresolved") for o in bad) and not any(
            o.status.startswith("counterexample") or o.status == "disagrees" for o in bad):
        verdict, cex = "open", _counterexample(bad[0], d)
    else:
        first = next(o for o in bad if o.status.startswith("counterexample") or o.status == "disagrees")
        verdict, cex = "refuted", _counterexample(first, d)
    return {"contract": T["id"], "assignment": x["id"], "check": "instance", "verdict": verdict,
            "obligations": obligations, "counterexample": cex, "certificate": None,
            "checker_time": perf_counter() - t0,
            "provenance": {"service": "agentic.agents.verifier", "design": d.to_dict()}}
