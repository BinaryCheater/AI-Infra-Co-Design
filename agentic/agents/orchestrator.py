"""The Orchestrator: a state machine, not a model (design §2.3, §2.4, §5).

It owns the refinement cursor and the run state, dispatches to the Formulator, Solver, Verifier and
Evaluator, diagnoses failures with the paper's table, and appends everything to the ledger. Every
agent step is a checkpoint. The architect's three modes (observe, step, drive) are the same code
path with different wait conditions, implemented by `Controller`.

Rules the architect cannot bypass: a human may edit or inject a template, but the Verifier still
has to certify it; an attempt to overrule a correctness verdict is refused and recorded.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

from ..chips import extrapolation
from ..ledger import Ledger, rebuild, search
from . import evaluator as evaluator_svc
from .evaluator import cuda as cuda_evaluator
from . import solver as solver_svc
from . import verifier as verifier_svc
from .evaluator import OptimismRecord, regime_of
from .formulator import well_formed

MODES = ("observe", "step", "drive")
STEP_CHECKPOINTS = ("proposed", "admitted")
CHECKPOINTS = ("proposed", "solved", "refuted", "regime_flip", "admitted")
DEFAULT_BREAKPOINTS = frozenset({"refuted", "regime_flip"})
DEFAULT_PARAMS = {"eps": 0.05, "B": 20, "delta": 0.15, "discrepancy_tol": 0.05}


class Rewind(Exception):
    def __init__(self, checkpoint: dict, fresh: bool):
        self.checkpoint, self.fresh = checkpoint, fresh


class Stop(Exception):
    pass


@dataclass
class Controller:
    """The architect's command interface, shared between the loop thread and the UI or CLI.

    pause / resume / step / set_mode change when the loop waits. edit / inject / override / rewind
    are queued and applied by the loop at the checkpoint where it is waiting, or at the next one.
    """
    mode: str = "observe"
    paused: bool = False
    # Breakpoints: checkpoint names that block in any mode (the console's "Stop at" toggles). The
    # console arms {"refuted", "regime_flip"} by default (DEFAULT_BREAKPOINTS); a headless Controller
    # starts with none so that `agentic run` and the tests' observe-mode runs never wait on nobody.
    breakpoints: set = field(default_factory=set)
    _go: int = 0
    _stop: bool = False
    _queue: deque = field(default_factory=deque)
    waiting: dict | None = None
    cv: threading.Condition = field(default_factory=threading.Condition)

    # --- what the loop calls
    def should_block(self, name: str) -> bool:
        if self.paused:
            return True
        if name in self.breakpoints:
            return True
        if self.mode == "observe":
            return False
        if self.mode == "step":
            return name in STEP_CHECKPOINTS
        return True

    def set_breakpoints(self, names):
        with self.cv:
            self.breakpoints = set(names)
            self.cv.notify_all()

    def wait_at(self, name: str, info: dict) -> tuple[list[dict], bool]:
        """Block at a checkpoint. Returns (commands, released): commands arrive while still waiting so
        the loop can apply them in place; released is True once the architect lets the loop go on."""
        with self.cv:
            if self.waiting is None or self.waiting.get("checkpoint") != name:
                self.waiting = {"checkpoint": name, **info}
            self.cv.notify_all()
            while self.should_block(name) and self._go == 0 and not self._stop and not self._queue:
                self.cv.wait(timeout=0.5)
            cmds = list(self._queue)
            self._queue.clear()
            if self._stop:
                self.waiting = None
                raise Stop()
            released = not self.should_block(name) or self._go > 0
            if released:
                if self._go > 0:
                    self._go -= 1
                self.waiting = None
            self.cv.notify_all()
            return cmds, released

    def update_pending(self, pending: str | None):
        with self.cv:
            if self.waiting is not None:
                self.waiting["pending"] = pending
                self.cv.notify_all()

    def drain(self) -> list[dict]:
        with self.cv:
            cmds = list(self._queue)
            self._queue.clear()
            return cmds

    # --- what the architect calls
    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError(mode)
        with self.cv:
            self.mode = mode
            self.cv.notify_all()

    def pause(self):
        with self.cv:
            self.paused = True

    def resume(self):
        with self.cv:
            self.paused = False
            self._go += 1
            self.cv.notify_all()

    step = resume

    def stop(self):
        with self.cv:
            self._stop = True
            self.cv.notify_all()

    def command(self, cmd: dict):
        with self.cv:
            self._queue.append(cmd)
            self.cv.notify_all()

    def edit(self, contract: dict, who: str = "architect", reason: str = ""):
        self.command({"op": "edit", "contract": contract, "who": who, "reason": reason})

    def inject(self, template_id: str, who: str = "architect", reason: str = ""):
        self.command({"op": "inject", "template": template_id, "who": who, "reason": reason})

    def override(self, target: str, decision: str, reason: str, who: str = "architect"):
        self.command({"op": "override", "target": target, "decision": decision, "reason": reason, "who": who})

    def rewind(self, checkpoint_seq: int, fresh: bool = False, who: str = "architect"):
        self.command({"op": "rewind", "to": checkpoint_seq, "fresh": fresh, "who": who})

    def snapshot(self) -> dict:
        with self.cv:
            return {"mode": self.mode, "paused": self.paused, "breakpoints": sorted(self.breakpoints),
                    "waiting": dict(self.waiting) if self.waiting is not None else None,
                    "queued": len(self._queue), "stopped": self._stop}


class Orchestrator:
    def __init__(self, ledger: Ledger, run: str, formulator, controller: Controller | None = None,
                 params: dict | None = None, evaluator_backend: str = "roofline", level: int = 4):
        self.ledger, self.run, self.formulator = ledger, run, formulator
        self.ctl = controller or Controller()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.backend = evaluator_backend
        self.optimism = OptimismRecord()
        self.cursor = {"step": 0, "iteration": 0, "level": level, "coarse_template": None}
        self.replay_from: dict | None = None     # set after a rewind
        self.overrides: dict = {}
        self.finished = False
        self.error: str | None = None

    # ------------------------------------------------------------------ helpers
    def _append(self, kind: str, payload: dict | None = None, artifact: str | None = None) -> dict:
        return self.ledger.append(self.run, kind, self.cursor["iteration"], payload, artifact)

    def _checkpoint(self, name: str, pending: str | None = None, apply=None) -> list[dict]:
        """Record a complete loop state, then wait as the mode requires. Commands that arrive while
        waiting are applied in place (through `apply`, which may return a new pending artifact id) so
        the architect sees their effect before letting the loop continue. Rewinds raise."""
        payload = {"name": name, "cursor": dict(self.cursor), "pending": pending, "mode": self.ctl.mode,
                   "ledger_seq": self.ledger.last_seq(self.run)}
        ev = self._append("checkpoint", payload, pending)
        info = {"seq": ev["seq"], "iteration": self.cursor["iteration"], "pending": pending, "step": self.cursor["step"]}
        applied = []
        while True:
            cmds, released = self.ctl.wait_at(name, info)
            cmds = self._apply_rewinds(cmds)
            if apply is not None:
                for c in cmds:
                    new_pending = apply(c)
                    if new_pending is not None:
                        info["pending"] = new_pending
                        self.ctl.update_pending(new_pending)
            else:
                self._apply_generic(cmds)
            applied += cmds
            if released:
                return applied

    def _apply_rewinds(self, cmds: list[dict]) -> list[dict]:
        rest = []
        for c in cmds:
            if c["op"] == "rewind":
                cp = next((e for e in self.ledger.checkpoints(self.run) if e["seq"] == c["to"]), None)
                if cp is None:
                    self._append("note", {"ignored": c, "reason": "no such checkpoint"})
                    continue
                raise Rewind(cp, c.get("fresh", False))
            rest.append(c)
        return rest

    def _diagnose(self, symptom: str, source: str, response: str, target: str | None, detail=None) -> str:
        d = self.ledger.put("diagnosis", {"symptom": symptom, "source": source, "response": response,
                                          "target_artifact": target, "detail": detail or {}})
        self._append("diagnosis", {"symptom": symptom, "source": source, "response": response}, d)
        return d

    def _evidence(self) -> list[dict]:
        """Most recent first: the last measurement as a bottleneck claim, the last counterexample."""
        ev = []
        events = self.ledger.events(self.run)
        for e in reversed(events):
            if e["kind"] == "measured" and not any(x["kind"] == "bottleneck" for x in ev):
                m = self.ledger.get(e["artifact"])
                ev.append({"kind": "bottleneck", "resource": m["dominant_resource"],
                           "regime": regime_of(m["dominant_resource"]),
                           "utilization": m["utilizations"][m["dominant_resource"]],
                           "utilizations": m["utilizations"], "artifact": m["id"]})
            if e["kind"] in ("family_check", "instance_check") and not any(x["kind"] == "counterexample" for x in ev):
                v = self.ledger.get(e["artifact"])
                if v["verdict"] == "refuted":
                    ev.append({"kind": "counterexample", "template": v["contract"], "check": v["check"],
                               "obligation": v["counterexample"]["obligation"], "artifact": v["id"]})
            if len(ev) == 2:
                break
        return ev

    # ------------------------------------------------------------------ the loop
    def run_loop(self):
        self._append("run_started", {"formulator": getattr(self.formulator, "name", "?"), "params": self.params,
                                     "evaluator": self.backend, "mode": self.ctl.mode})
        try:
            while True:
                try:
                    if self._iterate():
                        break
                except Rewind as r:
                    self._rewind(r)
        except Stop:
            self._append("note", {"stopped_by": "architect"})
        except Exception as e:          # record, then re-raise: the ledger must say why a run died
            self.error = f"{type(e).__name__}: {e}"
            self._append("note", {"error": self.error})
            self._append("run_finished", {"status": "error", "error": self.error})
            raise
        self.finished = True
        self._append("run_finished", {"status": "ok", "iterations": self.cursor["iteration"],
                                      "optimism": self.optimism.to_dict()})

    def _rewind(self, r: Rewind):
        cp = r.checkpoint
        frm = dict(self.cursor)
        self.cursor = dict(cp["payload"]["cursor"])
        self.cursor["iteration"] = frm["iteration"]        # the rewind closes the iteration it interrupts
        self.replay_from = {"checkpoint": cp["seq"], "step": self.cursor["step"], "fresh": r.fresh}
        self._append("rewind", {"to": cp["seq"], "checkpoint": cp["payload"]["name"], "from_cursor": frm,
                                "restored_cursor": dict(self.cursor), "fresh_proposal": r.fresh})

    def _propose(self, request: dict) -> dict | None:
        if self.replay_from and not self.replay_from["fresh"]:
            # replay the Formulator's recorded output for this step instead of calling it again
            for e in self.ledger.events(self.run, kinds=["proposed"]):
                if e["payload"].get("step") == request["step"] and e["seq"] <= self.replay_from["checkpoint"]:
                    T = self.ledger.get(e["artifact"])
                    self._append("replay", {"step": request["step"], "contract": T["id"],
                                            "from_checkpoint": self.replay_from["checkpoint"]})
                    return T
        return self.formulator.propose(request)

    def _iterate(self) -> bool:
        cur = self.cursor
        cur["iteration"] += 1
        views = rebuild(self.ledger, self.run)
        hits = [h["contract"]["id"] for h in search(views["population"])][-3:]
        request = {"step": cur["step"], "level": cur["level"], "coarse_template": cur["coarse_template"],
                   "evidence": self._evidence(), "population_hits": hits}
        t0 = perf_counter()
        T = self._propose(request)
        formulator_time = perf_counter() - t0
        if T is None:                       # the Formulator has nothing left to propose
            cur["iteration"] -= 1
            return True
        self._append("request", request)
        tid = self.ledger.put("contract", T, "tmpl")
        T = self.ledger.get(tid)
        self._append("proposed", {"step": cur["step"], "structure": T["structure"],
                                  "refines": T["provenance"].get("refines"), "workload": T["scope"]["workload"],
                                  "formulator_time": formulator_time}, tid)

        box = {"T": T, "tid": tid}

        def apply_proposed(c):
            box["T"], box["tid"] = self._apply_at_proposed(c, box["T"], box["tid"])
            return box["tid"]

        self._checkpoint("proposed", tid, apply_proposed)
        T, tid = box["T"], box["tid"]

        problems = well_formed(T)
        if problems:
            self._append("rejected", {"reason": "malformed", "problems": problems}, tid)
            cur["step"] += 1
            return False
        decision = "gate" if T["structure"]["strength"] == "proved" else "steer"
        self._append("mark", {"strength": T["structure"]["strength"], "assumptions_may": decision}, tid)
        # a template that changes a fixed field of a measured chip is extrapolating: costs are modeled for
        # this iteration whatever the run's backend, and its regime assumption may only steer (§4.2)
        extra = extrapolation(T["scope"]["chip"])
        backend = self.backend
        if extra:
            backend = "roofline"
            self._append("extrapolating", {"fields": extra, "measured_contract": T["scope"]["chip"].get("name"),
                                           "evaluator": backend, "strength_cap": "observed"}, tid)

        v = self.ledger.put("verdict", verifier_svc.certify_family(T, views["proofs"]), "cert")
        V = self.ledger.get(v)
        self._append("family_check", {"verdict": V["verdict"], "reused": bool(V["provenance"].get("reused_from")),
                                      "precondition_is_weakest": V["provenance"].get("precondition_is_weakest"),
                                      "checker_time": V["checker_time"]}, v)
        if V["verdict"] != "proved":
            cex = V["counterexample"] or {}
            self._diagnose(f"family certification {V['verdict']}: {cex.get('obligation', '?')}", "contract",
                           "repair the contract: the precondition admits a schedule the checker refutes; "
                           "strengthen Pre (and C3 with it, K1) using the replayed interleaving", tid,
                           {"verdict": v, "family_point": cex.get("design")})
            self._checkpoint("refuted", v)
            cur["step"] += 1
            return False

        x = self.ledger.put("assignment", solver_svc.solve(T), "asgn")
        X = self.ledger.get(x)
        self._append("solved", {"objective": X["objective"], "bound": X["bound"], "gap": X["gap"],
                                "solver_time": X["solver_time"], "values": X["values"]}, x)
        self._checkpoint("solved", x)

        w = self.ledger.put("verdict", verifier_svc.check_instance(T, X), "inst")
        W = self.ledger.get(w)
        self._append("instance_check", {"verdict": W["verdict"], "checker_time": W["checker_time"],
                                        "failed": [o["obligation"] for o in W["obligations"]
                                                   if o["status"] not in verifier_svc.ACCEPTED]}, w)
        if W["verdict"] != "proved":
            cex = W["counterexample"] or {}
            ob = cex.get("obligation", "?")
            if W["verdict"] == "open":
                self._diagnose(f"instance check open: {ob}", "contract",
                               "the checker did not resolve an obligation; raise the timeout or shrink the "
                               "event model before trusting this design", tid, {"verdict": w})
            elif ob.split("[")[0] in ("coverage", "exact_once", "no_phantom"):
                self._diagnose(f"instance check refuted: {ob}", "contract",
                               "a structural choice of the template admits a wrong index set; the replay shows "
                               "the missing, duplicated or phantom terms", tid, {"verdict": w})
            else:
                self._diagnose(f"instance check refuted: {ob}", "constraints",
                               "C admitted a design that violates Pre on this instance (K1 broken for this "
                               "point); add the violated predicate to C", tid, {"verdict": w})
            self._checkpoint("refuted", w)
            cur["step"] += 1
            return False

        t0 = perf_counter()
        if backend.startswith("cuda:"):
            kid = self.ledger.put("kernel", cuda_evaluator.kernel_artifact(T, X), "kern")
            m = self.ledger.put("measurement", cuda_evaluator.measure(T, X, backend[5:], self.ledger.get(kid)), "meas")
        else:
            m = self.ledger.put("measurement", evaluator_svc.measure(T, X, backend), "meas")
        evaluator_time = perf_counter() - t0
        M = self.ledger.get(m)
        if M.get("reference_agrees") is False:
            # not a performance obligation: a certified design computed the wrong answer on real hardware, so the
            # verifier's hardware contract is wrong somewhere. Record it against the trusted base and stop the run.
            self._append("measured", {"predicted": M["predicted_cost"], "measured": M["measured_cost"],
                                      "discrepancy": M["discrepancy"], "utilizations": M["utilizations"],
                                      "dominant_resource": M["dominant_resource"], "simulator": M["simulator"],
                                      "evaluator_time": evaluator_time, "reference_agrees": False}, m)
            self._diagnose("a certified design computed a wrong answer on real hardware "
                           f"(first disagreeing element {M['detail'].get('first_disagreeing_rc')})", "trusted_base",
                           "review the verifier's hardware assumptions (event model, async-copy ordering); the run is stopped",
                           tid, {"measurement": m, "kernel": M.get("kernel")})
            self._append("stopped", {"reason": "trusted_base", "measurement": m}, tid)
            return True
        self._append("measured", {"predicted": M["predicted_cost"], "measured": M["measured_cost"],
                                  "discrepancy": M["discrepancy"], "utilizations": M["utilizations"],
                                  "dominant_resource": M["dominant_resource"], "simulator": M["simulator"],
                                  "evaluator_time": evaluator_time, "reference_agrees": M.get("reference_agrees"),
                                  "ci95": M.get("ci95"), "host": M.get("host")}, m)
        was_clean = self.optimism.clean
        self.optimism.update(M["predicted_cost"], M["measured_cost"], m)
        self._append("optimism", {**self.optimism.to_dict(), "went_dirty": was_clean and not self.optimism.clean})
        if was_clean and not self.optimism.clean:
            self._diagnose("a measurement beat its prediction: the cost model is no longer one-sided",
                           "model", "demote every skip_regime decision to steer and reopen the skipped regimes",
                           m, {"dependents": self.optimism.dependents})

        strength = T["structure"]["strength"]
        assumed = T["structure"]["regime"].split(":")[0].strip()
        actual = regime_of(M["dominant_resource"])
        if assumed != actual:
            self._append("regime_flip", {"assumed": assumed, "actual": actual, "witness": x,
                                         "utilizations": M["utilizations"]}, m)
            self._checkpoint("regime_flip", m)          # the architect's stay / descend decision is taken here
            strength = self._handle_regime(T, X, M, views)
        elif strength == "observed":
            pr = evaluator_svc.probe(T, backend)
            if pr["regime_holds_as_bound"] and pr["regime"] == actual:
                strength = "proved"
            self._append("regime_bound", {"regime": pr["regime"], "holds_as_bound": pr["regime_holds_as_bound"],
                                          "feasible_designs": pr["feasible_designs"],
                                          "dominant_counts": pr["dominant_counts"], "strength_now": strength,
                                          "note": pr["note"]}, tid)

        if abs(M["discrepancy"]) > self.params["discrepancy_tol"]:
            self._diagnose(f"measured vs predicted differs by {M['discrepancy']:+.1%}", "cost_model",
                           "revise the cost model, not the template", m)

        if extra and strength == "proved":
            strength = "observed"            # extrapolating: a modeled bound may steer, not gate
        self._append("admitted", {"certificate": v, "assignment": x, "measurement": m, "strength": strength,
                                  "extrapolating": bool(extra), "evaluator": backend,
                                  "objective": X["objective"], "measured": M["measured_cost"]}, tid)
        cur["coarse_template"] = tid
        cur["step"] += 1
        self._checkpoint("admitted", tid)
        return False

    # ------------------------------------------------------- architect commands
    def _apply_at_proposed(self, c: dict, T: dict, tid: str):
        if c["op"] == "edit":
            new = dict(c["contract"])
            new.pop("id", None)
            new["status"] = "proposed"
            new["provenance"] = {**new.get("provenance", {}), "edited_by": c["who"], "derived_from": tid}
            try:
                nid = self.ledger.put("contract", new, "tmpl")
            except Exception as e:
                self._append("note", {"ignored": "edit", "reason": f"invalid contract: {e}"[:400]}, tid)
                return T, tid
            self._append("human_edit", {"from": tid, "to": nid, "who": c["who"], "reason": c["reason"]}, nid)
            return self.ledger.get(nid), nid
        if c["op"] == "inject":
            try:
                src = self.ledger.get(c["template"])
            except KeyError:
                self._append("note", {"ignored": "inject", "reason": f"no template {c['template']}"}, tid)
                return T, tid
            new = {k: v for k, v in src.items() if k not in ("id", "status", "certificate")}
            new["status"] = "proposed"
            new["parent"] = self.cursor["coarse_template"]
            new["provenance"] = {**new.get("provenance", {}), "edited_by": c["who"], "derived_from": src["id"]}
            nid = self.ledger.put("contract", new, "tmpl")
            self._append("human_inject", {"replaced": tid, "template": src["id"], "to": nid, "who": c["who"],
                                          "reason": c["reason"]}, nid)
            return self.ledger.get(nid), nid
        self._apply_generic([c])
        return T, tid

    def _apply_generic(self, cmds: list[dict]):
        for c in cmds:
            if c["op"] == "override":
                if c["target"] == "regime":
                    self.overrides["regime"] = c
                    self._append("human_override", {"target": "regime", "decision": c["decision"],
                                                    "reason": c["reason"], "who": c["who"]})
                else:
                    # every verdict the verifier issues is a correctness verdict: the gate applies to humans too
                    self._append("human_refused", {"target": c["target"], "decision": c["decision"],
                                                   "reason": c["reason"], "who": c["who"],
                                                   "rule": "correctness obligations gate; a human may change the "
                                                           "template, not the verdict"})
            elif c["op"] in ("edit", "inject"):
                self._append("note", {"ignored": c["op"], "reason": "only at the proposed checkpoint"})

    # ------------------------------------------------------------- regime flips
    def _handle_regime(self, T: dict, X: dict, M: dict, views: dict) -> str:
        """Bound, constraint, probe, budget (design §2.4). Returns the template's final strength."""
        tid, actual = T["id"], regime_of(M["dominant_resource"])
        pr = evaluator_svc.probe(T, self.backend if T["scope"]["chip"].get("kind") == "measured" else "roofline")
        if pr["regime_holds_as_bound"] and pr["regime"] == actual:
            self._append("regime_bound", {"regime": actual, "holds_as_bound": True, "strength_now": "proved",
                                          "feasible_designs": pr["feasible_designs"], "note": pr["note"]}, tid)
            return "proved"
        self._append("regime_bound", {"regime": actual, "holds_as_bound": False, "strength_now": "observed",
                                      "feasible_designs": pr["feasible_designs"],
                                      "dominant_counts": pr["dominant_counts"], "note": pr["note"]}, tid)

        # 2. the regime as a constraint, with a sibling one rung lower. v1 has a single template backend,
        #    so the sibling formulation is recorded as a request the Formulator cannot yet serve.
        next_resource = min(M["utilizations"], key=M["utilizations"].get)
        self._append("split", {"on": "regime", "in_regime": actual, "sibling_regime": regime_of(next_resource),
                               "children": [tid, None],
                               "note": "sibling template for the next resource is not available in v1 "
                                       "(single backend); recorded for the population's regime map"}, tid)

        # 3. bounded upside: the best any design in the other regime can predict vs the incumbent
        key = f"{T['scope']['workload']['name']}|{T['scope']['chip']['name']}|{T['cost'].get('objective', 'time')}"
        inc = views["incumbents"].get(key)
        incumbent = min(inc["cost"], X["objective"]) if inc else X["objective"]
        others = [d for d in _feasible_designs(T) if regime_of(d["dominant_resource"]) != actual]
        cap = min((d["time_us"] for d in others), default=float("inf"))
        human = self.overrides.pop("regime", None)
        if human:
            self._append("note", {"regime_decision": human["decision"], "by": human["who"]}, tid)
            return "observed"
        # The design note writes `cap <= incumbent * (1 + eps)`; the intended test is that the sibling's
        # ceiling is not usefully below the incumbent, so we skip when cap >= incumbent * (1 - eps).
        if self.optimism.clean and cap >= incumbent * (1 - self.params["eps"]):
            ev = self._append("skip_regime", {"reason": "bounded upside", "cap": cap, "incumbent": incumbent,
                                              "eps": self.params["eps"], "sibling_regime": regime_of(next_resource),
                                              "candidates": len(others)}, tid)
            self.optimism.depend(f"iteration {self.cursor['iteration']}: skip_regime on {tid}")
            return "observed"

        # 4. probe the sibling with a budget and a stated confidence (rule of three)
        wins = n = 0
        B, delta = self.params["B"], self.params["delta"]
        use_sim = self.backend == evaluator_svc.SIMULATOR and T["scope"]["chip"].get("kind") == "measured"
        for d in sorted(others, key=lambda d: d["time_us"])[:B]:
            n += 1
            if use_sim:                  # G5: the sibling's measurement is the simulator's time, labeled by its coverage rows
                from ..templates.smem_tiling import TilingDesign, Workload
                e = evaluator_svc.simulate(TilingDesign(**{k: v for k, v in d["design"].items() if k in TilingDesign.__dataclass_fields__}), Workload(**T["scope"]["workload"]), T["scope"]["chip"])
                md = e["time_us"]
                self._append("sibling_simulated", {"design": d["design"], "roofline_us": d["time_us"], "simulated_us": md, "dominant_resource": e["dominant_resource"], "coverage": e["coverage"]}, tid)
            else:
                md = d["time_us"]        # roofline backend: measurement is the model itself
            self.optimism.update(d["time_us"], md)
            if md < incumbent:
                wins += 1
            if wins == 0 and 3.0 / n < delta:
                break
        self._append("probe_result", {"n": n, "wins": wins, "bound_on_missed": (3.0 / n if n and wins == 0 else None),
                                      "budget": B, "delta": delta, "sibling_regime": regime_of(next_resource),
                                      "decision": "descend" if wins else "keep"}, tid)
        return "observed"


def _feasible_designs(T: dict) -> list[dict]:
    from itertools import product

    from ..chips import hardware_contract
    from ..templates.smem_tiling import (K_TILE_MENU, PREFETCH_MENU, STAGE_MENU, TILE_MENU, HardwareContract,
                                         TilingDesign, Workload, search_face_violations)
    from .evaluator import _evaluate
    w = Workload(**T["scope"]["workload"])
    hw = hardware_contract(T["scope"]["chip"])
    mode = next(p["mode"] for p in T["preconditions"] if p["id"] == "lifetime")
    s = T["scope"]["structure"]
    out = []
    for m, n, k, S, P, b in product(TILE_MENU, TILE_MENU, K_TILE_MENU, STAGE_MENU, PREFETCH_MENU, (False, True)):
        d = TilingDesign(m, n, k, S, P, b, leading_barrier=s["leading_barrier"], grid_rounding=s["grid_rounding"],
                         guard=s["guard"], stride_offset=s["stride_offset"])
        if search_face_violations(d, hw, mode):
            continue
        e = _evaluate(d, w, hw)
        out.append({"design": d.to_dict(), "time_us": e["time_us"], "dominant_resource": e["dominant_resource"]})
    return out
