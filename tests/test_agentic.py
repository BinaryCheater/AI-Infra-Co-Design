"""Stage tests from notes/multi_agent_design.md §7, S0–S3 and S6–S7 (headless), on the GEMM tiling fixture."""
import json
import threading
import time
from pathlib import Path

import pytest

from agentic.agents.evaluator import OptimismRecord, measure, probe
from agentic.agents.formulator.scripted import RAGGED, ScriptedFormulator
from agentic.agents.orchestrator import Controller, Orchestrator
from agentic.agents.solver import solve
from agentic.agents.verifier import certify_family, check_instance
from agentic.examples.gemm_tiling import fixture, tiling_contract
from agentic.ledger import Ledger, rebuild, replay_signature, template_view, timeline
from agentic.schema import validate, with_id

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def run_ledger(tmp_path_factory):
    L = Ledger(tmp_path_factory.mktemp("ledger") / "run.db")
    Orchestrator(L, "t", ScriptedFormulator(), Controller("observe")).run_loop()
    return L


# ----------------------------------------------------------------------------- S0
def test_s0_fixture_validates_and_matches_checked_in_json():
    T = with_id("contract", fixture(), "tmpl")
    validate("contract", T)
    on_disk = json.loads((HERE.parent / "agentic/examples/gemm_tiling/contract.json").read_text())
    assert on_disk == T, "regenerate contract.json with scripts/write_fixture.py"


def test_s0_stores_rebuild_from_ledger_alone(run_ledger):
    v = rebuild(run_ledger, "t")
    assert len(v["population"]) == 5                      # five admitted templates
    assert set(k.split("|")[0] for k in v["incumbents"]) == {"square_4096", "skinny_64x4096x4096",
                                                              "ragged_1000x1000x1000"}
    assert len(v["proofs"]) >= 3 and len(v["flips"]) == 2
    for entry in v["population"].values():
        assert entry["certificate"] and entry["evidence"] is not None


# ----------------------------------------------------------------------------- S1
def test_s1_family_certification_is_exact_on_24_points_and_refutes_unsound():
    T = with_id("contract", tiling_contract(RAGGED, precondition="exact"), "tmpl")
    v = certify_family(T)
    assert v["verdict"] == "proved" and v["provenance"]["precondition_is_weakest"] is True
    assert len(v["provenance"]["checker_admits"]) == 9     # the checker-safe schedule points of the 24
    U = with_id("contract", tiling_contract(RAGGED, precondition="unsound"), "tmpl")
    u = certify_family(U)
    assert u["verdict"] == "refuted" and u["counterexample"]["obligation"] == "lifetime"
    assert u["counterexample"]["replay"]["admissible"] and not u["counterexample"]["replay"]["agrees_with_reference"]
    assert any("STALE" in t for t in u["counterexample"]["replay"]["trace"])


def test_s1_instance_gate_rejects_floor_grid_with_replayed_coverage_defect():
    T = with_id("contract", tiling_contract(RAGGED, structural={"grid_rounding": "floor"}), "tmpl")
    x = with_id("assignment", solve(T), "asgn")
    w = check_instance(T, x)
    assert w["verdict"] == "refuted" and w["counterexample"]["obligation"].startswith("coverage")
    assert w["counterexample"]["replay"]["differences_total"] > 0


# ----------------------------------------------------------------------------- S2
def test_s2_solver_reproduces_paper_milp():
    T = with_id("contract", tiling_contract(RAGGED), "tmpl")
    x = solve(T)
    assert x["diagnostics"]["variables"] == 93 and x["diagnostics"]["binary_variables"] == 90
    assert x["diagnostics"]["constraints"] == 248
    assert x["values"] == {"Tm": 128, "Tn": 128, "Tk": 32, "S": 3, "P": 1, "beta": 0}
    assert x["gap"] == 0.0 and x["solver_time"] < 1.0 and abs(x["objective"] - 53.14) < 0.05


# ----------------------------------------------------------------------------- S3
def test_s3_scripted_loop_covers_every_path(run_ledger):
    rows = timeline(run_ledger, "t")
    outcomes = [r["outcome"] for r in rows]
    assert outcomes == ["admitted"] * 5 + ["refuted at family certification", "refuted at instance gate"]
    assert rows[3]["objective"] > rows[4]["objective"]          # exact precondition beats conservative
    assert abs(rows[3]["objective"] / rows[4]["objective"] - 1.0134) < 0.002   # the paper's 1.34%
    assert [r["flip"] for r in rows[:5]] == [True, False, False, True, False]
    assert rows[0]["strength_final"] == "proved"               # regime proved as a bound over the menu
    fam = [e["payload"]["reused"] for e in run_ledger.events("t", kinds=["family_check"])]
    assert fam == [False, False, True, True, True, False, False]   # certificate reuse
    assert all(r["family_verdict"] for r in rows)


def test_s3_replay_is_identical_across_processes(tmp_path):
    import subprocess, sys
    db = tmp_path / "r.db"
    cli = [sys.executable, "-m", "agentic.cli.main"]
    env = {"PYTHONPATH": str(HERE.parent), "PATH": "/usr/bin:/bin"}
    subprocess.run(cli + ["run", "--ledger", str(db), "--run", "a"], check=True, env=env, capture_output=True)
    out = subprocess.run(cli + ["replay", "--ledger", str(db), "--run", "a", "--as", "b"], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0 and "IDENTICAL" in out.stdout, out.stdout + out.stderr
    L = Ledger(db)
    sig = lambda run: replay_signature([e for e in L.events(run) if e["kind"] != "checkpoint"])
    assert sig("a") == sig("b") and len(sig("a")) > 50


# ----------------------------------------------------------------------------- S4 (analytic backend)
def test_s4_roofline_regime_and_optimism():
    T = with_id("contract", tiling_contract(RAGGED), "tmpl")
    x = with_id("assignment", solve(T), "asgn")
    m = measure(T, x)
    assert m["dominant_resource"] in ("compute", "DRAM") and abs(m["discrepancy"]) < 1e-9
    pr = probe(T)
    assert pr["feasible_designs"] == 722                       # the paper's brute-force count
    o = OptimismRecord()
    assert o.update(10.0, 10.0) and o.clean
    assert not o.update(10.0, 9.0) and not o.clean


# ----------------------------------------------------------------------------- S6 / S7 (headless)
def test_s6_template_view_shows_counterexample_from_timeline_alone(run_ledger):
    rows = timeline(run_ledger, "t")
    wrong = next(r for r in rows if r["flip"])                 # the iteration whose regime assumption was wrong
    v = template_view(run_ledger, wrong["contract"], "t")
    assert any(e["kind"] == "regime_flip" for e in run_ledger.events("t")) and v["admitted"]
    refuted = next(r for r in rows if r["instance_verdict"] == "refuted")
    v = template_view(run_ledger, refuted["contract"], "t")
    cex = [w for w in v["verdicts"] if w["counterexample"]][0]["counterexample"]
    assert cex["replay"]["differences_sample"] and v["diagnoses"][0]["source"] == "contract"


def _drive(L, run, actions, mode="step"):
    """Run the loop in a thread; `actions(ctl, waiting)` is called at every wait and may issue commands."""
    ctl = Controller(mode)
    o = Orchestrator(L, run, ScriptedFormulator(), ctl)
    t = threading.Thread(target=o.run_loop, daemon=True)
    t.start()
    seen = []
    deadline = time.time() + 120
    while t.is_alive() and time.time() < deadline:
        w = ctl.snapshot()["waiting"]
        if w and (not seen or seen[-1] != w["seq"]):
            seen.append(w["seq"])
            actions(ctl, w)
        elif w:
            ctl.resume()
        time.sleep(0.02)
    assert not t.is_alive(), "loop did not finish"
    return o, seen


def test_s7_edit_applies_in_place_and_is_recorded(tmp_path):
    L = Ledger(tmp_path / "e.db")
    done = {}

    def actions(ctl, w):
        if w["checkpoint"] == "proposed" and w["iteration"] == 1 and not done:
            T = L.get(w["pending"]); T["rationale"] = "ARCHITECT: " + T["rationale"]
            ctl.edit(T, reason="test")
            for _ in range(200):                          # the loop must still be waiting, on the new pending
                time.sleep(0.01)
                nw = ctl.snapshot()["waiting"]
                if nw and nw["pending"] != w["pending"]:
                    break
            nw = ctl.snapshot()["waiting"]
            assert nw and nw["checkpoint"] == "proposed" and nw["pending"] != w["pending"]
            done["new"] = nw["pending"]
        ctl.resume()

    _drive(L, "e", actions)
    ev = L.events("e", kinds=["human_edit"])
    assert len(ev) == 1 and ev[0]["payload"]["to"] == done["new"]
    assert L.get(done["new"])["provenance"]["edited_by"] == "architect"
    assert timeline(L, "e")[0]["contract"] == done["new"] and timeline(L, "e")[0]["outcome"] == "admitted"


def test_s7_correctness_override_is_refused_but_regime_override_recorded(tmp_path):
    L = Ledger(tmp_path / "o.db")

    def actions(ctl, w):
        if w["checkpoint"] == "solved" and w["iteration"] == 1:
            ctl.override("verdict", "accept", "bypass attempt")
            ctl.override("regime", "keep", "architect knows the shape")
        ctl.resume()

    _drive(L, "o", actions, mode="drive")
    assert len(L.events("o", kinds=["human_refused"])) == 1
    assert len(L.events("o", kinds=["human_override"])) == 1
    assert rebuild(L, "o")["overrides"][0]["kind"] in ("human_refused", "human_override")


def test_s7_rewind_replays_recorded_proposal_and_keeps_both_lineages(tmp_path):
    L = Ledger(tmp_path / "w.db")
    state = {"rewound": False}

    def actions(ctl, w):
        if w["checkpoint"] == "proposed" and w["iteration"] == 3 and not state["rewound"]:
            cp = [c for c in L.checkpoints("w") if c["payload"]["name"] == "proposed" and c["iteration"] == 2][0]
            state["rewound"] = True
            ctl.rewind(cp["seq"])
        ctl.resume()

    _drive(L, "w", actions)
    rows = timeline(L, "w")
    assert rows[2]["outcome"].startswith("rewound") and rows[3]["replayed"]
    assert rows[3]["contract"] == rows[1]["contract"]           # same proposal replayed from the ledger
    assert len(L.events("w", kinds=["replay"])) == 1
    assert [r["outcome"] for r in rows[4:]].count("admitted") >= 3
    assert len(rebuild(L, "w")["population"]) == 5
