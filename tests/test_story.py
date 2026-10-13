"""C0 acceptance test (notes/architect_console_design.md §6) on the recorded demo ledger."""
from pathlib import Path

import pytest

from agentic.agents.formulator.scripted import ScriptedFormulator
from agentic.agents.orchestrator import Controller, Orchestrator
from agentic.ledger import Ledger
from agentic.ledger.story import short_names, story

DEMO = Path(__file__).resolve().parents[1] / "runs" / "demo.db"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if DEMO.exists() and "gemm-001" in Ledger(DEMO).runs():
        return Ledger(DEMO), "gemm-001"
    L = Ledger(tmp_path_factory.mktemp("story") / "demo.db")
    Orchestrator(L, "gemm-001", ScriptedFormulator(), Controller("observe")).run_loop()
    return L, "gemm-001"


def test_c0_story_on_demo(demo):
    L, run = demo
    s = story(L, run)
    assert len(s) == 7
    by = {e["iteration"]: e for e in s}
    assert "Rejected" in by[6]["sentence"] and "bug" in by[6]["sentence"]
    assert "bottleneck moved" in by[4]["sentence"] and "stayed" in by[4]["sentence"]
    assert "reused" in by[3]["sentence"] and "from T2" in by[3]["sentence"]
    assert all(len(e["sentence"].split()) < 80 for e in s), [len(e["sentence"].split()) for e in s]
    assert [e["outcome"] for e in s] == ["accepted"] * 5 + ["rejected", "rejected"]
    assert [v["station"] for v in by[4]["visits"]] == ["formulator", "verifier", "solver", "gate", "evaluator", "orchestrator"]
    assert by[6]["visits"][-1]["station"] == "verifier" and not by[6]["visits"][-1]["ok"]
    assert by[7]["visits"][3]["station"] == "gate" and not by[7]["visits"][3]["ok"]
    assert by[5]["pool_delta"]["incumbent"] and by[3]["pool_delta"]["reused_from"] == by[2]["template"]["id"]
    names = short_names(L, run)
    assert sorted(set(names.values())) == [f"T{i}" for i in range(1, 8)]
    assert all("tmpl_" not in e["sentence"] for e in s)
    assert "for the ragged 1000³ GEMM" in by[4]["sentence"]                       # D4: display names
    assert all("ragged_1000x1000x1000" not in e["sentence"] for e in s)


def _drive_with_breakpoints(tmp_path, breakpoints, on_wait):
    import threading, time
    L = Ledger(tmp_path / "bp.db")
    ctl = Controller("observe", breakpoints=set(breakpoints))
    o = Orchestrator(L, "bp", ScriptedFormulator(), ctl)
    t = threading.Thread(target=o.run_loop, daemon=True)
    t.start()
    waits, deadline = [], time.time() + 120
    while t.is_alive() and time.time() < deadline:
        w = ctl.snapshot()["waiting"]
        if w and (not waits or waits[-1] != (w["checkpoint"], w["iteration"])):
            waits.append((w["checkpoint"], w["iteration"]))
            on_wait(ctl, w)
            ctl.resume()
        time.sleep(0.02)
    assert not t.is_alive()
    return L, waits


def test_c4_default_breakpoints_pause_at_bug_and_bottleneck(tmp_path):
    L, waits = _drive_with_breakpoints(tmp_path, {"refuted", "regime_flip"}, lambda ctl, w: None)
    assert waits == [("regime_flip", 1), ("regime_flip", 4), ("refuted", 6), ("refuted", 7)]
    assert [e["outcome"] for e in story(L, "bp")] == ["accepted"] * 5 + ["rejected", "rejected"]


def test_c4_proposed_breakpoint_pauses_before_the_verifier(tmp_path):
    L, waits = _drive_with_breakpoints(tmp_path, {"proposed"}, lambda ctl, w: None)
    assert waits[0] == ("proposed", 1)
    first = L.events("bp", kinds=["checkpoint", "family_check"])[:2]
    assert [e["kind"] for e in first] == ["checkpoint", "family_check"]


def test_c5_edit_at_proposed_shows_in_story(tmp_path):
    def on_wait(ctl, w):
        if w["checkpoint"] == "proposed" and w["iteration"] == 1:
            T = L_holder[0].get(w["pending"])
            T["preconditions"][0]["mode"] = "exact"; T["constraints"][2]["expr"] = "P <= S - 2 + beta"
            T["preconditions"][0]["expr"] = "P <= S - 2 + beta"
            ctl.edit(T, reason="exact precondition")
            import time; time.sleep(0.3)
    L_holder = []
    import threading, time
    L = Ledger(tmp_path / "c5.db"); L_holder.append(L)
    ctl = Controller("observe", breakpoints={"proposed"})
    o = Orchestrator(L, "c5", ScriptedFormulator(), ctl)
    t = threading.Thread(target=o.run_loop, daemon=True); t.start()
    seen = []
    deadline = time.time() + 120
    while t.is_alive() and time.time() < deadline:
        w = ctl.snapshot()["waiting"]
        if w and (not seen or seen[-1] != w["seq"]):
            seen.append(w["seq"]); on_wait(ctl, w); ctl.resume()
        time.sleep(0.02)
    s = story(L, "c5")
    assert s[0]["sentence"].startswith("You changed the proposal")
    assert s[0]["human"] and s[0]["human"][0]["kind"] == "human_edit"
    assert s[0]["template"]["short"] == "T1" and s[0]["outcome"] == "accepted"
    assert L.get(s[0]["template"]["id"])["preconditions"][0]["mode"] == "exact"
    assert sorted(set(short_names(L, "c5").values())) == [f"T{i}" for i in range(1, 8)]
