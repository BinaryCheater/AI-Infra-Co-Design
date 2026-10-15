"""D0 acceptance test (notes/describing_templates.md §6) on the recorded demo ledger."""
from pathlib import Path

import pytest

from agentic.agents.formulator.scripted import ScriptedFormulator
from agentic.agents.orchestrator import Controller, Orchestrator
from agentic.ledger import Ledger
from agentic.ledger.describe import describe, describe_all, display_name
from agentic.ledger.story import short_names

DEMO = Path(__file__).resolve().parents[1] / "runs" / "demo.db"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if DEMO.exists() and "gemm-001" in Ledger(DEMO).runs():
        L = Ledger(DEMO)
    else:
        L = Ledger(tmp_path_factory.mktemp("describe") / "demo.db")
        Orchestrator(L, "gemm-001", ScriptedFormulator(), Controller("observe")).run_loop()
    names = short_names(L, "gemm-001")
    return L, "gemm-001", {v: k for k, v in names.items()}


def test_d0_headline_and_lenses(demo):
    L, run, T = demo
    d5 = describe(L, T["T5"], run)
    assert "ragged 1000³ GEMM" in d5["headline"] and "accepted" in d5["headline"] and "53.1 µs" in d5["headline"]
    d6 = describe(L, T["T6"], run)
    p = d6["proof"]["text"]
    assert "found a bug" in p and "next tile is loaded over it" in p and "2 buffers, loading 1 step ahead, barrier after each step dropped" in p
    s = d6["systems"]["text"]
    assert "slow thread" in s and "beta" not in s and "Pre" not in s and "K1" not in s and "slot" not in s
    d7 = describe(L, T["T7"], run)
    assert "dropped, so part of C is never computed" in d7["delta_from_parent"]
    for d in describe_all(L, run).values():
        for lens in ("systems", "optimization", "proof"):
            assert "ragged_1000x1000x1000" not in d[lens]["text"], (d["short"], lens)
        assert "ragged_1000x1000x1000" not in d["headline"] and "refines" not in d["headline"]


def test_d0_lens_contents(demo):
    L, run, T = demo
    d5 = describe(L, T["T5"], run)
    o = d5["optimization"]
    assert o["size"] == {"variables": 93, "binary": 90, "constraints": 248} and o["solver"]["gap"] == 0.0
    assert "picked the fastest" in o["text"] and "edge of the safety rule" in o["text"] and "C3" not in o["text"]
    spec = [i for s_ in o["sections"] if s_.get("details") for i in s_["details"]["items"]]
    assert any("93 variables" in i and "248 constraints" in i for i in spec)
    assert "loosest safe one" in d5["proof"]["text"] or "reused" in d5["proof"]["text"]
    assert d5["proof"]["instance"]["proved"] == 12 and d5["is_best"]
    assert d5["systems"]["design_table"][0]["name"] == "Tm" and d5["systems"]["parent_values"] is not None
    assert "faster than T4" in d5["systems"]["text"] and "barrier after each step could be dropped" in d5["systems"]["text"]
    assert d5["delta_from_parent"].startswith("Same as T4, but the safety rule is exact")
    d6 = describe(L, T["T6"], run)
    assert d6["status"] == "rejected" and "before any design" in d6["headline"]
    assert "solver did not run" in d6["optimization"]["text"] and d6["optimization"]["solver"] is None
    assert d6["proof"]["counterexample"]["replay"]["trace"]
    d1 = describe(L, T["T1"], run)
    assert d1["delta_from_parent"].startswith("First template of the run")


def test_display_names_fall_back():
    assert display_name("square_4096") == "the 4096³ GEMM"
    assert display_name("skinny_64x4096x4096") == "the skinny 64×4096×4096 GEMM"
    assert display_name("tall_2048x64x512") == "the tall 2048×64×512 GEMM"
    assert display_name("something_new") == "something new"
    assert display_name("illustrative-fp32-gpu", "chip").startswith("the illustrative FP32 GPU")
    assert display_name("other-chip", "chip") == "other-chip"


def test_plain_words_in_every_tab(demo):
    """Every tab reads as plain sentences: no internal shorthand in the text people see (formal terms stay in the details)."""
    L, run, T = demo
    banned = ("K1", "C3", "Pre", "obligation", "precondition", "dual bound", "HiGHS", "family", "instance", "∈", "MILP",
              "mixed-integer", "prefetch", "slot", "warp", "β")
    for d in describe_all(L, run).values():
        for lens in ("systems", "optimization", "proof"):
            for sec in d[lens]["sections"]:
                for line in sec["paras"] + sec["items"]:
                    for b in banned:
                        assert b not in line, (d["short"], lens, sec["title"], b, line)
