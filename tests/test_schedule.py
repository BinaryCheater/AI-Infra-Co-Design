"""E0 acceptance test (notes/schedule_view_design.md §5) on the recorded demo ledger."""
from pathlib import Path

import pytest

from agentic.agents.formulator.scripted import ScriptedFormulator
from agentic.agents.orchestrator import Controller, Orchestrator
from agentic.ledger import Ledger
from agentic.ledger.schedule import schedule
from agentic.ledger.story import short_names

DEMO = Path(__file__).resolve().parents[1] / "runs" / "demo.db"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if DEMO.exists() and "gemm-001" in Ledger(DEMO).runs():
        L = Ledger(DEMO)
    else:
        L = Ledger(tmp_path_factory.mktemp("schedule") / "demo.db")
        Orchestrator(L, "gemm-001", ScriptedFormulator(), Controller("observe")).run_loop()
    return L, "gemm-001", {v: k for k, v in short_names(L, "gemm-001").items()}


def test_e0_t5_pipeline(demo):
    L, run, T = demo
    s = schedule(L, T["T5"], run)
    b = s["block"]
    assert s["axis"]["phases"] == 32 and s["design"]["S"] == 3 and len(s["lanes"]) == 3 + 3
    assert not [x for x in b["barriers"] if x["kind"] == "trailing"]
    assert len([x for x in b["barriers"] if x["kind"] == "leading"]) == 32
    load_end = {ld["tile"]: ld["t1"] for ld in b["loads"]}
    assert all(c["t0"] >= load_end[c["tile"]] for c in b["computes"])          # readiness by construction
    cm = s["resources"]
    compute_bars = sum(c["t1"] - c["t0"] for c in b["computes"])
    barrier_cost = s["axis"]["durations"]["barrier"] * len(b["barriers"])
    assert abs((compute_bars + barrier_cost) * s["axis"]["waves"] - cm["compute"]["s"]) < 1e-9
    # lifetime by construction: a slot is refilled only after its previous tile's last read
    by_slot = {}
    for ld in sorted(b["loads"], key=lambda x: x["t0"]):
        prev = by_slot.get(ld["slot"])
        if prev is not None:
            last_read = next(c for c in b["computes"] if c["tile"] == prev)["t1"]
            assert ld["t0"] >= last_read - 1e-12
        by_slot[ld["slot"]] = ld["tile"]
    assert s["resources"]["dominant"] == "compute" and s["grid"]["waves"] == 1 and s["grid"]["gm"] == 8
    assert s["proved"]["status"]["lifetime"] == "proved" and len(s["proved"]["readiness"]) == 32


def test_e0_t4_trailing_barriers_cost(demo):
    L, run, T = demo
    s4, s5 = schedule(L, T["T4"], run), schedule(L, T["T5"], run)
    assert len([x for x in s4["block"]["barriers"] if x["kind"] == "trailing"]) == 32
    extra = s4["block"]["block_time"] - s5["block"]["block_time"]
    assert extra > 0
    assert abs(extra - 32 * s4["axis"]["durations"]["barrier"]) < 1e-9 or extra >= 32 * s4["axis"]["durations"]["barrier"] - 1e-12


def test_e0_t7_uncovered_and_t6_counterexample(demo):
    L, run, T = demo
    s7 = schedule(L, T["T7"], run)
    assert s7["grid"]["uncovered"] and s7["grid"]["rounding"] == "floor"
    s6 = schedule(L, T["T6"], run)
    c = s6["counterexample"]
    assert c is not None and c["violation"]["obligation"] == "lifetime"
    assert len(c["warps"]) == 2 and len(c["violation"]["pair"]) == 2
    assert c["violation"]["pair"][0]["kind"] == "load" and c["violation"]["pair"][1]["kind"] == "compute"
    assert "slow warp" in c["words"]
    s1 = schedule(L, T["T1"], run)
    g = s1["grid"]                      # T1 chose 128 x 256 tiles: 32 x 16 = 512 blocks, 4 waves on 128 SMs
    assert g["blocks"] == g["gm"] * g["gn"] == 512 and g["waves"] == -(-g["blocks"] // g["sms"]) == 4
