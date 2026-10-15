"""G5: the simulator replaces the roofline in the evaluator and the regime probe, with every number labeled by the coverage
row it rests on. The roofline backend is unchanged (test_agentic's S4 test still holds)."""
from pathlib import Path

import pytest

from agentic.agents.evaluator import SIMULATOR, coverage_rows, measure, probe, rests_on, simulate
from agentic.agents.solver import solve
from agentic.chips import load_chip, scope_chip
from agentic.examples.gemm_tiling import tiling_contract
from agentic.schema import with_id
from agentic.templates.smem_tiling import TilingDesign, Workload

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "calibration" / "spark-6951" / "contract.yaml").exists(), reason="no measured contract")
SMALL = Workload("square_1024", 1024, 1024, 1024)


def _T():
    chip = scope_chip(load_chip("spark-6951-gb10"))
    return with_id("contract", tiling_contract(SMALL, hw=chip), "tmpl")


def test_g5_coverage_rows_parse():
    rows = coverage_rows("spark-6951-gb10")
    assert any(k.startswith("cp.async completion under load") for k in rows) and any(k.startswith("tensor") for k in rows)
    labels = rests_on("spark-6951-gb10", "copy")
    assert labels[0]["row"].startswith("cp.async completion under load") and "gap" in labels[0]["error"]


def test_g5_simulate_labels_every_number():
    chip = load_chip("spark-6951-gb10")
    e = simulate(TilingDesign(128, 128, 32, 3, 1, True), SMALL, chip)
    assert e["time_us"] > 0 and e["dominant_resource"] in ("compute", "copy", "barrier", "smem", "DRAM", "compute_queue")
    assert set(e["coverage"]) >= {"time_us", "dominant_resource", "dram_bytes"}
    for labels in e["coverage"].values():
        assert labels and all("row" in l and "error" in l for l in labels)


def test_g5_measure_and_probe_with_the_simulator():
    T = _T()
    x = with_id("assignment", solve(T), "asgn")
    m = measure(T, x, SIMULATOR)
    assert m["provenance"]["backend"] == SIMULATOR and m["coverage"]["time_us"] and m["measured_cost"] > 0
    assert m["measured_cost"] != m["predicted_cost"]                   # the simulator is not the roofline
    pr = probe(T, SIMULATOR, sample=3, budget_block_phases=20_000)
    assert pr["backend"] == SIMULATOR and pr["simulated_designs"] >= 1 and pr["regime_holds_as_bound"] is False
    assert pr["best_predicted"]["coverage"]["time_us"] and "observed" in pr["note"]
    r = probe(T)                                                        # the roofline path is unchanged
    assert r["backend"] == "roofline" and r["coverage"] is None and r["feasible_designs"] == pr["feasible_designs"]
