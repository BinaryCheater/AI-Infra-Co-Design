"""F3 acceptance (notes/hardware_calibration_design.md §6): codegen renders the solved designs, and the
recorded runner results on spark-6951 show the T5 kernel agreeing with cuBLAS and the T7 floor-grid
kernel disagreeing in the tail."""
import json
from pathlib import Path

import pytest

from agentic.agents.formulator.scripted import ScriptedFormulator
from agentic.agents.orchestrator import Controller, Orchestrator
from agentic.ledger import Ledger, template_view
from agentic.ledger.describe import _dedupe
from agentic.ledger.story import short_names
from agentic.templates.smem_tiling.codegen import codegen

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "runs" / "demo.db"
F3 = ROOT / "calibration" / "spark-6951" / "f3_results.json"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if DEMO.exists() and "gemm-001" in Ledger(DEMO).runs():
        L = Ledger(DEMO)
    else:
        L = Ledger(tmp_path_factory.mktemp("codegen") / "demo.db")
        Orchestrator(L, "gemm-001", ScriptedFormulator(), Controller("observe")).run_loop()
    return L, "gemm-001", {v: k for k, v in short_names(L, "gemm-001").items()}


def _tv(L, run, cid):
    v = template_view(L, cid, run)
    return v["contract"], _dedupe(v["assignments"])[-1]


def test_f3_codegen_renders_the_program(demo):
    L, run, T = demo
    T5, x5 = _tv(L, run, T["T5"])
    src = codegen(T5, x5)
    for s in ("#define TM 128", "#define TN 128", "#define TK 32", "#define SLOTS 3", "#define PREFETCH 1",
              "#define TRAILING_BARRIER 0", "#define GRID_CEIL 1", "#define GUARDED 1", "cp.async", "__syncthreads"):
        assert s in src
    assert "{{" not in src and T5["id"] in src
    T7, x7 = _tv(L, run, T["T7"])
    assert "#define GRID_CEIL 0" in codegen(T7, x7)
    T4, x4 = _tv(L, run, T["T4"])
    assert "#define TRAILING_BARRIER 1" in codegen(T4, x4)


def test_f3_runner_results_on_spark():
    if not F3.exists():
        pytest.skip("no runner results recorded")
    r = json.loads(F3.read_text())["cases"]
    t5 = r["T5_exact"]
    assert t5["compiled"] and t5["ran"] and t5["reference_agrees"] is True and t5["median_us"] > 0
    t7 = r["T7_floor_grid"]
    assert t7["compiled"] and t7["ran"] and t7["reference_agrees"] is False
    row, col = t7["first_disagreeing_rc"]
    assert row >= 896 or col >= 896                     # the dropped tail: floor(1000/128) * 128 = 896
    u = r["unsound_S2_P1_beta0"]
    assert u["compiled"] and u["ran"] and isinstance(u["reference_agrees"], bool)   # a result, whichever way it went
