"""The chip view's trace (agentic/sim/microtrace.py): recording changes nothing the simulator computes, and the files
carry what the view draws."""
import json

from agentic.sim.config import load
from agentic.sim.microtrace import PhaseRecorder, build, compute_regime
from agentic.sim.warp import WarpModel
from agentic.templates.smem_tiling import TilingDesign


def test_trace_phase_matches_phase():
    cfg = load("spark-6951-gb10"); m = WarpModel(cfg["contract"], cfg["structural"])
    args = (8, 1, 32, 64, 8, 2, 0, 1, 1, (2, 2, 2, 2, 2, 2, 2, 2), (1,) * 8)
    cycles, hist = m.phase(*args)
    rec = PhaseRecorder(8)
    c2, h2 = m.trace_phase(rec, *args)
    assert (c2, h2) == (cycles, hist)
    assert len(rec.issues[0]) == hist["issued"]
    assert m._rec is None


def test_t5_files(tmp_path):
    out = build("ragged_1000x1000x1000", TilingDesign(128, 128, 32, 3, 1, False), out=tmp_path)
    s = json.loads((out / "summary.json").read_text())
    assert len(s["strip"]) == 48 and s["default_sm"] in s["sm_files"] or str(s["default_sm"]) in s["sm_files"]
    t = json.loads((out / "templates.json").read_text())
    assert all(v["regime"] == "scheduler-bound" for v in t.values())
    d = json.loads((out / f"sm{s['default_sm']:02d}.json").read_text())
    assert len(d["phases"]) == 32 and d["phases"][0]["template"] in t
    kinds = {e["kind"] for e in d["events"]}
    assert {"block_start", "copy_issue", "copy_done", "wait", "barrier_arrive", "barrier_release", "phase_compute"} <= kinds


def test_regime_rule():
    assert compute_regime({"issued": 10, "not_selected": 5, "scoreboard": 12, "lsu": 0})[0] == "scheduler-bound"
    assert compute_regime({"issued": 2, "not_selected": 0, "scoreboard": 70, "lsu": 5})[0] == "load-bound"
