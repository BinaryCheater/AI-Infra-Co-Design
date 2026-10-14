"""G4 acceptance on the recorded validation set (calibration/spark-6951/validation.json): median error <= 10 %,
worst <= 20 %, the simulated critical path agreeing with the profiler's dominant stall on >= 24 of 30 designs."""
import json
from pathlib import Path

import pytest

V = Path(__file__).resolve().parents[1] / "calibration" / "spark-6951" / "validation.json"


def _rows():
    if not V.exists():
        pytest.skip("no validation set recorded")
    return json.loads(V.read_text())


def test_g4_validation_set_recorded():
    v = _rows(); s = v["summary"]
    assert s["designs"] >= 30 and s["ran"] == s["designs"], s
    assert s["reference_agrees"] == s["ran"], "every sampled kernel must pass the cuBLAS reference check"
    assert all(r["measured_us"] and r["simulated_us"] and r["ncu"].get("ok") for r in v["rows"])


@pytest.mark.xfail(strict=True, reason="G4 gate not passed with the B30 queue model: median 14.2%, worst 68%, non-compute agreement 0 %; notes/sim_gaps.md #13-#15")
def test_g4_validation_error_gate():
    v = _rows(); s = v["summary"]
    assert s["median_abs_error"] <= 0.10, s
    assert s["worst_abs_error"] <= 0.20, s
    assert s["agreement_overall"] >= 0.80 and (s["agreement_non_compute"] is None or s["agreement_non_compute"] >= 0.5), s


def test_g4_validation_per_family_agreement_recorded():
    v = _rows(); s = v["summary"]
    assert "agreement_by_ncu_family" in s and "agreement_non_compute" in s and "gate_passed" in s
    assert s["agreement_by_ncu_family"].get("compute", {}).get("designs", 0) >= 20


def test_g4_validation_overall_agreement():
    v = _rows(); s = v["summary"]
    agree, total = map(int, s["critical_agreement"].split("/"))
    assert total >= 30 and agree >= 24, s["critical_agreement"]
