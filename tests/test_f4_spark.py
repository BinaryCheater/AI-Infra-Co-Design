"""F4 acceptance on the recorded Spark run (calibration/spark-6951/f4_run.json): real measurements for
the five accepted scripted templates, a kernel artifact per measurement, five real optimism entries,
and the smem-changing template marked extrapolating and measured by roofline."""
import json
from pathlib import Path

import pytest

F4 = Path(__file__).resolve().parents[1] / "calibration" / "spark-6951" / "f4_run.json"


def test_f4_spark_run():
    if not F4.exists():
        pytest.skip("no recorded Spark run")
    r = json.loads(F4.read_text())
    cuda = [m for m in r["measurements"] if m["simulator"].startswith("cuda:")]
    assert len(cuda) == 5 and all(m["kernel"] and m["reference_agrees"] is True and m["measured_cost"] > 0 for m in cuda)
    assert all(m["kernel"] in r["kernels"] for m in cuda) and len({m["kernel"] for m in cuda}) == 5
    assert all(abs(m["discrepancy"]) < 10 for m in cuda)          # a real number beside the prediction
    opt = r["optimism"]
    assert len([o for o in opt]) >= 6 and opt[-1]["samples"] >= 6
    assert len(r["extrapolating"]) == 1 and "smem_bytes_per_block" in r["extrapolating"][0]["fields"]
    roofline = [m for m in r["measurements"] if m["simulator"].startswith("analytic")]
    assert len(roofline) == 1                                    # the extrapolating template, modeled
    s = {e["iteration"]: e for e in r["story"]}
    assert sum("Measured on" in e["sentence"] for e in s.values()) == 5
    assert any("changes the chip" in e["sentence"] for e in s.values())
    assert [e["outcome"] for e in r["story"]] == ["accepted"] * 5 + ["rejected", "rejected", "accepted"]
