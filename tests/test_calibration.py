"""F1 acceptance (notes/hardware_calibration_design.md §6) on the committed calibration records.
Skipped when a host's raw records are not present."""
import json
from pathlib import Path

import pytest
import yaml

from agentic.schema import validate_chip

CAL = Path(__file__).resolve().parents[1] / "calibration"
HOSTS = [p.name for p in CAL.iterdir() if p.is_dir() and (p / "raw").is_dir()] if CAL.exists() else []
DATASHEET_BW = {"spark-6951": 273e9, "rtx5090": 1.8e12}
CI_TARGET = 0.04          # clocks cannot be locked on the Spark (preflight.md), so 4 % per §9.1


@pytest.mark.parametrize("host", HOSTS or ["none"])
def test_f1_raw_records_converged(host):
    if host == "none":
        pytest.skip("no calibration records in the repo")
    recs = [json.loads(f.read_text()) for f in (CAL / host / "raw").glob("*.json") if not f.name.startswith("_")]
    assert recs
    for r in recs:
        if r["quantity"] == "device_properties" or not r["samples"]:
            continue                               # declared or "recorded, not measured" entries carry no samples
        if r.get("repeat_policy") in ("once", "structural"):
            continue                               # B13+ structure probes: run once, or inferred counts/booleans rather than rates
        assert r["value"] is not None and r["n"] >= 3, r["quantity"]
        assert r["ci95"] <= CI_TARGET or r["n"] >= 50, (r["quantity"], r["ci95"], r["n"])
        assert all(k in r for k in ("unit", "method", "clocks", "timestamp"))


@pytest.mark.parametrize("host", HOSTS or ["none"])
def test_f1_contract_fits_and_validates(host):
    if host == "none" or not (CAL / host / "contract.yaml").exists():
        pytest.skip("no fitted contract")
    c = yaml.safe_load((CAL / host / "contract.yaml").read_text())
    validate_chip(c)
    assert c["kind"] == "measured" and c["provenance"]["host"] == host
    sheet = next((v for k, v in DATASHEET_BW.items() if host.startswith(k)), None)
    if sheet:
        assert 0.6 * sheet <= c["dram_bytes_per_s"] <= 1.0 * sheet, c["dram_bytes_per_s"]
    assert 0 < c["barrier_s"]["256"] < 1e-6
    assert c["fp32_flops_per_s_per_sm"] > 0 and c["sms"] > 0
    assert set(["sms", "smem_bytes_per_block", "dram_bytes_per_s"]) <= set(c["fixed"])


@pytest.mark.parametrize("host", HOSTS or ["none"])
def test_f2_full_suite_fields(host):
    if host == "none" or not (CAL / host / "contract.yaml").exists():
        pytest.skip("no fitted contract")
    c = yaml.safe_load((CAL / host / "contract.yaml").read_text())
    if not str(c.get("suite_level", "")).startswith("full"):
        pytest.skip("minimum suite only")
    assert "tma_supported" in c and isinstance(c["tma_supported"], bool)
    assert c["unified_memory_penalty"] and all(v > 0 for v in c["unified_memory_penalty"].values())
    assert c["clock_variance_sustained"] is not None and 0 <= c["clock_variance_sustained"] < 1
    assert c["tensor_flops_per_s_per_sm"]["bf16"] > c["fp32_flops_per_s_per_sm"]
    assert c["launch_s"] > 0 and c["async_copy_latency_s"]
    readme = (CAL / "suite" / "README.md").read_text()
    for b in ("B3", "B8", "B9", "B10", "B11", "B12"):
        assert f"| {b} |" in readme


@pytest.mark.parametrize("host", HOSTS or ["none"])
def test_b13_plus_micro_fields(host):
    """B13_plus.md: the structure suite's fields, their provenance, and the functional block."""
    if host == "none" or not (CAL / host / "contract.yaml").exists():
        pytest.skip("no fitted contract")
    c = yaml.safe_load((CAL / host / "contract.yaml").read_text())
    if "+micro" not in str(c.get("suite_level", "")):
        pytest.skip("structure suite not run")
    src = c["provenance"]["sources"]
    for k in ("l2_sector_bytes", "cp_async_groups_in_flight", "block_dispatch_order", "warps_to_hide_dram_latency", "clock_warmup_s", "dram_channels"):
        assert k in c, k
        assert src[k].get("method") and src[k].get("from"), k
    assert isinstance(c["functional"], dict) and "fma_contraction_default" in c["functional"]
    # B13 and B18 inferences carry the tables they were inferred from
    assert src["l2_ways"]["inferred_from"]["grid"] and src["dram_channels"]["inferred_from"]
    assert c["block_dispatch_order"] and all("class" in v for v in c["block_dispatch_order"].values())
    readme = (CAL / "suite" / "README.md").read_text()
    for b in ("B13", "B16", "B17", "B26"):
        assert f"| {b} |" in readme
