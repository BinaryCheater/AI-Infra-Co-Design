"""Historical evidence corruption is fatal; failed cases remain in the denominator."""

import hashlib
import json

import pytest

from codesign.ventus.historical_replay import replay


def evidence_tree(root):
    flow = root / "archive/analysis/ventus_flow_20261006"
    provenance = flow / "search_model/kernel-provenance-v3.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps(
            {
                "hashes": {
                    "${REMOTE_PROJECT_ROOT}/sim-verilator/build/driver_example/debug/sim-VentusRTL": "a"
                    * 64,
                    "${REMOTE_PROJECT_ROOT}/sim-verilator/build/libVentusRTL/debug/libVentusRTL.so": "b"
                    * 64,
                    "${REMOTE_FLOW_ROOT}/trace_rtl-v4.so": "c" * 64,
                }
            }
        )
    )
    for suite, source, observed in (
        ("quick_complete_20261006", "probes", "rtl"),
        ("performance-v7", "memory-probes", "memory-rf4-v7"),
        ("performance-v7", "combined-probes", "combined-rf4-v7"),
    ):
        folder = flow / suite / source
        folder.mkdir(parents=True)
        (folder / "pre_run_predictions.json").write_text(
            json.dumps({"cases": [{"name": "failed", "input_sha256": {}, "expected_words": [1]}]})
        )
        observed_dir = flow / suite / observed / "failed"
        observed_dir.mkdir(parents=True)
        (observed_dir / "summary.json").write_text(json.dumps({"returncode": 1, "warnings": []}))
        (observed_dir / "trace.csv").write_text("")
    entries = [
        {
            "path": str(p.relative_to(root)),
            "export_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in (root / "archive").rglob("*")
        if p.is_file()
    ]
    # A changed live predictor is expected in a new calibration, not frozen evidence.
    entries.append({"path": "codesign/ventus/live.py", "export_sha256": "0" * 64})
    (root / "archive/export-manifest.json").write_text(json.dumps({"files": entries}))
    return provenance


def test_failed_runs_retained_and_old_output_never_replaced(tmp_path):
    evidence_tree(tmp_path)
    output = tmp_path / "new"
    replay(output, root=tmp_path)
    result = json.loads((output / "audit.json").read_text())
    assert result["summary"]["cases"] == 6
    assert result["summary"]["open"] == 6
    assert result["summary"]["passes"] == 0
    assert {r["endpoint"] for r in result["cases"]} == {
        "first-collect-to-compute",
        "first-collect-to-visible",
    }
    assert all(r["producer_error"] == "RTL run diagnostic failure" for r in result["cases"])
    with pytest.raises(FileExistsError):
        replay(output, root=tmp_path)


def test_modified_historical_evidence_is_rejected(tmp_path):
    source = evidence_tree(tmp_path)
    source.write_text("{}")
    with pytest.raises(ValueError, match="Historical artifact changed"):
        replay(tmp_path / "new", root=tmp_path)


def test_archive_integrity_does_not_claim_current_code_integrity(tmp_path):
    from ventus_project.evidence import verify_export

    evidence_tree(tmp_path)
    result = verify_export(tmp_path, archive_only=True)
    assert result["export_integrity"]
    assert result["scope"] == "archive only; current executable code excluded"
    with pytest.raises(FileNotFoundError):
        verify_export(tmp_path)
