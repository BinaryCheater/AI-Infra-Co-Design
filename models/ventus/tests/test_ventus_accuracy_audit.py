"""Prevent calibration summaries from concealing failures or incompatible rankings."""

import pytest

from codesign.ventus.accuracy_audit import audit, main


def row(identifier="a", **changes):
    return {
        "id": identifier,
        "family": "gemm",
        "split": "holdout",
        "provenance": "fresh-rtl",
        "prediction_cycles": 100,
        "reference_cycles": 100,
        "completed": True,
        "functional_ok": True,
        "reference_sha256": "a" * 64,
        "model_version": "candidate",
        "endpoint": "first-collect-to-last-writeback",
        "ranking_group": "equal-work",
        "workload_sha256": "b" * 64,
        "target_sha256": "c" * 64,
        "initial_state_sha256": "d" * 64,
        **changes,
    }


def test_thresholds_missing_runs_and_provenance_are_not_averaged_away():
    report = audit(
        [
            row(prediction_cycles=105),
            row("fail", prediction_cycles=106),
            row("missing", reference_cycles=None, completed=False),
            row("generic", family="vector", prediction_cycles=110, provenance="historical-rtl"),
        ]
    )
    assert [r["status"] for r in report["cases"]] == ["pass", "fail", "open", "pass"]
    assert report["summary"]["cases"] == 4
    assert not report["summary"]["all_cases_meet_threshold"]
    assert report["by_provenance"]["fresh-rtl"]["passes"] == 1
    assert report["by_split"]["calibration"]["all_cases_meet_threshold"] is False


def test_ranking_catches_inversion_despite_small_per_case_error():
    report = audit(
        [row(prediction_cycles=101), row("b", prediction_cycles=100, reference_cycles=101)]
    )
    assert report["summary"]["all_cases_meet_threshold"]
    ranking = report["rankings"][0]
    assert ranking["pairwise_accuracy"] == 0
    assert ranking["worst_tied_winner_regret"] == pytest.approx(0.01)


def test_prediction_ties_and_reference_tolerance_are_explicit():
    rows = [row(), row("b", reference_cycles=101)]
    ranking = audit(rows)["rankings"][0]
    assert ranking["counts"]["prediction-tie"] == 1
    assert ranking["pairwise_accuracy"] == 0
    assert ranking["worst_tied_winner_regret"] == pytest.approx(0.01)
    ranking = audit(rows, reference_tie_fraction=0.02)["rankings"][0]
    assert ranking["counts"]["reference-tie"] == 1
    assert ranking["pairwise_accuracy"] is None


@pytest.mark.parametrize(
    "field", ["workload_sha256", "endpoint", "target_sha256", "initial_state_sha256"]
)
def test_incompatible_comparisons_are_open(field):
    report = audit([row(), row("b", **{field: "different"})])
    assert report["rankings"][0]["counts"]["open"] == 1
    assert report["rankings"][0]["worst_tied_winner_regret"] is None


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_measurements_cannot_pass(value):
    assert audit([row(reference_cycles=value)])["summary"]["open"] == 1


def test_duplicate_ids_and_output_replacement_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        audit([row(), row()])
    source = tmp_path / "rows.json"
    source.write_text("[]")
    destination = tmp_path / "receipt"
    main(["--input", str(source), "--out", str(destination)])
    with pytest.raises(FileExistsError):
        main(["--input", str(source), "--out", str(destination)])


@pytest.mark.parametrize("rows", [{"id": "a"}, [None], "rows"])
def test_input_shape_is_checked(rows):
    with pytest.raises(ValueError, match="list of case objects"):
        audit(rows)


@pytest.mark.parametrize("value", [None, [], 1])
def test_malformed_family_is_open_instead_of_crashing(value):
    assert audit([row(family=value)])["cases"][0]["status"] == "open"


def test_invalid_signature_hashes_do_not_establish_ranking_compatibility():
    rows = [row(workload_sha256=[1]), row("b", workload_sha256=[1])]
    ranking = audit(rows)["rankings"][0]
    assert not ranking["compatible"]
    assert ranking["status"] == "open"
    assert ranking["worst_tied_winner_regret"] is None


@pytest.mark.parametrize(
    "field,value", [("model_version", "other"), ("provenance", "silicon"), ("split", "calibration")]
)
def test_different_evaluation_domains_do_not_share_ranking(field, value):
    ranking = audit([row(), row("b", **{field: value})])["rankings"][0]
    assert ranking["status"] == "open"
    assert ranking["counts"]["open"] == 1


def test_no_orderable_pairs_cannot_pass_ranking():
    assert audit([row()])["rankings"][0]["status"] == "open"
    assert audit([row(), row("b")])["rankings"][0]["status"] == "open"


def test_reference_uncertainty_excludes_overlapping_pairs():
    rows = [
        row(prediction_cycles=101, reference_interval_cycles=[99, 102]),
        row("b", reference_cycles=101, reference_interval_cycles=[100, 103]),
    ]
    ranking = audit(rows)["rankings"][0]
    assert ranking["counts"]["reference-tie"] == 1
    assert ranking["status"] == "open"
    rows[1]["reference_interval_cycles"] = [102, 103]
    assert audit(rows)["cases"][1]["status"] == "open"  # Does not contain reference point.


def test_pass_is_explicitly_conditional_on_unverified_evidence():
    report = audit([row(), row("b", prediction_cycles=120, reference_cycles=120)])
    assert report["rankings"][0]["status"] == "pass"
    assert report["summary"]["valid_comparisons"] == 2
    assert report["summary"]["median_absolute_error_fraction"] == 0
    assert not report["evidence_validation"]["artifact_contents_verified"]
    assert not report["evidence_validation"]["completion_independently_verified"]
    assert not report["evidence_validation"]["holdout_independence_verified"]


def test_matching_cycles_cannot_hide_wrong_prediction_hardware():
    report = audit(
        [
            row(prediction_hardware={"rf_banks": 4}, reference_hardware={"rf_banks": 8}),
            row("valid", prediction_hardware={"rf_banks": 8}, reference_hardware={"rf_banks": 8}),
        ]
    )
    assert report["cases"][0]["status"] == "open"
    assert report["cases"][0]["reasons"] == ["prediction/reference hardware mismatch"]
    assert report["cases"][1]["status"] == "pass"
    assert report["rankings"][0]["counts"]["open"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"prediction_hardware": {"rf_banks": 4}},
        {"prediction_hardware": {}, "reference_hardware": {}},
        {"prediction_hardware": {"rf_banks": True}, "reference_hardware": {"rf_banks": 1}},
    ],
)
def test_incomplete_hardware_identity_is_open(changes):
    assert audit([row(**changes)])["summary"]["open"] == 1
