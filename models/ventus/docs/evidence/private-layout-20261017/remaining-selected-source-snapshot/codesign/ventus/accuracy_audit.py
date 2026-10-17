"""Independent, endpoint-aware audit of performance predictions against RTL.

This module does not execute or tune a predictor. Missing/failed reference runs
remain in the coverage denominator. Ranking groups are declared by the experiment
author and must identify equal logical work and compatible measurement endpoints.
"""

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import median


def _positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _sha256(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _reference_interval(row):
    interval = row.get("reference_interval_cycles")
    if interval is None:
        return (row["reference_cycles"], row["reference_cycles"])
    return tuple(interval)


def audit(rows, *, reference_tie_fraction=0.0):
    """Return strict per-case thresholds and pairwise ordering, never a fitted score."""
    if (
        isinstance(reference_tie_fraction, bool)
        or not isinstance(reference_tie_fraction, (int, float))
        or not 0 <= reference_tie_fraction < 1
    ):
        raise ValueError("reference_tie_fraction must be in [0, 1)")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("input must be a JSON list of case objects")
    ids = set()
    evaluated = []
    groups = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("every case needs a unique nonempty id")
        ids.add(identifier)
        family = row.get("family")
        threshold = 0.05 if isinstance(family, str) and family.lower() == "gemm" else 0.10
        reasons = []
        for key in ("prediction_cycles", "reference_cycles"):
            if not _positive(row.get(key)):
                reasons.append(f"invalid or missing {key}")
        for key in ("completed", "functional_ok"):
            if row.get(key) is not True:
                reasons.append(f"{key} not established")
        for key in ("model_version", "endpoint", "family"):
            if not isinstance(row.get(key), str) or not row[key]:
                reasons.append(f"missing {key}")
        if not _sha256(row.get("reference_sha256")):
            reasons.append("missing or invalid reference_sha256")
        hardware_fields = ("prediction_hardware", "reference_hardware")
        hardware_declared = any(key in row for key in hardware_fields)
        if hardware_declared:
            configurations = [row.get(key) for key in hardware_fields]
            valid_hardware = all(
                isinstance(config, dict)
                and bool(config)
                and all(
                    isinstance(key, str)
                    and bool(key)
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                    for key, value in config.items()
                )
                for config in configurations
            )
            if not valid_hardware:
                reasons.append("missing or invalid prediction/reference hardware metadata")
            elif configurations[0] != configurations[1]:
                reasons.append("prediction/reference hardware mismatch")
        interval = row.get("reference_interval_cycles")
        if interval is not None and not (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and all(_positive(v) for v in interval)
            and _positive(row.get("reference_cycles"))
            and interval[0] <= row["reference_cycles"] <= interval[1]
        ):
            reasons.append("invalid reference_interval_cycles")
        if row.get("ranking_group") is not None and (
            not isinstance(row["ranking_group"], str) or not row["ranking_group"].strip()
        ):
            reasons.append("invalid ranking_group")
        if row.get("split") not in ("calibration", "holdout"):
            reasons.append("undeclared calibration/holdout split")
        if row.get("provenance") not in ("fresh-rtl", "historical-rtl", "silicon"):
            reasons.append("unsupported reference provenance")
        seconds = row.get("host_seconds")
        if seconds is not None and not _positive(seconds):
            reasons.append("invalid host_seconds")
        signed = None if reasons else row["prediction_cycles"] / row["reference_cycles"] - 1
        result = {
            **row,
            "threshold_fraction": threshold,
            "signed_error_fraction": signed,
            "absolute_error_fraction": None if signed is None else abs(signed),
            "status": "open" if reasons else "pass" if abs(signed) <= threshold + 1e-12 else "fail",
            "reasons": reasons,
        }
        evaluated.append(result)
        if isinstance(row.get("ranking_group"), str) and row["ranking_group"].strip():
            group = groups.setdefault(row["ranking_group"], [])
            group.append(result)

    rankings = []
    for name, members in sorted(groups.items()):
        # Require explicit signatures: hardware is intentionally allowed to vary.
        signature_fields = ("workload_sha256", "endpoint", "target_sha256", "initial_state_sha256")
        valid_signatures = all(
            all(_sha256(m.get(key)) for key in signature_fields if key.endswith("sha256"))
            and isinstance(m.get("endpoint"), str)
            and bool(m["endpoint"].strip())
            for m in members
        )
        # Comparing different predictor releases or reference classes is a
        # separate experiment, not one candidate-ranking validation group.
        comparable_fields = (*signature_fields, "model_version", "provenance", "split")
        compatible = valid_signatures and all(
            all(m.get(key) == members[0].get(key) for key in comparable_fields) for m in members
        )
        pairs = []
        for left, right in combinations(members, 2):
            state = "open"
            if compatible and left["status"] != "open" and right["status"] != "open":
                lr, rr = left["reference_cycles"], right["reference_cycles"]
                lp, rp = left["prediction_cycles"], right["prediction_cycles"]
                left_interval, right_interval = (
                    _reference_interval(left),
                    _reference_interval(right),
                )
                overlap = max(left_interval[0], right_interval[0]) <= min(
                    left_interval[1], right_interval[1]
                )
                if overlap or abs(lr - rr) <= reference_tie_fraction * min(lr, rr):
                    state = "reference-tie"
                elif lp == rp:
                    state = "prediction-tie"
                else:
                    state = "correct" if (lr < rr) == (lp < rp) else "inversion"
            pairs.append({"left": left["id"], "right": right["id"], "status": state})
        counts = {
            s: sum(p["status"] == s for p in pairs)
            for s in ("correct", "inversion", "prediction-tie", "reference-tie", "open")
        }
        orderable = counts["correct"] + counts["inversion"] + counts["prediction-tie"]
        regret = None
        if compatible and all(m["status"] != "open" for m in members):
            best_prediction = min(m["prediction_cycles"] for m in members)
            # Worst reference among predicted winners: ties cannot hide a bad choice.
            selected = max(
                m["reference_cycles"] for m in members if m["prediction_cycles"] == best_prediction
            )
            regret = selected / min(m["reference_cycles"] for m in members) - 1
        rankings.append(
            {
                "group": name,
                "compatible": bool(compatible),
                "status": (
                    "open"
                    if not compatible or counts["open"] or not orderable
                    else "fail"
                    if counts["inversion"] or counts["prediction-tie"]
                    else "pass"
                ),
                "regret_basis": "reference point estimates; worst predicted tie winner",
                "reference_tie_policy": "overlapping declared intervals or predeclared fraction",
                "compatibility_fields": list(comparable_fields),
                "pairs": pairs,
                "counts": counts,
                "pairwise_accuracy": counts["correct"] / orderable if orderable else None,
                "worst_tied_winner_regret": regret,
            }
        )

    def summarize(subset):
        errors = sorted(r["absolute_error_fraction"] for r in subset if r["status"] != "open")
        return {
            "cases": len(subset),
            "passes": sum(r["status"] == "pass" for r in subset),
            "failures": sum(r["status"] == "fail" for r in subset),
            "open": sum(r["status"] == "open" for r in subset),
            "valid_comparisons": len(errors),
            "median_absolute_error_fraction": median(errors) if errors else None,
            "p90_absolute_error_fraction": errors[math.ceil(0.90 * len(errors)) - 1]
            if errors
            else None,
            "max_absolute_error_fraction": max(errors) if errors else None,
            "p95_absolute_error_fraction": errors[math.ceil(0.95 * len(errors)) - 1]
            if errors
            else None,
            "all_cases_meet_threshold": bool(subset) and all(r["status"] == "pass" for r in subset),
        }

    return {
        "schema": "ventus-accuracy-audit-v1",
        "metric": "abs(prediction_cycles-reference_cycles)/reference_cycles",
        "reference_tie_fraction": reference_tie_fraction,
        "evidence_validation": {
            "mode": "caller_declared_metadata_and_arithmetic_only",
            "artifact_contents_verified": False,
            "completion_independently_verified": False,
            "holdout_independence_verified": False,
            "hardware_metadata_pairs_supplied": sum(
                "prediction_hardware" in row and "reference_hardware" in row for row in rows
            ),
            "hardware_metadata_scope": "Declared field equality only; unlisted fields and realized RTL require producer evidence.",
            "note": (
                "A valid SHA-256 string identifies a declared receipt; it does not prove "
                "that the artifact exists, completed, used the named endpoint, or was independent. "
                "Pass means the supplied valid point estimates meet the arithmetic gate."
            ),
        },
        "summary": summarize(evaluated),
        "by_split": {
            s: summarize([r for r in evaluated if r.get("split") == s])
            for s in ("calibration", "holdout")
        },
        "by_provenance": {
            s: summarize([r for r in evaluated if r.get("provenance") == s])
            for s in ("fresh-rtl", "historical-rtl", "silicon")
        },
        "cases": evaluated,
        "rankings": rankings,
        "scope": "Only declared cases/endpoints; no arbitrary-parameter or full-program guarantee.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON list of reference/prediction rows"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="New output directory; never overwritten"
    )
    parser.add_argument("--reference-tie-fraction", type=float, default=0)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    result = audit(json.loads(raw), reference_tie_fraction=args.reference_tie_fraction)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
