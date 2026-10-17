# Source frontend v3: numerical, timing and ranking evidence

The main audit separates 48 development cases from correctly configured,
pre-reference predictions for independent validation. RF4 and a separately
realized RF8 binary are measured. Every row records actual prediction hardware
and reference hardware; executable hashes link references to `realizations.json`.

## Correction discovered during self-review

The original `validation-predictions-rf8.json` filename was misleading: its rows
specified `banks=4`. Pairing it with RF8 RTL was invalid. The file is preserved;
`invalid-rf8-pairing-audit.json` marks all 24 such pairs open, and they are excluded
from main accuracy and ranking claims. Earlier 96-case aggregate claims are
superseded. A filename is never evidence of the hardware used.

Correct RF8 predictions for those same references are stored in
`validation-postreference-rf8.json` and audited separately in
`postreference-rf8-audit.json`. All timing-module hashes match the pre-RTL freeze,
but these predictions were produced after references and are not counted in the
main pre-reference validation. They have 21/24 cases within 5%, maximum error
12.760%; failures remain visible. Their machine-readable split is conservatively
`calibration`, with `evidence_role=post-reference-replay`, so automated aggregation
cannot accidentally include them in the independent denominator.

A new independent RF8 suite (4/6 warps, K=56/144; resident/reload/stream/shared;
two RF layouts) was generated and explicitly predicted with eight banks before
its 16 fresh RTL runs. Its prediction artifact is
`validation-rf8-corrected-predictions.json`. The predictor CLI now requires
explicit `--banks`, and audits reject mismatched prediction/reference hardware.

## Final measured coverage

| Main-audit split | Numerical + completion checks | GEMM error ≤5% | Error ≤10% | Maximum error |
|---|---:|---:|---:|---:|
| Development/calibration | 48/48 | 43/48 | 46/48 | 10.152% |
| Independent pre-reference validation | 40/40 | 30/40 | 38/40 | 11.709% |

The independent set comprises 24 RF4 and 16 corrected RF8 cases. Its five
same-work ranking groups contain 135 correct pairs and five inversions out of
140, with zero reference or prediction ties. Each group selects a reference-best
candidate (zero best-selection regret), while complete ranking accuracy remains
unmet. The separate 24-case post-reference RF8 replay is excluded from this table.
There are 112 distinct fresh RTL references in the package, plus 24 retained
invalid pairings that reuse existing references.

## Measurement contract

Error is `abs(prediction-reference)/reference`. The requested universal 90% and
GEMM 95% accuracy targets remain unmet. All failures remain in the receipt. The
initial directory called `holdout` became calibration after its evidence informed
mechanism development; it is never counted as independent validation.

The interval starts at first operand collect and ends at final Tensor writeback.
Full overlapping epilogues and ENDPRG are supplied; whole-kernel flush latency is
not predicted. Exact named RTL completion, trace instruction agreement and
numerical outputs are checked independently. Zero process exit alone is inadequate.

Ranking groups hold logical work and endpoints fixed, allowing strategy, physical
RF layout and RF-bank realization to vary. Inputs repeat identical K panels,
including identical inter-warp shared-memory values. This bounded equivalence
family does not validate arbitrary GEMM optimization. Exact reference ties are
retained, with no post-hoc tolerance for small reversals.

The model assumes warm I-cache hit service; RTL starts cold and has post-start
instruction misses. This mismatch, queued LSU request timing and partial cache
response/backpressure abstraction are explicit limitations. Metadata reserves
4096 bytes LDS in nonshared RTL fixtures while their model specifies zero; these
one-block probes do not validate equivalent occupancy limits.

## Files and reproduction

- `frontend-v3-audit-rows.json` and `frontend-v3-audit.json`: main measurements,
  errors and ranks, excluding the invalid and post-reference RF8 pairings.
- `postreference-rf8-audit*.json`: separate correctly configured replay.
- `invalid-rf8-pairing-audit*.json`: retained invalid comparison, all rows open.
- `*-predictions*.json`: frozen predictions; read each row's actual `banks` field.
- `realizations.json`, `rf-bank-width.patch`, `rf4-parameters.json`: source,
  configuration, generated-RTL, binary/library and observer identities.
- `reference-summaries/`: raw summaries with numerical values and timing hashes.
- `signature-preimages.json`: exact logical-work/target/initial-state hash inputs.
- `frontend-v3-runtime.json`: same-host graph/summary timings, three repetitions,
  fresh state and transformation/build/execution included; file parsing excluded.
- `reproduction-check.json`: summary/graph consistency, raw-summary hash checks,
  audit recomputation and unchanged timing-module hashes; internal checks only.
- `frontend-v3-phase-residuals.json`: instruction service diagnostics.
- `address-phase-development.json` and `address-phase-experiment.diff`: unpromoted
  queued-request phase experiment, including a 14.70% regression.
- `manifest.json`: artifact and case-manifest hashes, portable raw locations.

Recompute the main audit without running or fitting a predictor:

```bash
./run accuracy-audit \
  --input docs/evidence/frontend-accuracy-20261016/frontend-v3-audit-rows.json \
  --out results/new-frontend-audit
```

Use `rtl-accuracy generate --suite holdout` for development, `--suite validation`
for the original second suite, and `--suite validation-rf8` for the corrected RF8
suite. Always create new output locations. See [frontend model](../../frontend-model.md)
for API and remaining exact-state implementation work. Raw traces/logs remain in
ignored results and the private configured flow directory; their hashes are
recorded. Later model improvements need a new version and independent validation.
