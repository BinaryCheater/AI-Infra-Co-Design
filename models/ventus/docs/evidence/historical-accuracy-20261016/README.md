# Historical RTL calibration replay — 16 October 2026

This receipt compares new predictions from the default v7 body model against
**historical** RTL traces. It is calibration replay, not fresh RTL or an independent
holdout. The 1,834 frozen archive artifacts were checked against the exported
manifest before execution. Current executable sources intentionally differ from
that original export; their hashes are recorded separately in each row.

Eleven concrete kernel instances provide 22 endpoint comparisons: final selected
compute writeback and output memory visibility. All satisfy their declared
thresholds (5% for four GEMM memory-pattern instances, 10% for mixed kernels).
Maximum absolute relative error is **4.0789%**; median is **0.2464%**. The four
same-Tensor-work memory patterns have all six pairwise orderings correct at each
endpoint and zero selected-candidate regret. These twelve pair comparisons reuse
four instances; they are not twelve independent workloads.

Producer checks include frozen input hashes, binary/library/observer identities,
actual PC/instruction/warp streams, numerical output words and endpoint cardinality.
Completion here means the observed measurement endpoint; these old traces do not
establish a new full-launch or arbitrary kernel-termination accuracy guarantee.

## Prediction cost

On the configured execution machine, three uncached graph construction/execution
calls per instance had medians of 66.4–1,222.6 ms. The optional existing Rust backend
was also timed: five fresh hardware-state sessions with prepared instructions had
medians of **5.58–17.23 ms**, including session initialization and dispatch. Input
encoding took **1.33–38.44 ms** separately. Library load/build is recorded separately;
this is a warm software runtime with fresh modeled hardware state, not cached
answers. The native terminal agrees with the default graph terminal in all eleven
instances. This agreement does not establish frontend accuracy or a new RTL
endpoint comparison. At the time of this historical replay, the new frontend
mode was Python-only. The subsequent [native frontend port](../../native-frontend.md)
has separate parity/performance receipts and does not alter this frozen replay.

Wall-clock measurements are observations, not a machine-independent speed contract.
No large-network speed or accuracy bound follows from these small programs.

## Reproduction

```bash
./run historical-accuracy --out results/historical-new --benchmark-native
./run evidence --integrity-only --archive-only
./run accuracy-audit --input results/historical-new/rows.json --out results/audit-new
```

The wrapper uses the configured execution profile. Every output directory must be
new. The full legacy `evidence` command retains its original source identity checks;
it is not a validation route for a modified predictor. The archive-only check
excludes current code explicitly and cannot authorize legacy replay.

Files: [full audit](audit.json), [input rows](rows.json). These preserve the
`historical-accuracy-20261016-003` result, including source and trace identities.

## Strict completion replay (005)

The later [strict audit](strict-completion/audit.json) and
[strict rows](strict-completion/rows.json) additionally verify each exact named
kernel-completion marker and stdout/stderr error/limit diagnostics. All eleven
instances pass these checks; all 22 endpoint errors and pair rankings remain
unchanged. This supersedes the endpoint-only completion declaration in receipt
003 above. It still does not predict full-kernel launch/flush latency.

Configured-runtime integration initially exposed an older remote source checkout:
receipt 004 aborted on an unavailable native ABI symbol. Existing remote sources
were backed up, then maintained source/tests/examples were synchronized without
modifying historical evidence, results or private configuration. Receipt 005 uses
the matched current Python/Rust source. The aborted attempt remains in results.

In receipt 005, graph-build/execution medians span 46.76–946.47 ms; prepared-input
fresh native sessions span 3.20–7.25 ms, with encoding 1.15–30.90 ms. These
are a separate same-host batch and do not replace receipt 003 timings.
