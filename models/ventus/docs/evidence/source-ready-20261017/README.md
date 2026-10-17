# Source-ready experiment — 17 October 2026

**Not promoted to the default model.** This is a post-reference development
replay against 112 previously executed RTL runs. No new RTL, independent
validation or optimization experiment was performed.

## Source mechanism

`InstrBufferV2.SlowDown` only loads its next packet when `mask_next == 0` and
downstream `out.ready` holds. The latter is gated by the scoreboard, Chisel's
round-robin input grant and free collector slots. A producer need not be valid
to receive ready. When its mask is empty, its output control bits select slot
zero of the retained packet, not the next program instruction.

The v3 model substitutes next-instruction dependency readiness for this network.
`codesign/ventus/source_ready.py` adds an isolated source-derived candidate,
`ventus-source-ready-experimental-v1`, retaining the existing fetch-hit and
cache/LSU approximations. It uses the declared zero-initialized Verilator arbiter
profile. It accepts only the tied multi-warp collector geometry; it does not
establish arbitrary control-flow, masked/extended instruction, physical CTA or
cold-I-cache fidelity. It is Python-only and is not the workbench evaluator.

## Results and decision

| Existing reference set | Cases | v3 ≤5% | Candidate ≤5% | v3 ≤10% | Candidate ≤10% |
|---|---:|---:|---:|---:|---:|
| Development RF4 | 24 | 22 | 21 | 23 | 22 |
| Development RF8 | 24 | 21 | 20 | 23 | 22 |
| Former independent RF4 | 24 | 20 | 20 | 23 | 24 |
| Former independent RF8 | 16 | 10 | 10 | 15 | 15 |
| Post-reference RF8 | 24 | 21 | 22 | 22 | 23 |

All sets are development data for this experiment, including those formerly
independent of v3 development. The mixed 112-case total is not independent
accuracy evidence: within-5% cases fall from 94 to 93; within-10% remains 106.
Absolute error worsens in 22 cases. The RF4 shared-w5-k128-conflict0 span improves
from 2211 to 2309 against 2372 RTL cycles (6.788% to 2.656%), while other shared
and high-warp cases regress. No candidate-ranking improvement is claimed.

The experiment establishes a concrete omitted source mechanism and measures
the result of translating it within the current approximation. It does not
prove that every candidate handshake equals RTL, nor that another mechanism is
the sole root cause. Default v3 remains unchanged. Work continues on causal
cache/LSU acceptance and returns and their interaction with collector readiness.

## Evidence and reproduction

[replay.json](replay.json) retains every case's two predictions, reference span,
first observed divergence, event mismatch counts, trace hashes and source hashes.
Each replayed baseline prediction and reference span was checked exactly against
the existing v3 audit row with the same bank count and trace hash. All 112 match.
The raw diagnostic receipts remain in ignored `results/debug-20261017/`, with
their relative paths and hashes in the compact receipt. The original RTL traces
and evidence were not modified.

```bash
./run --local source-ready-replay \
  --cases results/accuracy-20261016/validation/cases.json \
  --reference-root results/accuracy-20261016/validation-rtl-rf4 \
  --banks 4 --out results/new-source-ready/rf4.json
```

`--local` explicitly chooses local replay of already-local evidence; the private
execution default is unchanged. Use a new output path. Other recorded pairings:
`holdout` with `rtl-rf4`/`rtl-rf8`; `validation` with `validation-rtl-rf8`;
`validation-rf8-corrected` with `validation-rf8-corrected-rtl`. Explicitly choose
four or eight banks to match each reference realization.

Internal checks include RR ready for invalid producers, downstream-full stalls,
stale-control hazards, refusal of unsupported collector geometry, existing native
parity tests and the full regression suite. They do not replace RTL mechanism
validation or user acceptance. No performance model default was changed.
