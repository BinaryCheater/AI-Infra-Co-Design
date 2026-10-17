# Causal L1/LSU candidate — 17 October 2026

**Experimental, not promoted.** `ventus-causal-memory-experimental-v1` replaces
L1 future-admission bookings and the uniform LSU address offset with current-edge
state. This is development replay of 112 existing RTL runs: no new RTL,
independent validation or optimization experiment was executed.

## Implemented path

`codesign/ventus/causal_memory.py` runs full overlapping straight-line probe
programs through the v3 hit-path frontend and RF/compute model, then:

1. Holds newly issued requests in InputFIFO until the following edge. Save and
   instruction-MSHR allocation, then cache/LDS handshakes, use subsequent states.
2. Accepts each line into a single non-flow cache request slot. Primary and
   individual targets allocate at actual service; same-address miss allocation
   blocks simultaneous conflicting admission.
3. Holds lower responses in a two-entry non-flow/non-pipe queue; drains targets
   through a one-entry pipelined target response, a one-entry response stage and
   `threads`-entry core response queue. A primary releases only after target drain
   and empty-entry acknowledgement; refill/release stalls affect current admission.
4. Delivers responses only to an idle instruction MSHR. Multi-line loads wait for
   all groups. Dcache responses take priority over available LDS responses. Local
   store acknowledgements and physical writes remain separate; writes contribute
   to the terminal drain.

Requests retain operation, instruction-MSHR entry and group identities. Prediction
checks exactly-once cache response conservation, drained queues/instruction groups,
causal response consumption and independent event-graph replay of internal times.

Source anchors: `LSU.scala` InputFIFO/AddrCalculate, `pipeline/MSHR.scala`,
`L1MSHR.scala` targets/releasing_stall and `DCache.scala` request/response queues.
This is a source-informed partial translation, not established full equivalence.

## Remaining approximations

The candidate retains Builder's L2/external-memory transaction calendar and L1
LRU approximation. Full tag-probe/status retention, miss-request versus return
acknowledgement interactions, SRAM conflicts, shared instruction/data arbitration
and return-queue backpressure propagated into the lower controller remain open.
LDS has finite bank rounds, without a full ShareMem controller translation. V3
collector/fetch readiness remains; the separate source-ready experiment is not
automatically included. Branches, regext, cold Icache/CTA and complete launch/flush
timing remain outside the claim. Input is explicit decoded instructions, not
arbitrary compiled ELF. This is Python-only, separate from native/default evaluator.

Hardware fields describe existing resources; no physical resource or cost
coefficient is added. No new realizability, synthesis, clock, area or energy result.

## Saved-reference results

| Reference set | Cases | v3 ≤5% | Candidate ≤5% | v3 ≤10% | Candidate ≤10% |
|---|---:|---:|---:|---:|---:|
| Development RF4 | 24 | 22 | 20 | 23 | 22 |
| Development RF8 | 24 | 21 | 21 | 23 | 22 |
| Former independent RF4 | 24 | 20 | 22 | 23 | 23 |
| Former independent RF8 | 16 | 10 | 13 | 15 | 15 |
| Post-reference RF8 | 24 | 21 | 22 | 22 | 24 |

All sets are development data for this new candidate. The mixed total is not
independent accuracy evidence: within-5% rises from 94 to 98; within-10% remains
106; worst error rises from 12.760% to 16.231%. Absolute error improves in 55 cases,
worsens in 51, and is unchanged in six. Across 16 compatibility groups inherited
from the original repeated-panel GEMM audit, pair inversions decrease from 12 to
nine. Selection by predicted cycles then candidate name retains zero reference-cycle
regret in every group. Prediction/reference ties are separately recorded. Neither
accuracy target is established by these aggregates.

The worst development failure is six-warp streaming without RF-layout conflict:
1378 versus 1645 RTL cycles (16.231%); v3 predicts 1478. RF4 seven-warp conflict
streaming predicts 1313 versus 1550. Continue diagnosis here; no constant correction
or per-case choice of whichever model fits is applied.

## Evidence and reproduction

[replay.json](replay.json) retains all case cycles/errors, first observed
divergences, event mismatch counts, counters, source/trace hashes, original audit
identities, raw-receipt hashes and ranking results. Every v3 prediction/reference
pair was checked against its frozen audit row by bank count and trace hash: all
112 agree. Historical inputs, traces, summaries and predictions remain unchanged.

```bash
./run --local causal-memory-replay \
  --cases results/accuracy-20261016/holdout/cases.json \
  --reference-root results/accuracy-20261016/rtl-rf4 \
  --banks 4 --out results/new-causal-replay/development-rf4.json
```

Use a new output path. Other pairings follow the source-ready replay guide.
`--local` explicitly analyzes locally retained evidence without changing the
private execution default. Incomplete/open cases remain in the report.

Internal regression covers hit edges, same-line merging, finite primaries, full
response-queue backpressure, mixed read/write conservation, multi-line completion,
sequential-block cache reuse and complete multiwarp probes. These checks do not
establish numerical/RTL equivalence for arbitrary programs. Overall repair remains
active, including compiled-program integration and new independent validation.
