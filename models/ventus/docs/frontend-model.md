# Straight-line frontend performance model

17 October follow-up: a separate [source-ready experiment](evidence/source-ready-20261017/README.md)
translates SlowDown packet admission through RR/collector readiness. Replaying
112 existing traces produced mixed gains and regressions; v3 remains the default.

Status: experimental opt-in `ventus-source-frontend-v3`, 2026-10-16. The default
`ventus-source-events-v7` body model and native Rust executor retain their existing
semantics. This extension has Python reference and Rust native implementations; it does not
silently change compiled ELF or Transformer predictions. See [native execution](native-frontend.md).

## Why instruction supply matters

The former multiwarp GEMM benchmark started its measurement at the first collected
instruction but admitted all warp streams immediately in the model. The RTL fetch
scheduler chooses the lowest eligible physical warp ID. With eight resident warps,
the later warps can wait while earlier warps fill their finite instruction buffers.
This delay belongs inside a first-collect-to-last-writeback interval.

The old benchmark also truncated every warp at its last Tensor instruction. An
earlier-finishing warp can execute its output-address epilogue and stores while
other warps are still computing. Those instructions compete for fetch, collectors,
RF, execution and memory resources. A compute endpoint may be narrow, but the input
program must still contain all concurrently executing instructions.

The new `complete_gemm` emitter retains the complete arithmetic and output-store
streams, with a separate 128-byte output slice per warp. Numerical output is checked
against a dense FP32 reference. Explicit ENDPRG uses the scalar collector and goes directly to the warp
scheduler, without fictitious scalar execution or writeback latency. Whole-kernel
flush timing remains outside the model; the RTL harness executes it and checks
named kernel completion separately. This distinction is especially relevant to short kernels.

## Source rules and explicit approximations

The pinned RTL snapshot is the same as the main project baseline.

| Mechanism | Rule | Source |
|---|---|---|
| Fetch selection | Lowest physical warp ID whose packet FIFO has space; independent of scoreboard readiness | `ventus/src/pipeline/warp_schedule.scala`, `pc_ready` loop |
| Fetch packet | Two consecutive 32-bit instructions | `ventus/src/top/parameters.scala`, `num_fetch` |
| I-cache hit pipeline | Request-to-response uses two registered stages | `ventus/src/L1Cache/ICache/ICache.scala`, `coreReqFire_st1/st2` |
| Packet storage | Two packets per warp, neither flow nor pipe | `ventus/src/pipeline/ibuffer.scala`, `InstrBufferV2.buffers` |
| Unpacking | One packet register per warp, one instruction consumed at a time | `InstrBufferV2.SlowDown` |
| Response overflow | Replay the failed request and invalidate younger same-warp requests | `pipe.scala` status override, `warp_schedule.scala` replay and `ICache.scala` external flush |
| RF requests | Live per-bank RR among collector × physical operand-slot IDs; synchronous returns and duplicate grants retained | `operandCollector.scala`, `operandArbiter` and `collectorUnit` |
| RF initialization | RR last-grant register initializes to zero in measured Verilator profile; first search starts after zero | Chisel `RRArbiter`; executable initialization assumption |
| Collector output | Choose RR winner before testing target ready; blocked LSU can block other vector units | `operandCollector.scala`, `DualIssueIO` |
| Collector reuse | Collector leaving `s_out` is available on following cycle | `collectorUnit` state registers |
| Instruction MSHR | Per-cycle idle/add/out controller; idle-only response/allocation, deferred simultaneous allocation, WB readiness and empty-out bubble | `pipeline/MSHR.scala`, `MSHRv2` |
| Cache admission | Primary/secondary capacity and refill-target release stall feed address release | `L1MSHR.scala`, `DCache.scala`; transaction approximation in `cache_acceptance.py` |
| Downstream readiness | Approximation using scoreboard and collector-token readiness | `frontend.py`; exact combinational RR-ready propagation remains a gap |

These are fixed baseline mechanism rules, not additional independent synthesized
hardware knobs. Existing `Hardware` fields still configure RF, execution, memory
and residency. A new FIFO depth/fetch-width design family needs corresponding RTL
and cost validation before becoming an exposed search parameter.

This extension assumes instruction-cache hits. A cold first-line fetch often
precedes the first collected instruction and is outside the stated interval;
later line misses, instruction/data sharing in L2, branch redirects, regext,
noncontiguous instruction streams and CTA initialization are not modeled here.
These warm-I-cache assumptions differ from the cold RTL runs: observed instruction
misses after first collect remain inside their reference interval. Agreement on
those runs does not establish matched initial cache state. Cache admission still
schedules future transactions: extending a target list can extend a release after
an earlier admission dependency was booked. The instruction MSHR controller also
aggregates partial-lane/multiple-line returns. Exact cache response backpressure
and shared instruction/data L2 arbitration remain gaps.

The input validator rejects gaps and collective template barriers, whose IR is
not a one-to-one hardware instruction stream. It does not certify the absence of
all unsupported ISA effects merely from a PC name.

## API and command-line use

```python
from codesign.ventus.config import Hardware
from codesign.ventus.rtl_accuracy import complete_gemm
from codesign.ventus.timing import build_graph
from codesign.ventus.graph import execute
from codesign.ventus.physical_operands import with_physical_operands
from codesign.ventus.termination import append_endprg

program = complete_gemm(steps=12, warps=6, strategy="resident")
workload = with_physical_operands(
    append_endprg(program.workload, program.words), program.words
)
graph = build_graph(workload, Hardware(), frontend=True)
result = execute(graph)
times = dict(zip(graph.names, result["times"], strict=True))
first = min(t for name, t in times.items() if name.endswith(".collect"))
last = max(times[f"w{w}.pc{program.final_tensor_pc:08x}.writeback"] for w in range(6))
compute_span = last - first
```

For repeated predictions, `frontend.predict_span(workload, hardware,
endpoint_names=[...])` uses the same controller without materializing the full DAG;
endpoint names omit `.writeback`. This reference path uses Python per-cycle frontend/LSU stepping.
`rust_frontend.predict_span` implements the same recurrence in Rust. The legacy
Rust ABI still rejects ENDPRG/physical-slot metadata; the new explicit frontend
ABI accepts them. See [native execution](native-frontend.md). The bounded physical
operand decoder covers the emitted GEMM subset, not the general ISA. Omitting
physical slots falls back to source tuple positions and is unsuitable for exact
RF-contention claims.

Use the configured execution wrapper (`./run rtl-accuracy ...`) for the private
machine default. Direct Python invocations execute on the machine that receives
them. Create a new directory for every collection. For example:

```bash
./run rtl-accuracy generate --suite holdout --out results/new-probes \
  --fixture archive/analysis/ventus_flow_20261006/search_model/kernel-probes-v3/stream-4-rf0-lds1/input
./run rtl-accuracy predict --cases results/new-probes/cases.json \
  --frontend --banks 4 --out results/new-predictions.json
./run rtl-accuracy rtl --cases results/new-probes/cases.json \
  --out results/new-rtl --jobs 2 --timeout 120
```

Prediction requires explicit `--banks 4` or `--banks 8`; the CLI has no implicit
bank-count default. Artifacts record the requested hardware, and the audit checks
prediction/reference hardware agreement independently of filenames.

The runner accepts explicit `--rtl-root`, `--observer`, and `--spdlog` paths or
reads the corresponding private-profile fields. It requires an existing compatible
binary/observer; it does not rebuild hardware. A supplied RF8 binary must correspond
to a documented realization of that geometry. The frozen predictions are inputs to
the evaluator only, never inputs to the RTL run. Raw trace, output, input hashes,
binary/library/observer hashes, functional status, timing interval and host cost
must accompany an accuracy claim.

## Interpreting validation

Use `abs(prediction - RTL) / RTL` per case. Averages must not hide cases above 10%,
or above 5% for GEMM. Compare ranks only between candidates with identical
mathematical work, precision, initial-state assumptions and endpoints. Near ties
need an explicit tolerance. A compute-span match is not a whole-kernel wall-time
match; numerical output checks do not establish timing accuracy by themselves.

The extension adds a mechanism that was missing from the first-collect interval.
It is not a universal cycle-accurate simulator or a guarantee for arbitrary
parameters, cache footprints, ISA extensions or full networks. The new numerical
and ranking receipts document the measured domain and failures separately.

## Development and independent validation

The initial 24-case suite (1/3/6/8 warps, K=24/96/192/96; resident, reload,
stream; two RF layouts) became development data after its discrepancies informed
source analysis. Its status remains calibration even though its directory was
originally named `holdout`. Both RF4 and separately realized RF8 RTL references
are retained. On RF4, v3 reduced the six-warp streaming underprediction from
25.1–28.4% to 10.15% and 9.99%; this still fails the requested GEMM 5% threshold.
No universal 90% or GEMM 95% claim follows from these results.

A second suite was generated with frozen v3 before RTL collection:
2/5/7 warps, K=40/128/160; resident, reload, stream and shared/LDS; two RF layouts,
on RF4 and RF8 (48 fresh reference runs). Only its RF4 prediction artifact was
correctly configured before collection. Self-review found the file named RF8
actually contained RF4 predictions; its original mismatched pairs are now marked
open and excluded. Correct RF8 predictions were recomputed after references with
unchanged timing code and are reported separately as post-reference replay.
A replacement independent RF8 suite uses 4/6 warps, K=56/144 and all four strategies
with both RF layouts (16 cases), with explicit RF8 predictions frozen before new
RTL runs. Prediction receipts hash the case manifest and model code. Results and ranking failures must be reported in
full, including shared-memory cases. Mathematical inputs repeat identical K
panels, making residency/reload alternatives equivalent in this bounded experiment;
this is not evidence for arbitrary GEMM layouts or precision. Fixed fixture
metadata reserves 4096 bytes LDS even for nonshared kernels, while their model
specifies zero; this one-block admission mismatch is recorded rather than treated
as exact metadata agreement.

## Execution cost and unresolved mechanisms

A same-host, three-repetition RF4 check over the 24 validation inputs measured
median graph-build-plus-execute costs of 3.36–146.02 ms and summary-controller
costs of 2.85–87.57 ms (Darwin arm64, Python 3.12, one Python worker). Both timings
include physical-operand/ENDPRG transformation and a fresh model state; file
parsing and process startup are excluded. All paired cycle counts matched. These
measurements are input/host specific and do not establish a speedup over a native
backend measured on a different machine or workload. The later
[native frontend evidence](evidence/native-frontend-20261016/README.md) measures
Python and Rust side by side on the configured execution host.

A separate, unpublished cold-I-cache experiment added two-cycle miss-status
replay and shared lower-level line reads. It left all 24 RF4 development compute
spans unchanged: modeled instruction misses were hidden by fetch-ahead. It is
not part of v3 and does not resolve the remaining streaming discrepancy. Further
work needs source-level shared-L2 response arbitration and transaction acceptance
analysis, rather than adding a fixed miss penalty to a measured endpoint.

## Queued LSU request edge: known v3 approximation

`LSU.scala` uses a one-entry `InputFIFO` with `pipe=true, flow=false`.
A newly issued request enters the FIFO at cycle `t`, dequeues at `t+1`, and
allocates its instruction-MSHR entry in `AddrCalculate.s_save` at `t+2`.
An already buffered request dequeued when the address unit becomes idle reaches
`s_save` one cycle after that dequeue. V3 uses a uniform two-cycle offset and
also inherits the transaction builder's previous-address-release dependency;
its queued-request edge is therefore approximate. This is separate from the
correctly retained MSHR source quirk: the `s_out` next-state condition tests the
free `valid_entry` bit, while actual retirement releases `output_entry`.

A source-motivated phase experiment distinguished buffered from newly enqueued
requests and moved the inherited previous-release boundary back one edge. All
12 one-/three-warp development cases then matched RTL exactly. Six-warp conflict
streaming improved from 1577 to 1749 cycles (RTL 1752), but eight-warp conflict
streaming regressed from 1027 to 1186 (RTL 1034, 14.70% error). The experiment's
origin-anchored dependency adapter is not a completed causal controller and is
not promoted as v4. Its patch description and all 24 outcomes are preserved in
[the evidence package](evidence/frontend-accuracy-20261016/README.md).

The next bounded implementation must replace the transaction admission calendar
with per-cycle state, not add a kernel-specific correction:

1. Represent InputFIFO occupancy and AddrCalculate idle/save/cache states using
   pre-edge valid/ready values, preserving enqueue/dequeue and release overlap.
2. Represent the Dcache one-entry core-request queue and tag-probe stage; gate on
   primary/secondary MSHR capacity, same-address in-flight miss and refill writes.
3. Allocate primary entries and individual target slots at actual miss-request
   acceptance, rather than future-booking all targets at instruction issue.
4. Hold each lower-memory return in the two-entry return queue, drain target slots
   through the one-entry miss-response queue and one-entry response stage, then
   clear the primary entry. Keep global `releasing_stall` until the actual return
   handshake; never mutate a release after dependent events have been scheduled.
5. Feed the `NLanes`-entry Dcache core-response queue into the idle-only instruction
   MSHR input, honoring Dcache-over-LDS priority and output WB backpressure.
6. Couple shared instruction/data L2 requests through their actual arbitration;
   validate request acceptance, returned target order and occupancy before timing
   endpoints. Freeze the replacement and measure new shapes/configurations.

The source's WB ordering and Tensor elastic-register recurrence were checked
against v3 and did not justify an extra fixed Tensor latency. In the difficult
six-warp streaming trace, mean collect-to-WB service is 31.98 versus RTL 35.38
cycles for loads, 18.53 versus 22.94 for Tensor, and 8.64 versus 10.14 for vector
operations. Eight-warp service means are much closer. The remaining error is
phase/contention dependent, so uniformly inflating execution latency would hide
the unresolved request/response scheduling problem.


The final corrected audit contains 48 calibration and 40 pre-reference validation
cases (24 RF4 plus 16 newly frozen RF8). Calibration has 43/48 within 5% and 46/48
within 10%; independent validation has 30/40 within 5% and 38/40 within 10%, with
maximum error 11.709%. Independent ranking has 135/140 pairs correct and five
inversions, although each group chooses a reference-best candidate. The separate
24-case post-reference RF8 replay reaches 21/24 within 5% and maximum error
12.760%. These are bounded measured results, not attainment of all requested
accuracy or ranking targets. The invalid filename-based RF8 pairings remain open
in a separate audit and contribute to none of these counts.
