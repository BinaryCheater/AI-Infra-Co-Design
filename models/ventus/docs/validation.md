# Validation and reproduction

Embedded-package note: [IMPORT.md](../IMPORT.md) defines the shipped subset.
Full historical datasets/netlists and a full upstream RTL checkout are not bundled;
older full-export commands and artifact counts below describe the original repository.
Use [the current model guide](model-v10.md) for this package.

Latest [native core repair](evidence/core-repair-20261017/README.md) exercises
current-edge memory/unit mechanisms on real compiled ELF and the evaluator.
A new full RTL diagnostic has a retained packed-store counterexample and exact
32-bit integer outputs. The 112-case replay records gains and regressions without
accuracy fitting or a threshold gate. Historical evidence below is unchanged.

Latest: [compiled synchronization and cooperative FP32 GEMM](evidence/compiled-sync-20261017/README.md)
adds three complete RTL checks covering repeated barriers, tails, private spills,
and 1,462 exact output words. Bounded workbench runtime/protocol integration is
recorded separately from numerical RTL and from unestablished timing accuracy.

17 October numerical integration: [fresh compiled FP32 RTL checks](evidence/compiled-functional-20261017/README.md)
cover 1/2/4 warps, multiple blocks and 1,024 exact output words. The compiled
workbench adapter rechecks raw references and performs native timing; it reports
modeled cycles. This is separate from compiled timing accuracy and a full
agent/solver/realizer run. Original failed compatibility/allocation inputs remain.

17 October workflow check: [compiled multiwarp execution](evidence/compiled-workflow-20261017/README.md)
uses archived real ELF variants at two/four warps per block, seven blocks per
dispatch and one SM to exercise residency/recycling. Streaming matches expanded
execution, complete output-address coverage and resource/cache accounting in the
regressions. This is control/address/timing integration evidence, with numerical
and RTL validation explicitly not run. No per-cycle accuracy fitting is involved.

17 October: [causal-memory development replay](evidence/causal-memory-20261017/README.md)
compares all 112 saved RTL references with an isolated current-edge memory path.
Internal conservation passes; accuracy has mixed gains/regressions and still
fails the requested bounds. Old references are not independent validation for
the new candidate. Original v3 prediction/reference pairs remain exact.

The 2026-10-16 extension adds [fresh frontend RTL evidence](evidence/frontend-accuracy-20261016/README.md).
The original extraction checks below retain their historical scope. New per-case
accuracy targets remain unmet; test success does not change that result.

The extraction adds portability and privacy checks without recalibrating the
model. A redundant event-MILP completion bound resolves the observed Linux proof
timeouts; read-only Transformer replay accepts a larger host budget on slower
machines. Neither change adjusts predicted cycles or synthesized costs. Code/receipt hash migration is documented in the export manifest. None of
these checks imply full-network RTL equivalence or arbitrary-parameter accuracy.

## Fast regressions

```bash
uv run ruff check codesign ventus_project tests
uv run ruff format --check codesign ventus_project tests
uv run pytest -q
```

For configured server execution use `ventus-project run tests -q`. Direct
commands above run locally, useful for a collaborator's independent checkout.
Tests include workload legality, source control-state timing, Tensor backpressure,
bank conflict/ports, parameter sensitivity, shared FP arbitration, Transformer
conservation, ISA/numerical fixtures and cost budget behavior.

## Current prediction checks against preserved evidence (2026-10-16)

```bash
./run evidence --integrity-only --archive-only
./run historical-accuracy --out results/historical-new --benchmark-native
./run rtl-accuracy --help
./run accuracy-audit --help
```

Archive-only integrity verifies the 1,834 retained evidence artifacts and explicitly
excludes changed executable source. Default full-export integrity remains strict
and will reject the changed model code; it must not be bypassed and described as
an unchanged historical-code replay. `historical-accuracy` instead hashes current
source, verifies old raw artifacts, instruction streams, numerical outputs and
exact named kernel completion, and makes new predictions against those traces.
Those observations remain historical calibration, not fresh holdouts.

[Fresh RTL receipts](evidence/frontend-accuracy-20261016/README.md) record both numerical and timing outcomes. Audit arithmetic
cannot itself prove raw completion, realized hardware or independence: the producer
must establish those facts. Failed/missing cases remain visible; nonzero return,
error diagnostics, missing exact completion, or output mismatch cannot pass by
providing a cycle count. Pair ranking requires identical work/target/state/endpoint;
no tolerance is selected after seeing inversions. Best-choice regret is separate
from correctness of every pair.

## Original frozen evidence replay

One configuration-aware command checks export integrity and the selected frozen
receipts on your default machine:

```bash
uv run ventus-project run evidence
```

Individual local commands, from the project root with its package installed:

```bash
uv run python analysis/ventus_flow_20261006/cost_model/scripts/build_table.py analysis/ventus_flow_20261006/cost_model/raw codesign/ventus_costs/table_v1.json --verify
uv run python analysis/ventus_flow_20261006/quick_complete_20261006/check.py --verify
uv run python analysis/ventus_flow_20261006/performance-v7/check.py --verify
uv run ventus-model verify-transformer --out analysis/ventus_flow_20261006/performance-v7/small-s16-final --budget 300
```

These consume saved RTL evidence and replay predictions. They do not rerun RTL or
synthesis. Whole-GPU traces start at the recorded instruction-collection endpoint;
compute completion and final memory visibility are distinct endpoints. Broader
observed kernel errors do not bound full Transformer error.

The official C++/SystemC comparison failed several timing/resource tests in the
original study. Current fast-model rules are extracted from RTL rather than fitted
to that simulator. Original comparator inputs, outputs and mismatch reports are
under `model_accuracy/` and `evidence/`; the upstream comparator source/toolchain
is not installed as a dependency of the Python model.

## New independent experiments

```bash
uv run ventus-model extract --rtl-root vendor/ventus-gpgpu --out results/source-rules.json
uv run ventus-model demo --out results/menu-001
uv run ventus-model verify --out results/menu-001
uv run python analysis/ventus_flow_20261006/cost_model/scripts/run_demo.py results/cost-menu-001
uv run ventus-model transformer --input examples/tiny-transformer.json --out results/tiny-001 --budget 60
uv run ventus-model verify-transformer --out results/tiny-001
uv run ventus-model transformer --preset gpt2 --mode census --out results/gpt2-census-001
```

Census is a shape/instruction-organization inventory, without a complete timed
execution. Full GPT-2 timed execution has not completed within the earlier budget.
Fresh hardware checks should target a changed parameter's realization and likely
bottleneck before claiming optimization gain. Retain failures and intervals.

The packaging verification outcome is recorded in `docs/extraction-checks.json`.

## Structural cost v2 checks (2026-10-07)

`tests/test_ventus_structural_costs.py` checks all 27 fields, baseline partition,
replicated-port storage, immutable targets, precision capabilities, random legal
factor reconstruction, sensitivity and actual HiGHS feasible/infeasible budgets.
It preserves the strict v1 rejection tests. The new evidence directory
`docs/evidence/cost-structural-v2-final-20261007/` records source/table hashes, fit
residuals and leave-one-out errors. These errors concern FP32 component shapes,
not an error guarantee for the wider architecture or precision implementation.


## Native compiled-program regressions

`tests/test_ventus_rust.py` compares the native executor with the Python detailed
path across mixed streams, shared pipelines, Tensor shapes, RF ports, cache and
LDS constraints, including barriers. `tests/test_ventus_instruction_program.py`
compares two real compiled vector-add implementations with expanded/native/Python
execution and repeated-dispatch cache state, and validates hashed-journal recovery.
`tests/test_ventus_compiled_precision.py` checks actual official BF16/FP16/TF32
MMA encodings, grouped A/B/C reads and eight destination writes, masked tails and
two-/four-byte traffic. It checks full Qwen dimensions, bias/weight tying, GQA
aliases, KV append bytes and cross-dispatch dependencies. Small complete compiled
programs check all declared address ranges and fast/per-cycle native equivalence
under several hardware configurations. Full-task receipts independently check
dynamic MMA counts against launch shapes.

[Native program execution and remaining accuracy gaps](instruction-programs.md)
are explicit. These comparisons validate executor semantics and program coverage;
they do not extend the existing local FP32 RTL error bound to full BF16 Qwen.
No fresh RTL or SystemC simulations were run for the original extraction delivery.
The 2026-10-16 extension below records new RTL separately.

## Unified baseline validation

`tests/test_ventus_unified_area.py` checks frozen config hashes and instruction
targets, default structure, area arithmetic, explicit external-memory projection,
unsupported configurations, and an actual MIP storage/compute exchange. The latter
selects more physical storage than the baseline under the same total-area budget,
ensuring no hidden legacy bit cap remains. It tests the area interface, not full
network optimality. Query `./run costs --model unified-v4 --hardware
examples/baseline-hardware-v1.json` through the configured execution wrapper.
Historical v1/v2 receipts remain unchanged.

## Frozen baseline execution and range audit

[Two completed Qwen runs and two 512 extrapolations](evidence/baseline-run-20261007/README.md) use the frozen hardware and program settings. [Range checks](evidence/baseline-range-audit-20261007/README.md) exercise every legal option on a compiled GEMM, record conditional changes, and prove 8-SM cost infeasibility within the current menu/budget. Probe inactivity is not a global performance proof. `./run --local baseline --out results/prepare-check-new --prepare-only` validates hashes and generates both default scenarios without executing them.

Expanded-domain validation (2026-10-07): [cost minima and compiled register-capacity probes](evidence/search-expansion-20261007/README.md), with version/hash compatibility, fixed 3-SM MIP decode and shared-ELF timing regressions in `tests/test_ventus_search_expansion.py`. Full suite: 220 passed; Ruff check and format check passed. This validates model behavior, not new RTL configurations.

Native-array v1 / unified v4: 225 tests passed, including full RF/LDS domain lookup, padding, baseline preservation, native multiport capacity accounting and MIP decode versus independent costing. CACTI timing is diagnostic only. No new full-network simulation was required: the retained baseline runtime and inputs match.

## Native frontend implementation parity

The [native frontend](native-frontend.md) is checked against Python v3 at all
instruction collect/operand/issue/writeback/ENDPRG events, every named counter
and the declared endpoints. Additional cases cover multiple blocks and SMs,
physical-warp reuse, RF ports, collectors, cache/MSHR capacities, memory targets,
FPU contention and partial/multiline LDS/global accesses. The native path is also
checked with Python timing entry points disabled, ensuring no hidden fallback.

The configured `frontend-parity` command records same-host full/prepared call
costs and exits nonzero on mismatch. Existing RTL accuracy remains a separate
claim. [Measurements](evidence/native-frontend-20261016/README.md).
