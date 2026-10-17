# Current status

## Active model

The compiled default is `ventus-instruction-rust-v10-causal`; native decoded
probes default to `ventus-source-frontend-v4-causal`. Both use finite current-edge
RF/LSU/data-cache behavior and the pinned external wrapper boundary. Historical
v3 probes remain available with `causal=False`; legacy v7 templates are separate.
See [the model guide](model-v10.md) for standalone commands and API entry points.

Compiled execution includes matched R1 work-item support, whole-bank-row RF
allocation, multiwarp/multiblock streams, interleaved private stacks, static
resource admission and uniform workgroup barriers. Independent multiply and
input-dependent divide units now replace generic ALU timing for those operations.

## Independent execution and evidence

The standalone tools compile kernels, prepare native/RTL inputs, execute the
configured target and recover immutable results. No agent runtime or UI is
required. [Compiled workflow](compiled-workflow.md) documents local/SSH execution.
With no private profile, execution is local and no server connection is attempted.

- [Core mechanism repair](evidence/core-repair-20261017/README.md): current-edge
  queues/units, finite external slots, a repeated-session determinism fix and
  112 saved-trace development comparisons with gains and regressions.
- [Private layout](evidence/private-layout-20261017/README.md): a same-ELF driver
  correction removes 32 wrong words from a 384-output diagnostic.
- [Cooperative execution](evidence/compiled-sync-20261017/README.md): real compiled
  exchange/GEMM kernels, repeated barriers, tails and private spills.
- [Static resources](evidence/compiled-resources-20261017/README.md): compiler and
  full-workgroup resource checks, distinct from achieved occupancy.
- [Frontend RTL evidence](evidence/frontend-accuracy-20261016/README.md) and
  [native v3 parity](evidence/native-frontend-20261016/README.md): historical
  calibration and implementation observations retain their original versions.

Frozen evidence includes observations from an external consumer of this model;
that consumer's agent/backend/UI code is not part of this branch. Each receipt
binds its own source hashes. Packaging restores the compact `to_bytes()` API;
older runtime hashes and worker snapshots are preserved, not relabeled current.

## Intended scope and remaining approximations

The model supports kernel/design comparison, including FP32 GEMM. Final selected
baseline/optimized designs still need independent RTL correctness/performance
measurements before reporting an observed speedup. Internal tests are engineering
checks, not research acceptance.

Compiled instruction fetch, cold I-cache, redirects and complete host launch
are excluded. L2 inclusive probes/SRAM arbitration and complete SFU overlap are
approximate; barriers use conservative block fences. Subgroup/atomic/system
fences and extended precision lack baseline RTL validation. A real packed
byte/halfword-store diagnostic fails on the fixed R1 execution path; its failure
is retained and its compiler/driver/RTL root cause remains unresolved.

External memory matches `Mem_SimWrapper`, not physical DRAM bank/row scheduling.
Changed channel/bandwidth/latency settings are explicit parametric extensions.
Resource capacity checks do not certify every hardware knob's RTL realization,
clock, synthesized area or achieved occupancy. Existing cost coefficients and
historical optimization records remain unchanged.

Earlier research experiments remain in [baseline records](evidence/baseline-run-20261007/README.md).
They retain their old model versions and are not rerun or claimed as new results.
