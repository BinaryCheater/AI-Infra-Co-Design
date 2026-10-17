# Concrete realization and automated workflow — 17 October 2026

An explicit selected assignment, `tile_m=4, tile_n=16, tile_k=8`, now produces
registered cooperative FP32 GEMM source and derived launch dimensions. The
single workflow controller compiled it locally and ran native timing plus full
RTL on the configured x86-64 target, then retrieved and rechecked the evidence.
No candidate search, optimizer, agent or Runner acceptance was invoked.

The new workload is M11 N13 K9: three 64-lane workgroups, K/M/N tails, cooperative
LDS staging, private spills, and 24 dynamic barrier arrivals. All **143 FP32 output
words match independent references exactly**, with named RTL completion and no
error diagnostics. Native control/address execution completes 3,259 instructions
and predicts 2,864 cycles. These are modeled cycles, not an RTL timing interval
or evidence of optimization benefit. No timing coefficient was fitted.

The registered checker regenerates the source, checks source/entry/launch against
the selected assignment, and verifies independent host FP32 FMA references before
compilation. [binding.json](work/binding.json) states that limited correspondence
claim; [spec.json](work/spec.json), [source](work/source/kernel.cl),
[compile receipt](work/compile/build.json), actual ELF and [case manifest](work/cases.json)
preserve the same lineage. This does not prove universal kernel correctness or
hardware feasibility beyond the recorded checks.

The first automatic worker snapshot omitted `codesign.solver`, a dependency
imported by the instruction CLI. RTL numerical execution succeeded but native
failed, and the whole workflow correctly remained incomplete. The snapshot now
includes the shared dependency; its isolated import is regression checked. The
[failed workflow record](failed-worker-dependency.json) is retained separately.
Its original private logs remain in ignored results; the shared failure record
includes the original stderr hash and diagnosis.

The clean rerun's [worker result](reference/worker-result.json) contains separate
native and RTL outcomes. Raw RTL logs/trace/input and the native execution receipt
are included under `reference/work/execution/`. [Worker code hashes](reference/code-hashes.json)
identify the exact snapshot used. Later read-only resume rechecked its hashes and
raw numerical outputs without submitting another job. Current controller changes
also preserve the generated ELF identity in a separate fresh build.

The actual worker [source snapshot](source-snapshot/code-hashes.json) is frozen
alongside the hashes, including shared dependencies and native Rust code.
[Target identity comparison](target-identity.json) confirms the binary, RTL library
and observer hashes equal those in the earlier baseline GEMM reference. This is
reuse of that recorded target, not a new hardware characterization.

Workbench evaluator v2 binds the realization receipt hash into the program and
rechecks source, compile receipt and frozen cases before timing. Its actual bounded
runtime invocation passed for both the original workflow directory and this
portable export. [Runtime record](workbench-runtime-v2.json) preserves the response
with the selected assignment, resources and counters. Outputs remain
`fidelity=modeled`; numerical RTL reference and timing prediction are separate.

Reproduce source realization and the full workflow using the commands in
[workflow documentation](../../compiled-workflow.md). To replay just the checked
workbench evaluator from the workbench repository:

```bash
.venv/bin/python scripts/check_ventus_compiled.py --ventus-source ../ventus-codesign --manifest ../ventus-codesign/docs/evidence/compiled-bindings-20261017/work/cases.json --rtl-evidence ../ventus-codesign/docs/evidence/compiled-bindings-20261017/reference/work/execution/rtl --out local/bound-gemm-check-NEW
```

Complete Formulator/Solver/Realizer/Runner integration, wider constructor/kernel
coverage and compiled timing accuracy remain open. Existing 112-case timing
results and their failures were not overwritten or used to tune this workload.
Current services stayed available. Overall research goal and user acceptance
remain pending.
