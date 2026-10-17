# Concrete source → compiled ELF → model/RTL workflow

Current native timing default is v10 causal; the same workbench GEMM ELF replays
at 2,845 modeled cycles with its saved RTL outputs rechecked. Earlier v9.5 costs
below remain historical receipts. Solver costs must be regenerated under the new
runtime identity before comparing a new optimization table. See
[core mechanisms and concrete remaining limits](evidence/core-repair-20261017/README.md).

`ventus_project.compiled_workflow` replaces manual source packaging, SSH transfer,
execution and artifact retrieval. It performs concrete integration checks and
does not invoke an optimizer, an agent, or acceptance. The controller compiles
with explicit local tools; native timing and RTL follow `local.toml`'s execution
default. An explicit `--execution local|remote` overrides that target only.

The workbench now uses this same controller for finite compiled-cost planning.
`start --phase native` runs only native timing and needs no RTL/spdlog roots;
default `--phase all` retains native plus full RTL numerical checks. Phase is
frozen in submission/result identities and checked again on retrieval. A
native-only completed job explicitly reports RTL `not_run`; it cannot become a
numerical-check receipt. An observer is accepted only with full RTL execution.
Existing receipts without phase retain historical `all` semantics.

`build(..., compiler_budget=...)` adds a finite shared host budget for compiler
commands, each still capped by its per-command timeout. Exhaustion writes a
failed receipt without launching another command. The workbench Solver shares
its remaining planning budget; ordinary standalone builds retain their existing
per-command 120-second default when no host budget is specified.

Actual workbench family → HiGHS → Realizer → checker → evaluator calls completed
for one declared tile=(4,16,8), M11 N13 K9 point. Solver modeled a fresh native-only
job, Realizer retained its ELF, checker separately ran full RTL (143 exact
outputs), and evaluator replayed 2,864 modeled cycles. Same-backend agreement is
not timing accuracy. Formulator configuration can be prepared without tools;
actual LLM/Runner research execution remains pending. No search study was run.

## Bind a selected GEMM assignment

The registered `ventus-cooperative-gemm-lowering-v1` constructor accepts
`tile_m`, `tile_n`, `tile_k`. One lane computes one row-major FP32 output;
cooperative A/B staging and both barriers are generated. Workgroups contain full
warps within the baseline warp capacity, and static staging must fit LDS while
leaving space for compiler stacks. Final compiler resources still require
feasibility checks; the constructor does not guess register/private stack use.

```bash
.venv/bin/python -m codesign.ventus.gemm_lowering --cases examples/gemm-lowering-input.json --tile-m 4 --tile-n 16 --tile-k 8 --out results/selected-NEW
.venv/bin/python -m ventus_project.compiled_workflow start --manifest results/selected-NEW/workflow.json --compiler-root <local-llvm-build> --llvm-source <local-llvm-source> --observer <configured-target-observer> --out results/workflow-NEW
```

The case file supplies A/B buffers, three pointer arguments plus M/N/K, and C
reference words. The constructor derives global/local sizes from the selected
tiles. It independently checks the references using host FP32 `fmaf` in sequential
K order. Inputs are finite FP32, row-major, with no transpose/bias/alpha/beta.
The diagnostic payload bound is the existing 1 MiB word-buffer limit; these are
checker/constructor limits, not a claim about all OpenCL programs or hardware.
K tails execute only valid K terms, avoiding extra padded-zero FMA rounding effects.

Before compilation, the checker regenerates source from the assignment and
checks exact source/entry, launch, ABI/buffer shape and independent references.
Unknown constructor fields, source drift, changed launch or bad references fail.
This establishes correspondence to the registered lowering. It is not a universal
correctness proof or model/hardware optimality certificate. Registered finite
Solver/Realizer calls now use this constructor as described above; arbitrary
constructors and actual Formulator/Runner research execution remain open.

## Arbitrary concrete single-source checks

An unbound `ventus-compiled-workflow-v1` manifest can instead supply a concrete
single OpenCL source file and frozen cases. It needs `source`, `source_sha256`,
`kernel`, `cases`, and optionally `optimization`/`workitem_profile`. Supported case
fields are exactly `name`, `arguments`, `global_size`, `local_size`, `inputs` and
`outputs`; ELF identity/entry are inherited from the fresh compilation. The source
must be self-contained apart from the configured compiler/work-item headers.
This path uses caller-supplied numerical references and carries no registered
source/assignment correspondence claim. Unknown ISA or execution fails explicitly.

All artifacts go into a new directory. `work/` contains frozen source/spec,
optional `binding.json`, compile receipts/ELF, numerical cases and initialized
driver input. Compiler receipts also bind linker script and generic support-header
tree hashes. Remote execution uses a fresh source snapshot without private config,
an immutable job ID and code hashes. Native timing uses fresh cold state per case;
RTL checks named completion, diagnostics and every addressed reference word.

The default `driver_profile=workgroup-private-v1` is frozen in the spec/input.
An explicit host-input adapter sets the per-warp private-base increment to zero,
matching the LSU's existing workgroup-wide word interleave. It is built separately
against the configured binary's generated headers; the RTL library is unchanged.
The adapter is active, separately hashed, and its application is checked in logs.
Fresh RTL runs therefore need the configured C++/Verilator headers in addition to
the existing binary. Adapter compilation failures are explicit; there is no
fallback to the upstream private layout. See the [source derivation, real numerical
failure and repair](evidence/private-layout-20261017/README.md).
Native v9-5 uses the matching actual private base through `r1-workgroup`.
The original `upstream-r1` driver profile remains identifiable for historical
references; unsafe multiwarp private use is rejected by the current launcher.

The target is the declared baseline used by the existing R1 checks; this does not
generate or certify arbitrary hardware configurations. The optional observer is
explicit: omitting it means no observer trace, rather than a substitute trace.
No detailed DRAM model is added. Timing uses native v9-5, with its documented
conservative barriers and unmatched RTL boot/CTA/cold-fetch/final-flush interval.

Compiled launch admission now checks per-workgroup warp/register/LDS capacity,
whole RF-bank rows and every warp's scalar stack before native execution or RTL
input emission. RTL preparation additionally checks the declared grid and private
interface envelope. Receipts retain compiler usage, allocated resources and a
static resident-block upper bound separately. This bound excludes achieved
occupancy, allocator timing/fragmentation and physical implementation. See
[source-bound resource checks and failure replay](evidence/compiled-resources-20261017/README.md).
This validation has its own version; timing coefficients are unchanged.

## Budgets, cancellation and recovery

Defaults are assistant-selected and configurable: 120 seconds per compiler
command, 900 seconds for worker execution, 180 seconds per native/RTL case,
and a poll-enforced 64 MiB stdout/stderr budget per external process. Compiler
timeout is an API argument; worker/per-case budgets are CLI arguments. The
controller also bounds transfer/observation calls and allows worker budget plus
60 seconds for its SSH transport. Process groups are terminated on timeout,
output overflow, interruption or termination/hangup signals, including descendants.
Budget failure never becomes a completed or correct result.

```bash
.venv/bin/python -m ventus_project.compiled_workflow resume --out results/workflow-NEW
```

Resume observes the recorded job and retrieves a terminal reference. It never
submits or restarts work. Running, failed, interrupted and unconfirmed observations
remain distinct. A transient observation/retrieval failure leaves the job identity
available for later observation. Downloads use fresh staging paths; verified
references are committed once and subsequently rechecked read-only. Profile drift
rejects observation before a connection. No profile implies local execution and
requires explicit local RTL/spdlog paths; an absent profile never invents a server.

`execute_prepared(out, ...)` now supports the stage boundary after compilation:
`out/work` supplies frozen source/spec, compiler/ELF, constructor/cases and input.
It revalidates those bindings and regenerates admission input before connecting
to the configured target. It neither recompiles nor resubmits an existing job.
The workbench Realizer/Gate path uses this entry point to execute the exact
transported ELF. Its standard-role integration is documented in the workbench's
`docs/evidence/compiled-roles-20261017/README.md`; this does not itself connect a
general Formulator/Solver or start an optimization experiment.

The controller rechecks downloaded identity, native receipt hashes and raw RTL
completion/output evidence. It retains partially failed workflows. Successful
compilation, native completion, finite numerical RTL and timing accuracy remain
separate claims. [Actual evidence](evidence/compiled-bindings-20261017/README.md)
includes the recovered missing-worker-dependency failure and a clean rerun.

The workbench evaluator v3 detects a registered realization receipt, regenerates
and rechecks it, verifies compile/source/ELF/case identity and binds its hash into
the checked program. V3 also requires matching corrected-driver evidence; old
upstream-driver receipts remain historical and cannot be substituted. Actual
evaluation still reports modeled timing; it does not rerun compilation or RTL.
