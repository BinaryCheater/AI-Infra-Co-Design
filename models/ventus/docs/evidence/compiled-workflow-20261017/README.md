# Compiled multiwarp workflow check — 17 October 2026

Current replay: [ABI-compatible receipt](abi-compatible/experiment.json),
[stage journal](abi-compatible/stages.jsonl) and [identities](abi-compatible/manifest.json).
This follow-up preserves the legacy private-size word at the end of launch
metadata and accepts both legacy and extended tails. Earlier records below are
retained with their original runtime identities; totals are unchanged.

The user's current priority is complete execution paths and major mechanism
coverage, rather than fine timing calibration. Runtime
`ventus-instruction-rust-v9-2` now accepts multiple full warps/block for compiled
programs without collective synchronization. No new accuracy sweep was run.

## Implemented capability

The native engine creates a stream and scoreboard for each warp, materializes
operations into the matching stream, and recycles all streams only after block
completion. LDS is allocated by available work-group intervals, independently of
physical warp numbering. The launch ABI includes warp-private scalar stack stride
and static local bytes; LDS residency charges static bytes plus every warp's stack.
Existing one-warp input remains accepted. The allocation policy is an explicit
model choice, not independently established official-runtime/RTL equivalence.

Compiler reference: [pinned FrameLowering](https://github.com/THU-DSP-LAB/llvm-project/blob/97df137e687669f9acf4bfd734779f52f2878614/llvm/lib/Target/RISCV/RISCVFrameLowering.cpp)
distinguishes scalar spill, vector spill and local-memory stacks. This check
verifies the independent scalar spill stacks exercised by the compiled vecadd
programs; arbitrary stack/local-memory combinations remain to be validated.

The experiment receipt now reports `completion_scope`, `functional_validation`
and `rtl_validation`. A completed timing dispatch does not imply numerical output
or RTL correctness. A collective barrier instruction fails explicitly.

## Actual execution

The reproducible manifest references the existing real official-compiler ELF
corpus in `tests/fixtures/compiled-vecadd`. Its source receipt pins compiler,
software and ELF hashes. The check reuses those binaries; no new compilation,
synthetic instruction replacement, LLM call or optimization search was performed.

| Dispatch | Warps/block | Blocks | Source |
|---|---:|---:|---|
| single-local64-blocks7 | 2 | 7 | Official FP32 vecadd |
| single-local128-blocks7 | 4 | 7 | Official FP32 vecadd |
| pairs-local64-blocks7 | 2 | 7 | Compiled two-output variant |
| pairs-local128-blocks7 | 4 | 7 | Compiled two-output variant |

All four dispatches completed on one modeled SM with four warp slots, exercising
several waves of admission and release. The recorded total is 8749 model cycles
and 4914 dynamic warp instructions. This is not hardware performance. Output
values were not numerically evaluated; numerical and RTL validation are `not_run`.

```bash
./run --local model instructions --input examples/compiled-workflow-check.json \
  --out results/new-compiled-workflow-check --budget 60
```

Use a new output directory. `--local` explicitly selects this locally available
corpus/runtime check, without changing the private execution default.

## Evidence and remaining work

[experiment.json](experiment.json) contains runtime/input identities, hardware,
stage resources, completion scope, counters and explicit limitations.
[stages.jsonl](stages.jsonl) records every completed dispatch.
[manifest.json](manifest.json) binds their bytes and the reproducible input.
Original v9-1 results and prior outputs were preserved. Source/runtime-bound old
checkpoints require their matching old runtime; they are not silently migrated.

Regression compares streamed and fully expanded execution, complete output-address
coverage, separate warp stacks, repeated-dispatch state and cache accounting for
both compiled variants, and legacy versus extended single-warp launch decoding.
Existing single-warp and native/frontend checks remain
required. These checks do not substitute for numerical or RTL validation.

Next major workflow obligations: independent numerical checking, compiled collective
barriers/shared communication, fresh compilation integrated with the checker,
and workbench evaluator binding. Source-ready and causal-memory timing candidates
remain isolated; no per-case timing selection or fitted correction is applied.
