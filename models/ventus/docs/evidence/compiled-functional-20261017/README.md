# Compiled FP32 workflow checks — 17 October 2026

Fresh OpenCL compilation, ELF loading, numerical full-GPU RTL execution, native
timing and direct workbench evaluator calls are connected for three registered
vector-add cases. All 1,024 output words match independent CPU references exactly.
These are finite integration checks. Overall goal and user acceptance remain open;
no optimization search, agent research run or full Runner loop occurred.

| Case | Warps/block | Blocks | FP32 words checked | RTL result | Native model cycles |
|---|---:|---:|---:|---|---:|
| single | 1 | 2 | 64 | completed, exact outputs | 415 |
| single | 2 | 3 | 192 | completed, exact outputs | 515 |
| adjacent pairs | 4 | 3 | 768 | completed, exact outputs | 1,015 |

Cycles are native v9-3 predictions from fresh cold model state per case, **not**
RTL intervals or accuracy comparisons. Model timing omits explicit boot context,
cold instruction-cache/CTA costs and RTL final output flush. The calibrated source
frontend and compiled dynamic provider remain separate paths.

## Repairs and retained failures

Archived ELF uses vector CSR instruction `0x80d02072`, which pinned R1 rejects.
Those files remain unchanged. The existing LLVM build was recovered from the
earlier Qwen session; its `clang` is a symlink. No compiler rebuild was needed.

Explicit `--workitem-profile rtl-r1` selects older official scalar-CSR/VID work-item
assembly. Original instructions remain byte-identical. Added version-3 resource
summaries retain six source call edges and conservatively charge v0..v6/x0..x31.
Link-time aggregation reports 28 LDS stack bytes per warp for these kernels.
Kernels compile normally without binary patching. This library does not certify
arbitrary kernel ISA compatibility.

The first four-warp test completed with 59 incorrect words. OperandCollector
shifts register bases by log2(bank count); allocating raw usage of seven caused
overlapping register rows across warps. `ELFProgram.launch` now separates usage
from allocation and rounds SGPR/VGPR footprints to complete RF-bank rows. The
instruction CLI passes the chosen hardware's bank count. The RTL loader targets
baseline RF4. Repeating the same programs passes all cases.

RTL input contains unchanged ELF text plus an explicit ABI boot stub. Output
buffers start with NaN poison; expected values never initialize output data.
Success requires exact named completion, clean diagnostics and correct words at
every output address. Failed checks exit nonzero. No arithmetic or latency was
fitted. [failures.json](failures.json) retains earlier failure summaries/mismatches;
original private execution directories remain preserved separately.

## Reproduction and scope

[Registered input](../../../examples/compiled-functional-r1-check.json) uses
[fresh compiled fixtures/resource receipts](../../../tests/fixtures/compiled-vecadd-r1).
The original newer-CSR input remains a negative compatibility example.

```bash
./run --local model compile-kernel --source <official-kernels>/src/vecadd_f32.cl --kernel vecadd_f32 --compiler-root <existing-llvm-build> --llvm-source <llvm-source> --workitem-profile rtl-r1 --out results/r1-single-NEW
./run --local model compile-kernel --source examples/kernels/vecadd_f32_pairs.cl --kernel vecadd_f32_pairs --compiler-root <existing-llvm-build> --llvm-source <llvm-source> --workitem-profile rtl-r1 --out results/r1-pairs-NEW
./run --local compiled-check prepare --manifest examples/compiled-functional-r1-check.json --out results/r1-input-NEW
./run compiled-check rtl --input <prepared-input>/cases.json --out results/r1-rtl-NEW --rtl-root <configured-rtl-root> --spdlog <configured-spdlog-root> --observer <configured-observer>
```

`rtl` honors the private execution profile; transfer prepared input/current checker
to that machine first. Fresh checks used isolated configured remote directories,
existing binary/library, environment and observer. Local compilation explicitly
used `--local`. Defaults, Web services and old result directories were retained.

[cases.json](cases.json) binds ELF, initialized input bytes, boot words and resources.
Each case includes metadata/data, raw stdout/stderr, trace and a summary binding
binary/library/log hashes. [manifest.json](manifest.json) indexes the evidence.

Workbench `scripts/ventus_compiled_tool.py` rebuilds the launch, rechecks raw
reference logs and executes the same ELF/input through native timing. Changed
program, registry, launch scope or hardware is rejected. The adapter currently
admits the declared baseline only and reports `fidelity=modeled`. All three direct
calls passed workbench response validation; see
[workbench-integration.json](workbench-integration.json).

The four-warp case also passed an actual stdin/stdout subprocess invocation and
response validation; [workbench-stdio.json](workbench-stdio.json) records it.

General template/solver/realizer binding, collective barriers, wider kernel/ISA/
tail/atomic coverage and compiled timing accuracy remain open. Detailed DRAM was
not added; this target retains the pinned RTL memory wrapper.
