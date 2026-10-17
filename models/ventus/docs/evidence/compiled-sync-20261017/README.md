# Compiled workgroup synchronization and FP32 GEMM — 17 October 2026

Native runtime `ventus-instruction-rust-v9-4` now preserves each warp's execution
state across uniform workgroup barriers and publishes memory at the selected
local/global fence. Actual compiled OpenCL programs exercise repeated barriers,
cross-warp LDS exchange, loops, padded tails and scalar/private vector spills.

| Fresh RTL program | Warps/block | Blocks | Output words | Barrier arrivals in model | Result |
|---|---:|---:|---:|---:|---|
| Exchange, 173 valid items, two rounds | 2 | 3 | 346 | 24 | exact uint32 outputs |
| Exchange, 351 valid items, three rounds | 4 | 3 | 1,053 | 72 | exact uint32 outputs |
| Cooperative FP32 GEMM, M7 N9 K11 | 2 | 2 | 63 | 16 | exact FP32 outputs |

All three full-GPU RTL runs have exact named completion, no error diagnostics and
correct independent reference words. The GEMM reference uses small signed integer
FP32 inputs with exactly representable sums. It stages 8x8 A/B fragments in LDS,
uses actual compiled `vfmadd.vv`, iterates K stages and keeps padded lanes in both
barriers. This is a real compiler-generated kernel, not a handwritten ISA fixture.
The source uses Clang's `__builtin_fmaf`; this bounded compile wrapper does not
link a full libclc math library. An earlier standard `fma` external-call link
failure remains in the ignored compiler result directory.

## Execution contract and limits

Each warp retains scalar/vector registers, PC, SIMT state, VL and instruction
budget between barrier epochs. Warps see their own stores immediately. Other
warps' writes are published only by the matching local/global memory fence.
Access histories retain non-fenced regions. Byte masks detect unordered cross-warp
RAW/WAR/WAW conflicts instead of manufacturing a host execution order. Uniform
barriers require all workgroup warps, the same PC/flags, full active masks and no
outstanding divergent path. Missing arrivals, wrong flags, different sites and
data races fail explicitly. This is not a full OpenCL correctness/race proof.
Subgroup synchronization, atomics, cross-workgroup communication and standalone
device/system fence execution remain unsupported or unverified.

The generated IR retains each arrival instruction and places every epoch's
pre-barrier work before all arrivals. Timing uses the existing conservative block
fence, waiting for preceding work to retire. Streamed materialization remaps
dependencies when operation slots are recycled. Exact per-warp barrier control
timing has not been calibrated. Expanded and streamed native execution agree for
the tested programs, including resource-limited one-SM recycling and repeated
dispatches. These checks do not establish RTL timing accuracy.

The source frontend v3 and native compiled timing remain distinct paths. Native
model cycles for the declared baseline are 1,301, 2,166 and 2,667 respectively.
They omit the RTL boot/CTA/cold-I-cache/final-flush interval and must not be
reported as measured RTL cycles or used as proof of optimization benefit.

## Private spill allocation repair

The first exchange runs produced correct outputs but were rejected because RTL
logged an unallocated private-memory backing read. Compiler-generated vector
spills use interleaved private storage; a partial cache-line store may fetch its
backing line. The loader had reserved user buffers but omitted that allocation.

The loader now supplies a zeroed backing segment for all padded launched threads,
using the compiler's private bytes/thread and whole-cache-line rounding. It rejects
overlap and private backing above 1 MiB; this is an assistant-selected checker
limit, adjustable in `compiled_check.py`, not an architectural capacity. Zeroing
does not certify source-level initialized private reads. Native functional state
still treats absent private values as unknown. No latency or kernel arithmetic
was fitted. [missing-private-backing.json](missing-private-backing.json) preserves
the rejected runs, including their otherwise-correct outputs.

## Reproduction and artifact identities

Sources: [exchange](../../../examples/kernels/workgroup_exchange_u32.cl),
[GEMM](../../../examples/kernels/workgroup_gemm_f32.cl).
[Compiled artifacts and resource receipts](../../../tests/fixtures/compiled-workgroup)
pin ELF/source/compiler/linker identities. Compile with the existing official LLVM
and explicit compatibility library, using new output directories:

```bash
./run --local model compile-kernel --source examples/kernels/workgroup_exchange_u32.cl --kernel workgroup_exchange_u32 --compiler-root <llvm-build> --llvm-source <llvm-source> --workitem-profile rtl-r1 --out results/exchange-NEW
./run --local model compile-kernel --source examples/kernels/workgroup_gemm_f32.cl --kernel workgroup_gemm_f32 --compiler-root <llvm-build> --llvm-source <llvm-source> --workitem-profile rtl-r1 --out results/gemm-NEW
./run --local compiled-check prepare --manifest examples/compiled-sync-check.json --out results/exchange-input-NEW
./run --local compiled-check prepare --manifest examples/compiled-gemm-check.json --out results/gemm-input-NEW
./run compiled-check rtl --input <prepared-input>/cases.json --out results/sync-rtl-NEW --rtl-root <configured-rtl-root> --spdlog <configured-spdlog-root> --observer <configured-observer> --timeout 180
```

The configured remote default is retained. Transfer prepared input/current checker
to it before `rtl`; actual checks used fresh remote snapshots and the existing
configured binary/library/observer. Existing services and results were retained.
[cases.json](cases.json) and [GEMM input](gemm/cases.json) bind unchanged ELF words,
boot context, private backing, launch and input bytes. Each case directory stores
actual metadata/data, stdout/stderr, observer trace and hash-bound summary.

From the workbench repository, reproduce evaluator calls through its actual
bounded subprocess runtime and response validator:

```bash
.venv/bin/python scripts/check_ventus_compiled.py --ventus-source ../ventus-codesign --manifest ../ventus-codesign/examples/compiled-sync-check.json --rtl-evidence ../ventus-codesign/docs/evidence/compiled-sync-20261017 --out local/sync-check-NEW
.venv/bin/python scripts/check_ventus_compiled.py --ventus-source ../ventus-codesign --manifest ../ventus-codesign/examples/compiled-gemm-check.json --rtl-evidence ../ventus-codesign/docs/evidence/compiled-sync-20261017/gemm --out local/gemm-check-NEW
```

The diagnostic freezes each design/registry/launch identity, uses a new call
workspace and the standard runtime's process-group cancellation, 180-second tool
budget and 4-MiB output budget. These are assistant-selected diagnostic defaults
in its tool specification. It rechecks recorded RTL references, then runs native
timing and reports `fidelity=modeled`, barrier counts and resource/policy metadata.
It does not rerun RTL, invoke an agent/search or create/accept Runner state.
Raw numerical RTL evidence is distinct from modeled timing and user acceptance.
Generic template/solver/realizer binding and compiled timing accuracy remain open.
