# Standalone Ventus performance model

This branch packages the Ventus model, independent compiler/RTL tools, examples,
regressions and model evidence. It imports no agent backend, UI or chat/session
service. Python 3.12+, NumPy/HiGHS and Rust are sufficient for compiled timing;
fresh LLVM compilation and RTL execution are optional additional environments.

## Run a checked-in FP32 GEMM ELF

From `models/ventus/` (use this package's environment):

```bash
uv sync --frozen --extra dev --python 3.12
uv run ventus-model instructions \
  --input examples/compiled-gemm-model.json \
  --out results/gemm-model-001 --budget 60
```

Use a fresh output directory. The example supplies a real R1-compatible ELF,
launch dimensions, input words and the corrected private-memory layout. No
compiler, RTL build, agent or remote server is needed. The first native call
builds a dependency-free Rust library in ignored `results/.rust/`.

`experiment.json` records modeled cycles, counters, the source/runtime/input
identities and explicit numerical/RTL `not_run` status. `stages.jsonl` adds each
dispatch's finite-memory accounting; `checkpoint.json` supports recovery between
completed dispatches. These model results do not constitute a numerical check.

Pass `--hardware path/to/overrides.json` to compare declared hardware settings.
Each run starts cold unless explicitly resumed; dispatches within a run retain
the documented cache state. Record software/input/target identities when comparing
costs. Changing a model field does not establish a corresponding RTL realization.

## API entry points

| Input or action | Entry point |
|---|---|
| ELF loading and launch/resource metadata | `codesign.ventus.elf.ELFProgram` |
| Native streamed compiled execution | `codesign.ventus.rust.PreparedProgram`, `RustSession` |
| Serialize/inspect expanded instructions | `PreparedInstructions.to_bytes/from_bytes`, `compact_records` |
| Native decoded straight-line probes | `codesign.ventus.rust_frontend.predict_span` |
| Generate registered tiled FP32 GEMM source | `codesign.ventus.gemm_lowering` |
| Fresh LLVM compilation | `ventus-model compile-kernel --help` |
| Independent numerical RTL checker | `python -m codesign.ventus.compiled_check --help` |
| Standalone compile/execute/recover controller | `python -m ventus_project.compiled_workflow --help` |
| Source-derived model bounds | `codesign.ventus.runtime_contract.contract` |

The optional controller reads ignored `local.toml` and follows its local/SSH
default. Direct model APIs/CLI execute on the invoking machine. There is no
implicit fallback between native, Python and RTL. See [compiled workflow](compiled-workflow.md)
and [configuration](local-configuration.md) for tool/profile setup.

## Mechanisms and scope

Compiled v10 uses current-edge RF/collector/issue state, finite instruction MSHRs,
L1 primary/target/return queues, bounded L2 transactions and external response
slots. Multiply has an independent elastic pipeline; divide service depends on
known operands and active lane groups. Pending transactions and dirty writes
drain before terminal completion. Native v4 probes share the data-memory core;
their fetch-hit frontend is separate from compiled decoded instruction supply.

Intentional approximations remain in full instruction supply, inclusive L2/SRAM
arbitration, SFU overlap and barrier timing. Physical DRAM scheduling is outside
the wrapper target. Packed byte/halfword stores have a retained R1 numerical
failure; subgroup/atomic/system fences and extended precision are not baseline
validated. See the [machine-readable scope](evidence/core-repair-20261017/mechanism-contract.json).

Historical receipts retain their exact sources and failures. The packaging fix
restores `PreparedInstructions.to_bytes()` at class scope and checks execution
after a real compiled packet round trip; it changes no Rust timing semantics.
Old runtime hashes are not rewritten to match the packaged Python API.

## Engineering checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
cargo test --manifest-path codesign/ventus/rust_core/Cargo.toml
uv run python -m ventus_project.evidence --integrity-only --archive-only
uv run ventus-project audit
```

This embedded package checks its shipped small regression-fixture archive; it
does not require LFS netlists or include the full historical synthesis archive.
Original full manifests remain reference records. New evidence manifests bind
their own files. See [import scope](../IMPORT.md). Private configuration, build outputs and fresh
results are ignored. Tests and evidence checks are not user or research acceptance.
