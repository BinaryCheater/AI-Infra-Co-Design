# Ventus performance model

Standalone Ventus model embedded in `memxlife/AI-Infra-Co-Design` under
`models/ventus/`. It has no dependency on the repository's `agentic/` system.

## Run the real compiled GEMM example

Python 3.12+ and a stable Rust toolchain are required. From the repository root:

```bash
cd models/ventus
uv sync --frozen --extra dev --python 3.12
uv run ventus-model instructions \
  --input examples/compiled-gemm-model.json \
  --out results/gemm-model-001 --budget 60
```

The ELF and input words are included; LLVM, RTL, a GPU and server access are not
needed for this timing run. Use a fresh output directory. The first call builds
the dependency-free Rust core into ignored `results/.rust/`. Results explicitly
separate modeled timing from numerical/RTL validation, which this command does
not run.

## Included capability

- Compiled runtime `ventus-instruction-rust-v10-causal`: dynamic control/address
  execution, multiwarp/multiblock streams, resource admission, private spill
  layout, uniform workgroup barriers and finite RF/LSU/cache/return behavior.
- Native probe `ventus-source-frontend-v4-causal`, sharing the causal data-memory
  core; explicit v3 and legacy v7 paths retain their separate semantics.
- Independent multiply and operand-dependent divide service; finite five-slot
  external wrapper response behavior, distinct from physical DRAM scheduling.
- Existing cost/MILP APIs, optional compilation/RTL/local-SSH tools, examples,
  regression tests and versioned numerical/timing evidence.

Start with [the model guide](docs/model-v10.md), [scope and status](docs/status.md)
and [core evidence](docs/evidence/core-repair-20261017/README.md).
[IMPORT.md](IMPORT.md) records exactly what was copied and what was omitted.
`local.toml`, new results and build caches are ignored; no owner/server profile
is included. With no private profile the controller defaults to local execution.

## Scope

The model supports FP32 GEMM/design comparison with documented fetch/L2/SFU/barrier
approximations. Packed narrow stores have a retained R1 numerical failure.
Final selected designs need independent RTL correctness/performance measurements
before claiming observed speedup. No physical DRAM controller or surrounding
agent system is imported.

## Check the package

Run inside this directory, using its independent environment:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
cargo test --manifest-path codesign/ventus/rust_core/Cargo.toml
uv run python -m ventus_project.evidence --integrity-only --archive-only
uv run ventus-project audit
```

The reference source is an explicitly labeled excerpt for static audits, not a
complete RTL build checkout. Fresh RTL uses a separately configured full checkout.
The archive contains only small model regression fixtures; full synthesis
intermediates and the original repository's Git history are omitted.
