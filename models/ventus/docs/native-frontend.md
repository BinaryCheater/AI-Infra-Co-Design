# Native source frontend

17 October update: `predict_span` defaults to `ventus-source-frontend-v4-causal`,
sharing the compiled executor's current-edge data-memory core. Set `causal=False`
for historical v3 described below and used by the parity command. Fetch remains
the finite hit-path model; compiled ELF supply has a decoded-stream boundary.
See [repair evidence and limits](evidence/core-repair-20261017/README.md).

The source-frontend-v3 timing model now has a Rust implementation. The new backend
executes instruction supply, live physical RF arbitration, collector/issue
blocking, execution pipelines, ENDPRG, instruction-MSHR control and the bounded
Dcache admission recurrence in native code. The hardware-only LDS service
interval template is also derived in Rust. Python validates and encodes inputs;
it does not construct a timing graph, schedule memory events or supply a cached
cycle answer to the native backend.

This is an implementation of the existing model, including its documented
approximations. It does not resolve the remaining RTL error or add arbitrary
branch/ELF semantics. The legacy compiled-program backend retains its old timing
contract. See [the model contract](frontend-model.md) and
[RTL evidence](evidence/frontend-accuracy-20261016/README.md).

## Execution

The configured `./run` entry point now selects Rust for `--frontend` predictions:

```bash
./run rtl-accuracy predict --cases results/probes/cases.json \
  --frontend --banks 8 --out results/native-predictions-new.json
```

Use `--backend python` to evaluate the reference implementation. Without
`--frontend`, the existing Python body-model route remains unchanged. Explicit
`--backend rust` requires `--frontend`. No backend silently falls back to another
implementation. Every output path must be new. Native build/load failures are
errors, not permission to substitute legacy timing.

`--max-cycles` defaults to 10,000,000 simulated cycles per fresh native execution.
This is an assistant-selected execution bound, not an architecture parameter or
measured kernel limit. Override it explicitly for longer inputs. Exhaustion is
an error; partial execution cannot become a valid prediction. The first call
builds the dependency-free release library if its source-hashed cache is absent.

## Direct API and repeated parameter queries

```python
from codesign.ventus.config import Hardware
from codesign.ventus.rust_frontend import PreparedFrontend, predict_span

# workload contains contiguous PC-named decoded operations, physical source slots
# and all overlapping epilogues/ENDPRG; endpoint_names identifies writeback ops.
hardware = Hardware()
prepared = PreparedFrontend.from_workload(workload, hardware)
result = predict_span(
    prepared,
    hardware.with_changes(rf_banks=8),
    endpoint_names=endpoint_names,
)
```

Preparation produces an immutable instruction snapshot and compact ABI input. It
caches no timed events, hardware state or result. Each prediction creates fresh
native state. Reuse checks hardware legality and block residency; changes to lane
count or LDS address classification revalidate the workload. Actual instruction
words can be mapped using `with_physical_operands` before preparation.

Pass `trace=True` to export per-instruction collect, operands-ready, issue,
writeback and ENDPRG events. Missing event kinds are omitted. Counters include
fetch packets/replays, duplicate RF grants, LSU states/returns, operation counts
and memory/stall counters. `max_cycles` is also available on the Python API.

The explicit frontend ABI is separate from legacy instruction ABIs. Supported
operations are scalar/vector, FADD/FMAX/FMA/FTOI/ITOF, FP32 Tensor, load/store and
ENDPRG. The native frontend rejects more than 32 lanes before creating timing
state; the measured calibration profile has 32 lanes. Collective template barriers, noncontiguous/branching streams and extended
compiled-provider operations are rejected. Omitted physical-slot metadata uses
tuple positions, with the same accuracy limitation as the Python model.

## Validation and timing evidence

`./run frontend-parity` compares the Rust implementation to the detailed Python
model, including all five event classes, endpoint spans and every named counter.
It reads frozen manifests without regenerating kernels. Failed cases and errors
remain in its output, and its process exits unsuccessfully if any case differs.
A mismatching result has no reported speedup.

The benchmark alternates Python/native call order, uses fresh hardware state and
records three repetitions. Full-call timing includes instruction encoding,
validation, repeated bridge/library access, FFI and result decoding. Prepared-call
timing excludes the separately measured encoding step, but still executes fresh
native hardware. The initial build/first call is recorded separately. Host timings
are observations, not stable throughput guarantees.

[Native parity and performance receipts](evidence/native-frontend-20261016/README.md)
record the frozen source and measured scope. The prior RTL calibration and its
failed thresholds remain unchanged; implementation agreement alone cannot improve
the reported 90%/95% accuracy coverage.
