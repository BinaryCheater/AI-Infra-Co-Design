# Compiled launch resource admission — 17 October 2026

Compiled ELF execution and RTL input preparation now use
`ventus-compiled-launch-contract-v1`. Reviewed R1 source declarations bind the
baseline capacities: 8 warp slots, 8 block slots, 1,024 vector-register slots,
2,048 scalar-register slots and 128 KiB LDS per SM, with 32 lanes and 4 RF banks.
[Source audit](source-audit.json) records exact pinned file hashes and lines.
This checks reviewed source expressions; it does not elaborate arbitrary Scala
or establish that an arbitrary simulator binary has those parameters.

Register usage is rounded to whole bank rows and charged for every warp in a
workgroup. Static LDS plus every warp's scalar spill stack must fit the allocated
LDS. Impossible blocks and inconsistent lane/bank allocations fail before native
instruction expansion/scheduling and before RTL metadata is emitted. A rejected
launch leaves an existing native session usable and unchanged. Earlier native
scheduler assertions caught some capacity failures; the new check provides a
shared explicit contract and receipts, including the previously unguarded RTL path.

The RTL-specific contract also checks a complete full-warp grid, the declared
1,024 workgroups/axis envelope and 4,096 private bytes/thread. The latter follows
the per-wavefront interface maximum and the driver's multiplication by 32 lanes.
The generic model's capacity checks remain distinct from the pinned RTL envelope.
The checker's separate 1 MiB diagnostic backing limit still applies.

The [failure replay](failure-replay.json) injects an oversized resource declaration
into a real unchanged ELF's launch. The frozen previous loader admits 1,028 vector
slots for one warp against a 1,024-slot SM pool and emits metadata; the new loader
rejects it without emitting executable input. This is a synthetic negative fixture,
not a hardware run. Other regression fixtures exercise full-block register
charging, LDS stacks, interface limits and invalid hardware lane/bank choices.

[Seven real compiled cases](recheck.json) pass the new admission. Prepared RTL
metadata/data hashes equal their original bytes exactly; their raw recorded RTL
completion/output checks were re-evaluated read-only, covering 2,629 output words.
The original references and source snapshots remain unchanged. RTL was not rerun.

A fresh bounded [workbench evaluator call](workbench-runtime.json) passes for
the selected 11x13x9 GEMM. Its receipt reports 72 vector-register slots, 64 scalar
slots and 768 LDS bytes per workgroup, and a static upper bound of 4 resident
workgroups/SM. Modeled cycles remain 2,864 with 3,259 instructions and 24 barrier
arrivals. A capacity upper bound does not establish achieved occupancy, allocator
timing, fragmentation or physical feasibility. Timing coefficients and native
v9-4 semantics are unchanged; the Python runtime hash includes the new contract.

Reproduce the negative boundary and source checks from this checkout:

```bash
.venv/bin/pytest -q tests/test_ventus_launch_contract.py
.venv/bin/python -m codesign.ventus.compiled_check prepare --manifest examples/compiled-gemm-check.json --out results/resource-check-NEW
```

For the fresh workbench call, use the evaluator replay command in the
[previous concrete-binding evidence](../compiled-bindings-20261017/README.md)
with a new output directory. Its execution/program metadata now includes
`resource_admission`. The new [code hashes](code-hashes.json) identify this change.
Full static parameter/ISA audit, arbitrary hardware realizations, generic research
loop integration and timing accuracy remain open. No search or acceptance occurred.
