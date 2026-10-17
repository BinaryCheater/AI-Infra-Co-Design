# Private spill layout and driver correction — 17 October 2026

The upstream R1 host driver and LSU disagree about the unit of the private base.
The driver advances that base by private-bytes/thread times lanes for each warp.
The LSU already interleaves private words across the whole workgroup and adds the
warp/lane thread ID. Combining both transformations can alias private storage.
[Pinned source derivation](source-audit.json) records the exact files and lines;
the pinned `MemBox` harness already supplies zero per-warp base increment.

This was visible in an earlier compiled GEMM that happened to produce correct
outputs. Its private external line offsets were 0/512/768/1280/1536/2048, while
the old model used 0/128/768/896/1536/1664 after renaming the private base. Matching
ELF and finite outputs therefore did not establish a matching private footprint.
Earlier v9-4 compiled receipts remain historical evidence, with this limitation.
Their private-locality predictions cannot be treated as validated against that
upstream driver. The earlier straight-line calibration is a separate dataset.

A new real compiler diagnostic gives every lane three volatile private words,
stores distinct integer values, crosses a uniform workgroup barrier and reloads
them. The compiler declares 20 private bytes/thread including other spills.
With 2 warps/block and 2 blocks, the original driver finishes but **32 of 384
outputs are wrong**. [Original result](upstream/worker-result.json) is retained as
a failed workflow. This source does not intentionally communicate through private
storage, so the alias is an implementation/interface defect.

The explicit `workgroup-private-v1` driver profile corrects one host input:
`host_pds_size_per_wf=0`. An independently built, named shared-library adapter
applies it before the existing Verilated model evaluates a host request. The
original RTL library/driver executable and historical results remain intact.
This is an active host-input adapter, not a passive performance observer. Its
source, generated header, compiler, binary and target-library hashes are recorded;
a unique stderr marker confirms loading/application. If a separate observer is
requested, the adapter precedes it in `LD_PRELOAD`.

The same ELF now completes with **384/384 exact outputs** on the corrected
driver. The original and corrected RTL-library hashes are identical. The corrected
[worker result](corrected/worker-result.json) and raw logs remain separate from
the original failed run. No timing coefficient was fitted or RTL core rebuilt.
[Identity comparison](target-comparison.json) also confirms identical RTL executable
and initialized metadata/data bytes. Both actual worker source snapshots are frozen
alongside their code hashes; private transport paths are excluded from this export.

Native v9-5 adds explicit private base/warp-increment launch fields. The matching
`r1-workgroup` profile uses the actual 0xA0000000 private base and zero increment.
The prior generic `model-workgroup` profile retains its symbolic 0x60000000 base
and is labeled in receipts. Unsupported upstream multiwarp private increments
fail admission because they cannot guarantee lane isolation. Private accesses
also require their whole byte span to fit the compiler allocation. Old launch
tails remain readable; changed runtime identities prevent checkpoint confusion.
Subword private usage reserves complete interleaved words; the compiler's usage
and rounded bytes/thread allocation remain distinct.

Evaluator v3 requires a matching corrected-driver reference and rechecks its
adapter identity/marker with raw numerical evidence. Its native program uses
`r1-workgroup`; reference profiles cannot be silently interchanged. A fresh bounded
[runtime call](workbench-runtime.json) passes for the diagnostic. Workbench calls
still report modeled timing, independently from numerical RTL validation.

Two setup failures are preserved: the Verilator version output lists both a stale
build default and the actual installation, and the first relative-output CLI run
resolved its adapter destination twice. Current discovery checks reported existing
headers, honors an explicit configured root, and resolves output paths once.
These were setup failures, without a successful RTL claim; see
[failure records](setup-failures.json).

The seven existing vector/exchange/GEMM cases also passed fresh corrected-driver
RTL checks, covering 2,629 exact output words. The first batch exhausted its
240-second SSH transport budget after six cases. Its process was confirmed absent;
that batch remains [incomplete](regression/first-six/partial-result.json). Only the
unstarted selected GEMM ran in a new independent job. [Raw-reference rechecks](regression/recheck.json)
keep these two runs distinct; all seven [bounded evaluator v3 calls](regression/workbench-runtime-v3.json)
pass. This totals 3,013 numerical outputs including the new private diagnostic,
without establishing a universal or timing-accuracy result.

[Private line comparison](footprints.json) confirms the selected GEMM's corrected
native/private external-bus line sets now agree. This checks coverage for one
frozen input, not per-lane event order, cache timing or complete LSU behavior.
Current source hashes are [recorded separately](current-code-hashes.json) from
each actual frozen execution snapshot.

Fresh execution uses the commands in [compiled workflow documentation](../../compiled-workflow.md).
The source diagnostic is `examples/kernels/private_spill_u32.cl`, with a real ELF
fixture in `tests/fixtures/compiled-private-r1/`. Focused regression replay:

```bash
.venv/bin/pytest -q tests/test_ventus_private_memory.py
.venv/bin/python -m codesign.ventus.compiled_check prepare --manifest examples/compiled-private-check.json --out results/private-input-NEW
```

The original failure can be replayed on the configured RTL target with the frozen
`upstream-source-snapshot` module and `work/input/cases.json`; that historical
module uses the upstream host driver without the correction. Use a fresh output
directory and explicit configured RTL/spdlog paths, as in the workflow commands.
Current evaluation intentionally rejects an upstream-driver reference substituted
for the corrected profile.

Full generic research-loop integration, timing/accuracy targets, broader kernels,
subgroup/atomic behavior and physical feasibility remain open. Finite output and
address checks do not establish universal correctness or user acceptance.
