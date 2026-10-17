# Native core mechanism repair — 17 October 2026

The repaired mechanisms now execute on the real compiled ELF path and workbench
evaluator. The user's priority is mechanism completeness and usable flow.
Saved-case errors are observations, not a tuning gate; no optimization run or
universal timing accuracy is claimed.

## Implemented behavior

- Compiled default `ventus-instruction-rust-v10-causal` uses current-edge RF
  grants, collector release, issue readiness and finite data-memory state.
  Duplicate grants survive stalls. Integer multiply has an independent elastic
  pipeline; integer division uses input-dependent IntDivMod/SFU service and
  active-lane groups. Unknown division operands are rejected.
- Instruction MSHRs remain occupied through all line responses and retirement.
  L1 primary/target entries, return queues and backpressure are finite. Bounded
  L2 directory/MSHR/SourceD stages replace unbounded future reservations.
  Local write acknowledgement differs from downstream completion; dirty lines
  and pending transactions drain before terminal completion.
- The pinned `Mem_SimWrapper` boundary has five response slots, old-state request
  admission, registered delay and eligible-slot response arbitration. Channel,
  bandwidth and latency overrides are explicit parametric extensions. Physical
  DRAM bank/row scheduling is not included.
- Native probe default `ventus-source-frontend-v4-causal` shares this data-memory
  controller. `predict_span(..., causal=False)` retains historical v3. Legacy
  compact ABIs retain old timing; newly generated programs carry unit/service/
  physical-slot metadata and select the causal path.
- Deterministic SM/set/line dirty-writeback selection fixes a host HashMap
  iteration bug that caused repeated sessions to diverge. This ordering is an
  assistant-selected abstraction, not exact RTL writeback arbitration. Existing
  repeated-dispatch regressions were retained without weakened expectations.

## Actual execution evidence

[Workbench evaluation](workbench-evaluation.json) executes the retained real
M11 N13 K9 ELF at **2,845 modeled cycles** and independently rechecks its saved
RTL logs: 143 exact outputs. It does not rerun RTL. Historical 2,864-cycle receipts
remain frozen with the earlier model; the difference is not a speedup claim.

The [new compiled diagnostic](subword/kernel.cl) covers multiply/divide, signed
byte/halfword loads, private narrow accesses, packed narrow stores, two warps,
two blocks and a tail. Its actual configured RTL job completed but produced
**31 wrong words**: 15/17 byte-output words and 16/34 halfword-output words.
All **134 32-bit integer result words** matched. The packed-store root cause is
unresolved; the whole program is not numerically correct. References were not
adjusted. Retained [summary](subword/summary.json), [stdout](subword/stdout.log),
[stderr](subword/stderr.log), [worker result](subword/worker-result.json),
[build](subword/build.json) and [ELF](subword/program.elf) identify this failure.

The RTL job's earlier frozen v10 worker reported 949 native cycles; its sources
are in `probe-worker-source-snapshot/`. A separate [current native execution](subword/current-native.json)
completes the same ELF at 992 modeled cycles, exercising both integer units and
1/2/4-byte accesses with exactly-once group/instruction/external accounting.
Current sources are in `worker-source-snapshot/`. These source identities differ.

[Probe replay](probe-replay.json) preserves 112 saved RTL trace identities,
old/new cycles, counters, first divergences and current native hashes. All are
development data, with mixed gains/regressions: old/new maximum error is
12.76%/17.11%. This is not an accuracy improvement claim. Full difference receipts
remain in the named ignored result paths with hashes in the compact export.
No further fitting follows.

## Remaining boundaries

Compiled supply is a dynamic decoded stream: cold I-cache, fetch/redirect and
complete launch timing are excluded. Probe fetch remains a finite hit-path model.
Full inclusive L2 probes, SRAM-port arbitration and complete SFU queue overlap
remain approximate. Workgroup barriers retain conservative timing fences.
Subgroup/atomic/system fences and extended precision lack baseline RTL validation.
Packed global byte/halfword stores have the concrete counterexample above.

The [machine-readable contract](mechanism-contract.json) is also emitted by the
instruction CLI. No DRAM simulator dependency was introduced: the immediate
target is the existing RTL wrapper. A physical-memory backend can later replace
external admission/response while retaining finite credits and transaction IDs;
it needs separate configuration and validation.

Internal checks cover Python/Rust regressions, formatting, historical archive
integrity and privacy. These are engineering checks, not research or user
acceptance. `manifest.json` hashes all exported files other than itself.
Earlier evidence directories are unchanged.
