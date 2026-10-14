# Calibration suite, part 4: B30, the queue between the warp scheduler and the load/store unit

*v0.1 · October 15, 2026 · Li Shang, drafted with Claude. Companion to `B27_plus.md`. One benchmark this round; B31–B34 later.*

## 0. Why

Three of the thirty G4 designs are 30 to 60 % wrong. The profiler says they sit in `mio_throttle`: the queue from the warp scheduler to the load/store unit is full. B27 and B28 saw the symptom (the shared-load rate rises with occupancy and with burst length, gaps 9 to 11) and the warp model has no such queue. B30 measures it.

Same conventions as B13+: one CUDA program, JSON lines, cycles beside seconds, `driver.py --micro3`, a fitter in `fit_micro.py`, provenance in the contract, `ncu` cross-checks, no address or key in the repository.

## 1. B30 `b30_mio_queue` — depth, drain rate, cost per load, sharing

**Measures.** Four things about the queue between a sub-partition's scheduler and the LSU.

1. **Depth per sub-partition.** One warp alone on the SM issues N back-to-back independent `ld.shared.32` (streaming addresses, nothing between them), N = 1..64. `clock64` before the first and after the last issue. Issue costs about a cycle per load while the queue has room; when it is full the warp stalls until an entry drains. The N where cycles per issued load jumps is the depth. Repeat with two warps on the same sub-partition (block of 8 warps: warps w and w+4 share one): if their bursts stall at half the N, the queue is per sub-partition, not per warp.
2. **Drain rate, wavefronts per cycle.** Sixteen warps per SM (four per sub-partition) run bursts of 32 loads each with the results consumed after the burst, so the queue is always full. Widths 32, 64, 128 bits; word strides 1, 2, 4, 8 (B6's conflict degrees). Instructions per cycle from `clock64`; wavefronts per instruction from `ncu l1tex__data_pipe_lsu_wavefronts_mem_shared` on the same kernel. Drain = instructions per cycle × wavefronts per instruction, per SM and per sub-partition.
3. **Cost per load.** The wavefronts per instruction table of item 2, by width and conflict degree. Expected: 32-bit stride 1 = 1 wavefront, 64-bit = 2 (half-warp rule, B26.6), 128-bit = 4 (quarter-warp), times the conflict degree.
4. **Sharing.** A warp issues 32 `ld.shared` then 8 `cp.async` (16 B) chunks; the issue time of the 8 chunks with and without the preceding burst says whether `cp.async` waits in the same queue. Same with 8 `ldmatrix.x4`. Also: issue cycles per `cp.async` and per `ldmatrix` alone.

**Pass criterion.** Item 2 at 32-bit stride 1 reproduces B21's `lds_32` rate within 10 %. Item 1 shows a clear jump (cycles per load at least 3× the pre-jump value). Item 4's `cp.async`-alone cost reproduces B15.3 within 20 %.

**Contract.** `mio_queue_depth_per_smsp` (instructions), `mio_queue_scope` (`per-smsp` or `per-warp`), `mio_issue_cycles_per_load_in_queue` (cycles per load while the queue has room), `lsu_drain_wavefronts_per_cycle_per_sm`, `lds_wavefronts_per_instruction` (by width and stride), `lds_instr_per_cycle_per_sm_by_width_stride`, `mio_shared_with_cp_async` (bool), `mio_shared_with_ldmatrix` (bool), `cp_async_issue_cycles_per_chunk` (confirms B15.3), `ldmatrix_issue_cycles`.

**Spark time.** 3 min. **ncu.** `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`, `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio`, `sm__inst_executed_pipe_lsu.sum`.

## 2. What the warp model does with it

`sim/warp.py`: each sub-partition owns a queue of `mio_queue_depth_per_smsp` entries. An `ld.shared` (or a `cp.async` chunk, if B30 says they share) is eligible only if its sub-partition's queue has a free entry; issuing it takes `mio_issue_cycles_per_load_in_queue`. The LSU, shared by the SM, drains entries in order at `lsu_drain_wavefronts_per_cycle_per_sm`; an entry costs its wavefronts (width × conflict degree). The data is ready `scoreboard_latency_cycles.lds` after the entry drains. The per-residency LSU rate table and `lsu_queue_instructions` are removed. Nothing is tuned: B27, B28 and B29 become predictions.

## 3. Build order

| Step | Deliverable | Acceptance |
|---|---|---|
| 1 | B30 written, run, fitted | its own pass criteria |
| 2 | queue in `sim/warp.py`, table removed | the model runs with contract parameters only |
| 3 | B27, B28, B29 rerun as predictions | printed; misses to `notes/sim_gaps.md` |
| 4 | G4 on the same thirty designs | every design within 20 %, median within 10 %, agreement on at least half of the copy/barrier designs |
| 5 | report | what B30 measured, the thirty-design table before and after, the gap-log diff |

Do not start G5 unless step 4 passes.
