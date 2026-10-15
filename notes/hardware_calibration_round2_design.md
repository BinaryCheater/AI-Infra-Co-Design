# Calibration, round 2: the paths the GEMM template never touched

*Design note, v0.1 · October 15, 2026 · Li Shang, drafted with Claude. For Claude Code. Follows B1–B30; same format as `B13_plus.md` and `B27_plus.md`.*

---

## 0. Where we are

The FP32 GEMM path through shared memory is measured end to end: issue, scoreboard, the LSU queue, banks, `cp.async`, barriers, L2, DRAM. The model predicts the thirty test designs with a median error of 8 %. Every number in that path came from a benchmark, and the model was checked against kernels it had not seen.

Three paths on the chip have not been touched, and the next two templates need them:

1. **The tensor core.** We have its peak rate and the `ldmatrix` feed. We do not have how many `mma.sync` a warp must keep in flight, what an `ldmatrix`-to-`mma` dependency costs, or how `mma` and FFMA share the schedulers. Any real GEMM uses BF16 tensor ops.
2. **TMA under load.** One measurement: one block, one tile size, 500 ns. The `cp.async` path was wrong by 2–5× until it was measured with six resident blocks. TMA will be the same.
3. **What attention needs.** Softmax on the SFU pipe, warp shuffles, converts, and the register file as a budget. None measured.

This round is B31–B40. Same rules: one kernel per benchmark under `calibration/suite/bench/`, cycles beside seconds, a counter cross-check for every timing number, the result in the contract with its raw record, nothing fitted. Each group ends with a composite kernel the model must predict without a parameter change.

The Chapter 9 items of `gb10_microarchitecture.md` this closes: 8 (tensor operand path) in full; 1 (why the LSU queue backs up) partly, through the `ldmatrix` and shuffle tests; 3 (register-file bank conflicts) is still out of reach.

---

## 1. The tensor core path (B31–B34)

**Feeds:** a new `sim/tensor.py`: the tensor pipe as a resource per sub-partition with an issue cost, a latency, and an operand path from shared memory through `ldmatrix`. The warp model's scoreboard gains `mma` and `ldmatrix` as instruction classes.

### B31 `b31_mma_issue` — issue rate and latency per shape

**Kernel.** One warp per sub-partition. (a) A dependent chain: the accumulator of one `mma.sync` feeds the next, 1000 deep, for `m16n8k16` BF16, FP16, `m16n8k8` TF32, `m16n8k32` FP8 e4m3 and INT8. (b) Independent: `a` accumulators, `a` = 1, 2, 4, 8, 16, round-robin. (c) Same as (b) with 2 and 4 warps per sub-partition.

**Measures.** Latency per shape (cycles, from the chain). Issue cost per shape (cycles between back-to-back independent `mma` from one warp). The number of independent `mma` a warp needs in flight to reach the pipe's rate, and how much a second warp substitutes.

**Counter check.** `sm__pipe_tensor_cycles_active` and `sm__inst_executed_pipe_tensor`; rate at saturation must match B20's per-shape TFLOP/s within 3 %.

**Pass.** Latency 29 ± 2 cycles for FP16 `m16n8k16` (B20); the (a, w) grid is monotone and the saturated rate equals B20.

**Contract.** `mma_latency_cycles[shape]`, `mma_issue_cycles[shape]`, `mma_in_flight_to_saturate[shape]`.

### B32 `b32_ldmatrix_mma` — the operand path

**Kernel.** A warp loads fragments with `ldmatrix.x4` from a pre-filled, swizzled shared tile and feeds `mma.sync`. (a) Dependent: each `mma` waits for its own `ldmatrix`. (b) Pipelined: fragments for step k+1 loaded during step k, double-buffered in registers, depth 1, 2, 3. (c) Unswizzled tile, same loop. (d) With 1, 2, 4 blocks per SM.

**Measures.** `ldmatrix`-to-`mma` dependency latency. The pipeline depth at which the loop reaches the `mma` rate. The cost of the unswizzled layout inside a real loop (B20 measured 8× on `ldmatrix` alone). Whether the operand path saturates per SM or per sub-partition.

**Counter check.** `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldsm`, `smsp__warp_issue_stalled_mio_throttle`, `..._short_scoreboard`, `sm__pipe_tensor_cycles_active`.

**Pass.** (a) gives `ldmatrix` latency + `mma` latency within 10 %; (b) reaches ≥ 90 % of B31's rate at some depth ≤ 3; (c) and (d) reported.

**Contract.** `ldmatrix_latency_cycles`, `ldmatrix_wavefronts[x1,x2,x4]`, `mma_pipeline_depth_to_saturate`, `tensor_loop_efficiency[swizzled,unswizzled]`.

### B33 `b33_mma_mio` — the tensor path and the LSU queue

**Kernel.** The B32 pipelined loop with added traffic on the same SM: (a) another block doing `LDS.128` loads, (b) another block doing `cp.async`, (c) both. 1 + 1, 2 + 2, 3 + 3 blocks.

**Measures.** How much `ldmatrix` slows behind shared loads when the tensor pipe is consuming (B30 saw 12 → 36 cycles with loads alone, no consumer). Whether `mma` issue itself is ever blocked by `mio_throttle`.

**Counter check.** `mio_throttle` per warp by block, `sm__pipe_tensor_cycles_active`.

**Pass.** Reported; the number goes to the shared LSU-entry model as one more instruction class with its wavefront cost.

**Contract.** `ldmatrix_behind_lds_cycles[residency]`, `tensor_rate_under_lds_load[residency]`.

### B34 `b34_mixed_issue` — FFMA and `mma` from the same schedulers; accumulator pressure

**Kernel.** (a) One warp alternating `mma.sync` and independent FFMAs in ratios 1:0, 1:1, 1:4, 1:16. (b) A tensor GEMM inner loop with 64, 128 and 192 accumulator registers per thread, compiled with `-maxrregcount` at 255, 128, 96. (c) The same loop with the epilogue (FP32 accumulators to BF16, stored) included.

**Measures.** Whether FFMA issue slots are free while the tensor pipe runs (dual use of the scheduler) or compete. The spill cliff for tensor kernels. The epilogue's share of a tile's time.

**Counter check.** `smsp__issue_active`, `sm__pipe_fma_cycles_active`, `sm__pipe_tensor_cycles_active`, `l1tex__t_sectors_pipe_lsu_mem_local`.

**Pass.** (a) 1:0 equals B31; (b) 255 equals B32; the rest reported.

**Contract.** `mma_ffma_overlap` (fraction of FFMA slots free during `mma`), `tensor_spill_curve`, `epilogue_cycles_per_tile[shape]`.

**Composite check for the group.** A BF16 tensor GEMM tile kernel (128×128×32, `ldmatrix` + `mma.sync` m16n8k16, swizzled, double-buffered), generated by the tensor template once it exists, otherwise hand-written in the suite, at 1, 2, 4 blocks per SM. The model with B31–B34's numbers must predict cycles per k-step within 10 % at every residency and name the right dominant stall family. A miss is a gap-log entry.

---

## 2. TMA under load (B35–B37)

**Feeds:** `sim/memory.py`'s copy mechanism, which today has TMA as a one-block latency. It becomes a second copy engine on the shared per-SM path with its own completion rule.

### B35 `b35_tma_residency` — completion by residency and tile size

**Kernel.** B8's `cp.async.bulk` + mbarrier kernel, 1-D, tiles 4, 8, 16, 32, 48 KB, at 1, 2, 4, 6 resident blocks per SM, 10 000 trials each, per-trial issue-to-completion cycles into a histogram.

**Measures.** p1/p50/p99/max completion per tile and residency; per-SM and chip throughput by residency.

**Counter check.** `lts__t_sectors_srcunit_tex_op_read` (bytes through L2), `smsp__warp_issue_stalled_barrier` for the mbarrier wait.

**Pass.** Resident = 1, 16 KB p50 equals B8's 500 ns within 10 %.

**Contract.** `tma_latency_dist[tile,resident]`, `tma_bytes_per_s_per_sm_by_residency`.

### B36 `b36_tma_contention` — TMA with `cp.async` and with shared loads

**Kernel.** Per SM: (a) TMA blocks and `cp.async` blocks in ratios 1:1, 1:3, 3:1 at 4 resident; (b) TMA blocks alongside `LDS.128` loops; (c) 2-D tensor-map copies instead of 1-D; (d) TMA store (shared to global) alongside TMA load.

**Measures.** The shared path: B15 saw 1.67 TB/s mixed against 1.96 and 1.90 alone. Whether TMA issue (one thread) is ever queued behind shared loads at the LSU entry. 2-D vs 1-D cost. Load/store interference.

**Counter check.** `l1tex__throughput`, `mio_throttle`, `lg_throttle`.

**Pass.** (a) at 1:1 reproduces 1.67 TB/s within 10 %.

**Contract.** `tma_cp_async_mixed_bytes_per_s[ratio]`, `tma_behind_lds_cycles`, `tma_2d_vs_1d_ratio`, `tma_store_bytes_per_s`.

### B37 `b37_mbarrier_cluster` — mbarrier costs and cluster multicast

**Kernel.** (a) `mbarrier.arrive` and `try_wait` costs with 1, 32, 256 arriving threads and with `expect_tx` pending. (b) Phase flip latency. (c) `__cluster_dims__(2,1,1)` and `(4,1,1)`: a TMA load with `.multicast::cluster` into both blocks' shared memory, bytes through L2 per block vs non-multicast. (d) DSMEM read bandwidth and latency between the two blocks of a cluster.

**Measures.** mbarrier cost by arrival count (B15 has 46 ns for 256 threads, no breakdown). Whether multicast halves L2 traffic on `sm_121`. DSMEM as a data path.

**Counter check.** `lts__t_sectors` per block for (c); `smsp__warp_issue_stalled_barrier`.

**Pass.** (a) at 256 threads equals B15 within 10 %; (c) reports multicast supported or not with the traffic ratio.

**Contract.** `mbarrier_cycles[arrivals]`, `tma_multicast_supported`, `tma_multicast_l2_traffic_ratio`, `dsmem_latency_ns`, `dsmem_bytes_per_s`.

**Composite check for the group.** The GEMM T5 design generated with TMA instead of `cp.async` (the template already has a TMA flag), run at 1, 2, 4 resident blocks on the Spark; the model must predict within 10 % and the copy wait family must agree with the counters.

---

## 3. What attention needs (B38–B40)

**Feeds:** the warp model's instruction classes (`sfu`, `shfl`, `cvt`), the LSU-entry model (if shuffles go through it), and `sim/warp.py`'s register budget. The attention template is not written yet; these numbers are what its cost model will be built from.

### B38 `b38_sfu_convert` — SFU and conversion pipes

**Kernel.** B2's pattern (ILP 8, 8 blocks per SM) for `ex2.approx`, `rcp.approx`, `rsqrt.approx`, `lg2.approx`, `tanh.approx`, `F2F` FP32→BF16, `F2FP` packed, `I2F`. Dependent chains for latency. Mixed with FFMA at 1:1, 1:4, 1:16 to see whether the SFU and FMA pipes overlap.

**Measures.** Throughput and latency per op; overlap with FFMA. Softmax is `max`, `sub`, `ex2`, `add`, `rcp`, `mul`: the SFU rate bounds it.

**Counter check.** `sm__pipe_xu_cycles_active` (the SFU), `sm__pipe_fma_cycles_active`, `sm__inst_executed_pipe_xu`.

**Pass.** `MUFU.RSQ` throughput equals B21's 0.15 of FFMA within 10 %.

**Contract.** `sfu_throughput_per_sm[op]`, `sfu_latency_cycles[op]`, `sfu_ffma_overlap`, `cvt_throughput_per_sm[op]`.

### B39 `b39_shuffle_reduce` — warp shuffles and reductions

**Kernel.** (a) `shfl.sync` `bfly`, `up`, `idx` latency (dependent chain) and throughput (independent, ILP 8). (b) A full 32-lane butterfly reduction (5 shuffles + 5 FADD). (c) Shuffles running alongside `LDS` loads from another warp on the same sub-partition: does `shfl` wait behind loads (it may use the MIO path). (d) `redux.sync` if it compiles on `sm_121`. (e) Block-wide reduction through shared memory for comparison.

**Measures.** The row-max and row-sum of online softmax are reductions; this gives their cost and tells the LSU model whether shuffles are in its queue.

**Counter check.** `smsp__inst_executed_op_shfl`, `mio_throttle`, `l1tex__data_pipe_lsu_wavefronts_mem_shared` (if shuffles count as wavefronts, they are in the LSU).

**Pass.** (b) equals 5 × shuffle latency + 5 × 4 within 15 %.

**Contract.** `shfl_latency_cycles[mode]`, `shfl_throughput_per_sm`, `shfl_in_lsu_queue` (bool), `warp_reduce_cycles`, `redux_supported`.

### B40 `b40_register_budget` — the register file as a budget, and spills under load

**Kernel.** (a) An attention-shaped inner loop (Q·Kᵀ tile, online softmax, P·V tile) hand-written with 96, 128, 160, 200, 255 registers per thread by varying the tile, compiled at each `-maxrregcount`; spill bytes from `ptxas -v`. (b) Local-memory throughput with 4, 8, 16 warps spilling at once (B19 measured one). (c) Residency vs registers from the occupancy API cross-checked by measurement (B9 did 20 registers only).

**Measures.** The spill cliff for a loop with 3 live tiles, the local-memory path under contention, and the full register-residency table.

**Counter check.** `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st`, `sm__warps_active`.

**Pass.** (a) at 255 shows no spills; (c) matches the occupancy API exactly.

**Contract.** `attention_spill_curve`, `local_mem_bytes_per_s_per_sm_by_warps`, `max_blocks_per_sm_by_regs[regs]`.

**Composite check for the group.** A single-block online-softmax kernel (one row block of 128 keys, FP32, shuffles for the row max and sum, `ex2` on the SFU, BF16 convert on the way out), at 1, 2, 4, 6 blocks per SM. The model with B38–B40's numbers must predict within 15 % (looser than GEMM: three pipes interact) and name the dominant pipe.

---

## 4. Build order

| Step | What | Spark time | Done when |
|---|---|---|---|
| 1 | B31–B34, contract, `sim/tensor.py` | 15 min | tensor composite within 10 % at 1, 2, 4 blocks, nothing fitted |
| 2 | B35–B37, TMA as a second engine on the shared copy path | 20 min | T5-with-TMA within 10 % at 1, 2, 4 blocks |
| 3 | B38–B40, new instruction classes and the register budget | 15 min | softmax composite within 15 % at 1–6 blocks |
| 4 | `gb10_microarchitecture.md`: Chapter 3 gains the tensor path, Chapter 5 gains TMA under load and shuffles, a new Chapter 8a for the SFU; Chapter 9 rewritten as "measured for GEMM and attention; open: …" | — | every new number has its benchmark in brackets |

About an hour of Spark time in all; the writing is the cost, as before. Groups are independent; if time is short, do group 1 first, since the tensor template is next.

Two cautions. The composites are checks, not calibration: if one misses, the gap goes to `notes/sim_gaps.md` with the counter that shows the cause, and no number is adjusted. And the attention composite is the first kernel where three pipes (FMA, SFU, LSU) interact in one loop; expect the first miss there, and expect it to name a queue we have not drawn yet.

---

## 5. Instruction for Claude Code

> Read `notes/hardware_calibration_round2_design.md`. Implement B31 to B40 as `calibration/suite/bench/b31_*.cu` … `b40_*.cu` in the B13+ style, with `driver.py --round2`, fitters in `fit_micro.py`, and the new fields in `schema/chip.json`. Run them on spark-6951 with the counter cross-checks and put the results in the contract with raw records. Then, group by group: add `sim/tensor.py`, TMA as a second engine on the shared per-SM copy path in `sim/memory.py`, and the `sfu`, `shfl`, `cvt` instruction classes and the register budget in `sim/warp.py`, every parameter from the contract. Run the three composite kernels on the Spark and check the model against them as predictions, nothing fitted; a miss goes to `notes/sim_gaps.md` with the counter that shows why. Update `gb10_microarchitecture.md` chapters 3, 5 and 9 with the new numbers, each with its benchmark in brackets. Report in plain language: the contract diff, the three composite tables, the gap-log diff. No address or key in the repository.
