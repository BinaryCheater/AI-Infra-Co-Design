# Calibration suite, part 2: B13–B26

**Microarchitecture-level functional and timing features of the GB10 that the mechanistic simulator needs and that can be measured from user space.**
*v0.1 · October 15, 2026 · Li Shang, drafted with Claude. Companion to `notes/hardware_calibration_design.md` (B1–B12) and `notes/mechanistic_simulator_design.md` (M1–M11). Written for Claude Code to execute.*

---

## 0. Why now, and what order

B1–B12 measured the *rates and latencies* of the GB10. The simulator also needs its *structure*: how many slices the L2 has and how addresses map to them, how deep the copy queue is, in what order the block scheduler fills 48 SMs, how many warps it takes to hide DRAM latency, how LPDDR5X interleaves channels. None of that is in the contract, and the two known misses (4096³ at −33 %, skinny at +67 %) depend on exactly these: L2 reuse across concurrently resident blocks, and scheduler order plus latency hiding on a one-wave grid.

The original plan was to let G2/G3 fail and let the misses order the measurements. For the parameters below we are already confident they bite, measuring them costs about half an hour of Spark time, and having them before G2 means the mechanisms are built once. So: **run B13–B26 before G0**, extend the contract, then start the simulator.

Same conventions as B1–B12: one CUDA program per benchmark under `calibration/suite/bench/`, JSON lines `{quantity, unit, samples, detail}`, `driver.py` repeats to the 4 % interval target or 50 reps, `fit.py` writes the contract with method strings and raw links, no address or key in the repository. Where the clock cannot be locked, every timed quantity is reported in **cycles** (from `clock64` against `%globaltimer`, as the runner now does) as well as seconds. New flag: `remote.sh <alias> --micro` runs B13–B26.

**Prerequisite P0: profiler counters.** `ncu` on spark-6951 fails with `ERR_NVGPUCTRPERM`. Fix (needs sudo on the Spark, one-time): create `/etc/modprobe.d/nvidia-profiling.conf` containing `options nvidia NVreg_RestrictProfilingToAdminUsers=0`, then `sudo update-initramfs -u` and reboot. Record in `preflight.md` whether this was done. Several benchmarks below have an `ncu` cross-check column; they still run without it, but the cross-check is what catches a misread.

---

## 1. Benchmarks

Each entry: what it measures · mechanism it calibrates · kernel sketch and procedure · pass criterion (what a correct run must show) · contract fields · Spark time · `ncu` cross-check · reference.

### B13 `b13_l2_structure` — L2 slices, address hash, associativity, replacement, sector size

**Mechanism:** M5 (L2). Today we have only the 16 MiB knee from B4. The simulator needs sets × ways × slices, the hash from address to slice, the replacement policy, and the fill granularity.

**Procedure (four sub-tests, all single-thread or one-warp pointer chases so that latency, not bandwidth, is what is read):**

1. **Sector and line size.** Chase through a 2 MiB footprint (L2-resident, L1-missing) with strides 32, 64, 128, 256 B. Then measure DRAM bytes per request for a 1 GiB footprint at the same strides (via `ncu dram__bytes.sum`, or without ncu, from B4-style bandwidth per *request*). The stride at which bytes-per-request stops growing is the fill granularity (expected 32 B sectors in 128 B lines).
2. **Set conflicts → ways and sets.** Fix the footprint at 12 MiB (inside effective L2) and chase over `n` addresses spaced by a power-of-two stride `s`, `s` from 4 KiB to 8 MiB, `n` from 2 to 64. For each `s`, the smallest `n` at which latency jumps from the L2 value (136 ns) to the DRAM value (300 ns) is the associativity *as seen at that stride*; the stride at which that jump first occurs reveals the set-index bits. Report the (`s`, `n`) grid and the inferred (sets, ways).
3. **Slice hash.** Latency of a one-warp chase between pairs of addresses that differ in exactly one address bit, bits 7–31, at a 1 MiB footprint. Bits whose flip changes latency by a step are slice-select bits (near vs far slice; on a two-partition L2 the far-partition penalty is tens of ns). Also run with 48 blocks each chasing in its own slice-candidate region, measuring aggregate bandwidth: a wrong slice assumption shows as bandwidth collapse at particular strides. Report the bit mask and the near/far latency pair.
4. **Replacement policy.** Cyclic chase over a footprint `f` from 0.5× to 2× effective L2, both sequential-cyclic and random-cyclic order. Pure LRU gives a hit rate that drops to zero once `f` > capacity for the cyclic pattern; a random or protected policy keeps a fraction of hits. Report the hit-rate curve for both orders and classify as `lru`, `random`, `plru`, or `other`.

**Pass criterion:** sub-test 1 gives a consistent (sector, line) pair across strides; sub-test 2's inferred `sets × ways × line` lies within 10 % of `l2_bytes_effective` (16 MiB) — if it lands near 24 MiB instead, the "effective" capacity in B4 was a hash artifact, and that is a finding to record; sub-test 4 classifies the policy with the two curves differing by more than the CI.

**Contract:** `l2_line_bytes`, `l2_sector_bytes`, `l2_sets`, `l2_ways`, `l2_slices`, `l2_slice_hash_bits` (list), `l2_latency_near_far_ns` (pair), `l2_replacement` (enum, with the hit-rate curves in raw).

**Spark time:** 10 min with repetitions. **ncu:** `lts__t_sectors_op_read_hit_rate`, `dram__bytes.sum`. **Ref:** Mei & Chu 2017 [4] (set/way probing), Jia et al. 2018 [1] (slice and sector), Accel-Sim tuner [3].

### B14 `b14_l1_carveout` — L1 capacity, line size, latency vs shared-memory carve-out; `cp.async` L1 bypass

**Mechanism:** M5 (L1), M3 (smem capacity), M4 (copy path).

**Procedure:**
1. For each `cudaFuncAttributePreferredSharedMemoryCarveout` in {0, 25, 50, 75, 100} and for the dynamic-smem request that forces each resident-block count from B9: pointer chase over footprints 4 KiB–256 KiB with stride 128 B; the knee is L1 capacity at that carve-out. Report capacity and hit latency per setting.
2. Line size: same chase at strides 32–256 B at a 16 KiB footprint.
3. **Does `cp.async.cg` allocate in L1?** Copy a 32 KiB tile global→smem with `cp.async.cg`, then immediately chase the same global addresses with `ld.global.ca`; L1-hit latency means it allocated. Repeat with `cp.async.ca` and with `cp.async.bulk` (TMA). This is a *functional* property the simulator's copy path needs.
4. **Unified L1/smem total:** sum of measured L1 capacity and configured smem per setting; should be constant (the carve-out pool size).

**Pass criterion:** the L1 + smem sum is constant across settings within one line; `cp.async.cg` shows no L1 allocation (expected) and `.ca` does.

**Contract:** `l1_bytes_by_carveout` (map), `l1_line_bytes`, `l1_latency_ns`, `l1_smem_pool_bytes`, `cp_async_cg_allocates_l1` (bool), `cp_async_ca_allocates_l1`, `tma_allocates_l1`.

**Spark time:** 3 min. **ncu:** `l1tex__t_sector_hit_rate`. **Ref:** [1], Luo et al. 2024 [2].

### B15 `b15_copy_queue` — in-flight copy groups, issue stalls, completion-time distribution, TMA/`cp.async` contention, mbarrier cost

**Mechanism:** M4. The prefetch-distance decision (`P`) couples directly to how many groups can be pending, and the race probability in the unsound design depends on the *distribution* of completion times, not the median B8 reports.

**Procedure:**
1. **Queue depth.** One block per SM, 256 threads. Issue `k` tile copies (8 KiB each) and commit each as its own group, then `wait_group 0`. Record with `clock64` the time to *issue* group `k` (before the wait). The `k` at which issue time jumps is the per-warp pending-group limit; repeat with 1, 2, 4, 8 blocks per SM to find whether the limit is per warp, per block or per SM.
2. **Completion-time distribution.** Single tile, 10 000 trials, per-trial issue→`wait_group 0` cycles into a histogram (bins of 50 cycles); report p1, p50, p99, max, for tile sizes 4–48 KiB and for 1 vs 6 resident blocks per SM. The p1/p99 spread is what the simulator's copy latency must be drawn from (or at least bracketed by) when it is asked for the race window.
3. **Contention.** Half the blocks on each SM use `cp.async`, half use `cp.async.bulk` (TMA), same bytes; compare aggregate throughput with the all-`cp.async` and all-TMA cases. A sum below either pure case means a shared path.
4. **TMA descriptor and mbarrier.** `cuTensorMapEncodeTiled` for a 2-D tile; latency of a 2-D TMA copy vs the 1-D bulk copy from B8; cost of `mbarrier.arrive` + `try_wait` with no copy pending (barrier overhead alone), 256 threads.
5. **Issue cost.** Cycles per `cp.async` 16 B instruction per thread (the per-instruction issue cost the simulator charges before the latency clock starts).

**Pass criterion:** sub-test 1 shows a clear step; sub-test 2's p50 matches B8 within the CI; sub-test 4's 1-D TMA latency matches B8's `tma_bulk_latency_s` within 10 %.

**Contract:** `cp_async_groups_in_flight` (per warp / per SM), `cp_async_issue_cycles_per_16B`, `cp_async_latency_dist` (p1/p50/p99/max by tile size and occupancy, cycles and seconds), `tma_cp_async_shared_path` (bool), `tma_2d_latency_s`, `mbarrier_s`.

**Spark time:** 5 min. **ncu:** none needed. **Ref:** [2] (Hopper async copy dissection).

### B16 `b16_block_scheduler` — assignment order, dispatch rate, wave behaviour, dynamic refill, tail

**Mechanism:** M8. The skinny GEMM (32 blocks on 48 SMs) is a scheduler case; the simulator's greedy rule is a guess until this runs.

**Procedure:**
1. **Order.** Each block records `%smid`, `%globaltimer` at entry, and `blockIdx` into a table, then spins for a fixed 20 µs so all blocks of the first wave are resident together. Grid sizes 1, 8, 24, 32, 47, 48, 49, 96, 144, 192 blocks; block sizes 128 and 256; one and two blocks' worth of smem. Report for each grid: the SM visited by each block index in order, and the number of SMs holding two blocks when the grid is below 2× SMs. This settles whether the Spark fills SMs round-robin across TPCs/GPCs, packs, or interleaves, and whether 32 blocks land on 32 distinct SMs.
2. **Dispatch rate.** Entry-timestamp spacing of consecutive blocks on an empty GPU, 4096 trivial blocks: blocks per µs, and whether dispatch is per-GPC parallel (timestamps cluster) or serial.
3. **Dynamic refill.** Half the blocks spin 10 µs, half 100 µs, grid 4× SMs: do short blocks' successors start on the freed SM before long ones finish (expected yes)? Report the refill latency (gap between a block's exit and its successor's entry on the same SM).
4. **Tail and wave quantisation.** Grid = `w × 48 + r` for `r` in 0..47, equal 50 µs blocks: total time vs `r`; the simulator must reproduce the staircase.

**Pass criterion:** sub-test 4's staircase has step width 48 (or the measured resident count); sub-test 1 is reproducible across 10 repetitions (report the fraction of identical orderings; the scheduler may be nondeterministic and that itself is a finding).

**Contract:** `block_dispatch_order` (enum + the raw table), `block_dispatch_rate_per_s`, `block_refill_latency_s`, `block_dispatch_deterministic` (fraction), `sm_to_gpc_map` (if inferable from timestamp clustering).

**Spark time:** 2 min. **ncu:** none. **Ref:** Accel-Sim [3] (scheduler policies), [1].

### B17 `b17_latency_hiding` — resident warps vs achieved bandwidth, memory-level parallelism, outstanding requests, issue width

**Mechanism:** M1 (warp), M5 (queues). Little's law: bytes in flight per SM = bandwidth × latency. The simulator's request queues need their depth, and the warp model needs to know when a single warp's independent loads stop overlapping.

**Procedure:**
1. **Occupancy sweep.** Grid-stride read over 1 GiB (DRAM) and over 8 MiB (L2), with resident warps per SM forced to 1, 2, 4, 8, 16, 32, 48 (via dynamic smem) and one 128-bit load per thread per iteration: achieved bandwidth vs warps. The knee is the warp count that hides latency; bandwidth at 1 warp gives per-warp request throughput.
2. **MLP per warp.** One warp per SM; `u` independent 128-bit loads unrolled before any use, `u` = 1..32: time per iteration vs `u`. The point where time stops being flat is the per-warp outstanding-load limit.
3. **Per-SM outstanding requests.** 48 warps per SM each with `u` = 8: aggregate requests in flight at the bandwidth knee = per-SM limit (MSHR-equivalent). Report for DRAM and for L2 separately.
4. **Issue.** Independent FP32 FMAs from 1, 2, 4, 8 warps per SM (one per scheduler partition): instructions per cycle per SM vs warps; the saturation count is the number of schedulers and their issue width. Dependent-chain latency of FMA, IMAD, shared load (`ld.shared`), global L1-hit load, in cycles.

**Pass criterion:** sub-test 1 at 48 warps reproduces B4's DRAM plateau within 5 %; sub-test 4's FMA throughput at saturation matches B2 within 2 %.

**Contract:** `warps_to_hide_dram_latency`, `warps_to_hide_l2_latency`, `per_warp_outstanding_loads`, `per_sm_outstanding_requests` (dram, l2), `schedulers_per_sm`, `issue_width_per_scheduler`, `op_latency_cycles` (fma, imad, ld_shared, ld_l1hit).

**Spark time:** 4 min. **ncu:** `smsp__warps_active`, `sm__inst_executed`. **Ref:** [1] (dependent-chain latencies), Volkov 2016 (latency hiding, PhD thesis UC Berkeley).

### B18 `b18_dram` — channel count and interleave, row-buffer locality, read/write turnaround, CPU contention (LPDDR5X)

**Mechanism:** M5 (DRAM), M11. The Spark's memory is LPDDR5X shared with the CPU, not HBM: channel count, interleave granularity and open-page behaviour differ from the datacenter parts the literature describes.

**Procedure:**
1. **Interleave.** 1 GiB footprint, all SMs, each warp reading 128 B at stride `s`, `s` = 128 B … 64 MiB: bandwidth vs stride. Each drop by a factor ~2 at a power-of-two stride marks an interleave bit; the stride at which bandwidth bottoms out gives the channel count (bandwidth there ≈ one channel). Report the bit set and `channels`.
2. **Row locality.** Sequential 128 B reads vs random 128 B reads vs random 4 KiB chunks over 1 GiB: the sequential/random ratio is the open-page benefit; the chunk size at which random reaches 90 % of sequential is the effective page size.
3. **Mixed read/write.** Streams with read fraction 0, 0.25, 0.5, 0.75, 1 over 1 GiB: aggregate bandwidth vs mix; the dip below the linear interpolation of the pure cases is the turnaround cost.
4. **CPU contention (Spark-specific).** Run the GPU read stream while a host thread does a 10-thread `memcpy` loop over 4 GiB; report GPU bandwidth with and without CPU load. This separates the B12 unified-memory penalty into *bandwidth sharing* (shows here) vs *migration* (does not).
5. **Refresh/thermal noise.** 60 s of sustained streaming, bandwidth sampled per second (as B11 did for clocks): report drift.

**Pass criterion:** sub-test 1's unstrided bandwidth equals B4 within 5 %; channel count × per-channel bandwidth is within 15 % of the 273 GB/s datasheet.

**Contract:** `dram_channels`, `dram_interleave_bits`, `dram_page_bytes_effective`, `dram_seq_random_ratio`, `dram_rw_turnaround_penalty` (curve), `dram_bytes_per_s_under_cpu_load`, `dram_drift_60s`.

**Spark time:** 6 min. **ncu:** `dram__sectors_read.sum`. **Ref:** [4], Accel-Sim [3].

### B19 `b19_registers_local` — register-pressure cliff, spill cost, local-memory latency and bandwidth, register bank conflicts

**Mechanism:** M2, M7. The paper's Ventus kernel needed 173 registers; the compute block's rate collapses under spills, and the simulator must know by how much.

**Procedure:**
1. **Spill cost.** An FMA-dense kernel with a 64-accumulator working set, compiled with `-maxrregcount` in {255, 128, 96, 64, 48, 32}: throughput vs register cap, and (from `cuobjdump -res-usage` / `--ptxas-options=-v`) the spill bytes per setting. Report throughput as a fraction of the unspilled rate vs spill bytes.
2. **Local memory.** Pointer chase and streaming over a thread-local array forced to local memory (dynamic index): latency and per-SM bandwidth; whether it hits in L1 (compare with B14 latency).
3. **Register bank conflicts.** FFMA sequences whose operand register numbers are chosen (via inline PTX with explicit register pinning in SASS being impossible, use the Jia et al. method: generate variants, read SASS with `cuobjdump -sass`, bin by operand-bank pattern) and measure throughput per pattern. If SASS inspection is impractical, record `not measured` with the reason; this is the lowest-priority item here.
4. **Allocation granularity.** From B9's occupancy API sweep, the register-per-thread step at which blocks-per-SM changes (likely already in raw; just extract).

**Pass criterion:** sub-test 1 at 255 matches B2 within 2 %; local-memory latency at a small footprint equals the L1 latency from B14 (local memory is cached in L1).

**Contract:** `spill_penalty_curve`, `local_mem_latency_ns`, `local_mem_bytes_per_s_per_sm`, `reg_alloc_granularity`, `reg_bank_conflict_penalty` (or `not_measured`).

**Spark time:** 3 min. **ncu:** `l1tex__t_sectors_pipe_lsu_mem_local`. **Ref:** [1] §register banks.

### B20 `b20_tensor_detail` — FP8 via cublasLt, per-shape `mma.sync` rates, tensor-op latency, smem→tensor feed (`ldmatrix`), `wgmma`/`tcgen05` availability

**Mechanism:** M7 (tensor unit) for the coming tensor-core template.

**Procedure:**
1. **FP8.** `cublasLtMatmul` with `CUDA_R_8F_E4M3` inputs, FP32 accumulate, 8192², also `E5M2`; and the `mma.sync` FP8 shapes if they compile on `sm_121`. Fill the `fp8: null` in B3.
2. **Shapes.** Per-SM rate for every `mma.sync` shape the compiler accepts on `sm_121` for FP16/BF16/TF32/FP8/INT8: `m16n8k16`, `m16n8k8`, `m16n8k32`, `m16n16k16` (WMMA) etc. Record the accepted set (functional) and the rate (timing).
3. **Latency.** Dependent `mma.sync` chain (accumulator feeds the next): cycles per op; this sets how many independent tiles a warp needs in flight.
4. **Feed.** `ldmatrix.x4` throughput from shared memory per SM, with and without the swizzle that avoids bank conflicts (the shared-memory layout decision in the tensor template couples to this); ratio of feed bandwidth to the tensor unit's consumption at peak.
5. **ISA probes (functional).** Compile-and-run probes for `wgmma.mma_async`, `tcgen05.mma`, `cp.async.bulk.tensor` (2-D TMA), `setmaxnreg`, thread-block clusters (`cluster_dims` launch) and distributed shared memory. Record each as `supported` / `compiles_but_traps` / `unsupported` with the error text. Expectation for `sm_121`: no `wgmma`, no `tcgen05`, clusters uncertain.

**Pass criterion:** sub-test 2's `m16n16k16` BF16 rate matches B3 within 3 %.

**Contract:** `tensor_flops_per_s_per_sm.fp8_e4m3`, `.fp8_e5m2`, `.int8`, `tensor_shapes_supported` (list), `tensor_op_latency_cycles` (by precision), `ldmatrix_bytes_per_s_per_sm` (swizzled / unswizzled), `isa_features` (map).

**Spark time:** 3 min. **ncu:** `sm__pipe_tensor_cycles_active`. **Ref:** [2] (Hopper tensor dissection), [1].

### B21 `b21_compute_block` — non-FMA instruction rates, address arithmetic share, the GEMM inner loop's effective rate

**Mechanism:** M7 (the compute block's issue model). The simulator collapses the stretch between memory events into "flops at a rate"; this benchmark measures how far the real inner loop is from that rate and why.

**Procedure:**
1. Per-SM throughput for IMAD, IADD3, SHF, LOP3, I2F/F2I, MUFU (rsqrt), FADD, FMUL, branch (uniform), `ld.shared` 32/64/128-bit, `st.shared`, using the B2 pattern (ILP 8, 8 blocks/SM).
2. The tiling template's own generated inner loop (from `codegen`, T5's design) with the global loads removed and the smem tile pre-filled: cycles per tile MAC step. Compare with pure FMA count / B2 rate; the ratio is the compute-block efficiency and the SASS (`cuobjdump -sass`) instruction mix explains it (fraction of FFMA vs LDS vs integer).
3. The same for the register-blocked variants the menu contains (`R` in {2, 4, 8} accumulators per thread dimension): efficiency vs `R`.

**Pass criterion:** FFMA rate equals B2; sub-test 2's efficiency is reported with its SASS mix in raw.

**Contract:** `inst_throughput_per_sm` (map by opcode class), `compute_block_efficiency` (by design id, with the instruction mix), `ld_shared_bytes_per_s_per_sm` (by width).

**Spark time:** 3 min. **ncu:** `sm__inst_executed_pipe_*`. **Ref:** [1].

### B22 `b22_clock_governor` — clock response to load steps, warm-up, power-capping, sustained drift

**Mechanism:** M10. Clocks cannot be locked; F3 measured the same design at 242 µs and 217 µs on different days because one run sat at 1267 MHz. The runner needs a warm-up policy and the simulator needs the sustained clock *for the kind of load it is simulating*.

**Procedure:**
1. **Step response.** From idle, launch a 2 s FMA burst; sample `clock64`/`%globaltimer` every 1 ms inside the kernel: clock vs time since start. Report time-to-sustained and the start clock.
2. **Load dependence.** Sustained clock under four 30 s loads: FMA-only (B2), DRAM-stream (B4), tensor (B3), the T5 GEMM itself. The simulator uses the clock matched to the critical-path resource.
3. **Power cap.** `nvidia-smi` power and clock at 1 s during each load; report whether the governor is thermal or power limited (clock falls as power approaches the cap vs as temperature rises).
4. **Gap behaviour.** Back-to-back 200 µs kernels with idle gaps of 0, 1, 10, 100 ms between them: clock at the start of each kernel vs gap; this is what the F3/F4 runner's `reps=50` loop actually experiences.

**Pass criterion:** sub-test 2's FMA case matches B11's sustained clock within 2 %.

**Contract:** `clock_warmup_s`, `clock_idle_hz`, `clock_hz_sustained_by_load` (map), `clock_governor_limit` (enum power/thermal), `clock_after_gap_hz` (curve). **Runner change:** warm up for `clock_warmup_s` before timed reps, and report the in-run clock (already done) against `clock_hz_sustained_by_load`.

**Spark time:** 4 min. **ncu:** none. **Ref:** B11.

### B23 `b23_unified_mechanics` — page-migration cost by page size, prefetch, access counters, coherent host access, CPU-side cost

**Mechanism:** M11. B12 gave a 1.46× penalty with no mechanism. The simulator needs to know whether a managed page costs a migration on first touch (a latency event) or a bandwidth share (a rate), and what `cudaMemPrefetchAsync` buys.

**Procedure:**
1. **First-touch cost.** Managed buffer touched by the CPU; GPU kernel touches one byte per page for page sizes 4 KiB, 64 KiB, 2 MiB (set via `cudaMemAdvise` / allocation alignment): time per first touch vs steady-state touch.
2. **Prefetch.** Same buffer with `cudaMemPrefetchAsync` to the GPU first: bandwidth vs B12's `managed_cpu_first_pass1`.
3. **Coherent access without migration.** `cudaMemAdviseSetAccessedBy` / `SetPreferredLocation(cpu)`: GPU read bandwidth over host-resident pages (this is the `coherent_access_bytes_per_s` row, currently unexplained).
4. **Steady-state share.** GPU streams over device memory while the CPU streams over its own buffer (B18.4 reuses this); and GPU over managed memory while the CPU is idle. The difference between `managed` and `device` with the CPU idle is the managed-page overhead itself (TLB/translation), not sharing.
5. **Allocation granularity and limits.** Largest `cudaMalloc` that succeeds, largest managed allocation, and whether oversubscription beyond the 128 GB works (it should on unified memory).

**Pass criterion:** sub-test 2's prefetched bandwidth lies between `managed_gpu_first` and `device` from B12; sub-test 1's steady-state touch time matches the DRAM latency from B5.

**Contract:** `um_first_touch_s` (by page size), `um_prefetched_bytes_per_s`, `um_translation_overhead` (managed/device, CPU idle), `um_host_resident_bytes_per_s`, `um_max_alloc_bytes`.

**Spark time:** 3 min. **ncu:** none. **Ref:** NVIDIA Grace-Blackwell unified-memory documentation; B12.

### B24 `b24_atomics_reductions` — atomic throughput (same address, distinct, L2 vs smem), `red` vs `atom`, cluster/DSMEM if present

**Mechanism:** a new one, **M12 (atomic unit)**, needed by split-K GEMM, attention softmax reductions and the decode kernels later. Cheap to measure now, and the Spark's L2 atomics on LPDDR5X may be unusual.

**Procedure:** 48 blocks × 256 threads doing `atomicAdd` (fp32, int32) to (a) one address, (b) one address per warp, (c) one address per thread, in global memory and in shared memory; `red.global.add` (no return) vs `atom.global.add` (return); contention curve for 1..48 blocks on one address.

**Pass criterion:** distinct-address global atomic throughput approaches the L2 bandwidth / 4 B order of magnitude; the same-address curve is monotone in contention.

**Contract:** `atomic_throughput_per_s` (map: scope × pattern × op), `atomic_latency_ns` (global, shared).

**Spark time:** 2 min. **ncu:** `lts__t_sectors_op_atom`. **Ref:** [1].

### B25 `b25_multi_kernel` — kernel-to-kernel gap, CUDA-graph launch, concurrent kernels on streams, L2 persistence across kernels

**Mechanism:** M9 extended for whole-model inference (decode = chains of small kernels). The GEMM template never sees this; the attention/decode templates will on day one.

**Procedure:**
1. **Dependent-chain gap.** 1000 back-to-back 20 µs kernels on one stream: gap between kernel `i`'s end and `i+1`'s start (from `%globaltimer` at exit/entry), vs B10's `launch_s`.
2. **Graph launch.** The same chain as a CUDA graph: per-kernel gap.
3. **Concurrency.** Two streams each running 24-block kernels: do they co-run (total ≈ max) or serialize (total ≈ sum)? With 48-block kernels?
4. **L2 across kernel boundaries.** Kernel A writes 8 MiB; kernel B reads it immediately: B's bandwidth at L2 level means L2 persists across launches (expected); then with a 4 MiB cudaMemcpy in between, and with `cudaCtxResetPersistingL2Cache` / access-policy windows if supported on `sm_121` (functional probe).

**Pass criterion:** sub-test 1's gap is ≥ 0 and within 2× of B10's `launch_s`.

**Contract:** `kernel_gap_s`, `graph_kernel_gap_s`, `concurrent_kernels` (bool, with the block-count condition), `l2_persists_across_kernels` (bool), `l2_persisting_window_supported` (bool).

**Spark time:** 2 min. **ncu:** none. **Ref:** [1] §launch, CUDA Programming Guide (L2 persistence).

### B26 `b26_functional_probes` — the functional semantics the verifier's event model assumes

**Mechanism:** the *functional* reading of M3–M7. These are yes/no or "how many" questions whose answers are currently assumptions in `smem_verify.py` and in the simulator spec. Each runs a small kernel many times and reports a count.

1. **Partial visibility of an in-flight `cp.async`.** Destination slot pre-filled with a sentinel; one warp issues a 32 KiB `cp.async` and, *without waiting*, another warp polls the slot and records which 16 B chunks have changed and in what order, with `clock64` stamps. Reports: whether chunks land in issue order, the spread between first and last chunk (this *is* the race window the amplifier exploited), and whether any chunk is ever observed half-written (torn 16 B). The verifier's current model treats the slot as "undefined until `wait_group`"; this tells us whether a finer model (ordered chunk arrival) is sound.
2. **`wait_group N` semantics under reordering.** Issue groups 0,1,2; `wait_group 1`; check that groups 0 and 1 are complete and whether group 2 is ever already complete (allowed) — confirms the event model's "may complete any time after issue" rule.
3. **Barrier vs memory visibility.** Plain `st.shared` by warp 0, `__syncthreads`, `ld.shared` by warp 1: always visible (expected). Then with `bar.arrive`/`bar.sync` split barriers (named barriers): is `arrive` alone sufficient (expected no). Then `st.shared` + `__threadfence_block()` without a barrier, polled by another warp: visible after how many cycles (the "readiness without barrier" question for the β = 0 designs).
4. **TMA completion granularity.** As (1) for `cp.async.bulk` with mbarrier: can any byte be observed before `complete_tx` fires?
5. **FP32 arithmetic.** Whether `a*b+c` contracts to FMA by default (compare against `__fmul_rn`/`__fadd_rn`), denormal behaviour (`-ftz` default), and the rounding of the tensor-core accumulate for BF16→FP32 and TF32 (compare `mma.sync` against FP64 reference on inputs designed to expose truncation). This is what the 3e-6 cuBLAS agreement tolerance rests on, and the reference check should use the measured bound rather than a guessed one.
6. **Shared-memory bank mapping.** The address→bank function (4 B interleave over 32 banks assumed): confirm with the B6 conflict pattern at 8 B and 16 B access widths, and whether 64-bit accesses use the Volta+ half-warp rule.
7. **Warp lockstep.** Divergent branch inside a warp with a `clock64` on each side: does `sm_121` interleave divergent paths (independent thread scheduling) and does a `__syncwarp()` reconverge? This bounds how far two threads of one warp can drift, which the two-warp model treats as zero.

**Pass criterion:** every probe reports a definite answer with a count over ≥ 10 000 trials; nothing here has a numeric tolerance.

**Contract:** a new `functional:` section: `cp_async_chunk_order` (enum), `cp_async_window_cycles` (p1/p50/p99), `cp_async_torn_chunks_observed` (count), `wait_group_overcomplete_observed` (count), `threadfence_visibility_cycles`, `tma_early_bytes_observed`, `fma_contraction_default`, `ftz_default`, `tensor_accumulate_ulp_bound` (by precision), `smem_bank_map` (enum), `independent_thread_scheduling` (bool).

**Spark time:** 3 min. **Ref:** PTX ISA (async copy and mbarrier memory model), [2].

---

## 2. Summary table

| # | Benchmark | Fills contract fields for | Mechanism | Spark time |
|---|---|---|---|---|
| B13 | `b13_l2_structure` | sets/ways/slices/hash/replacement/sector | M5 L2 | 10 min |
| B14 | `b14_l1_carveout` | L1 size/line/latency per carve-out; copy-path L1 allocation | M5 L1, M3, M4 | 3 min |
| B15 | `b15_copy_queue` | pending groups, issue cost, latency distribution, TMA contention, mbarrier | M4 | 5 min |
| B16 | `b16_block_scheduler` | dispatch order/rate/refill/determinism, wave staircase | M8 | 2 min |
| B17 | `b17_latency_hiding` | warps-to-hide, MLP, outstanding requests, schedulers, op latencies | M1, M5 queues, M7 | 4 min |
| B18 | `b18_dram` | channels, interleave, page, R/W turnaround, CPU contention, drift | M5 DRAM, M11 | 6 min |
| B19 | `b19_registers_local` | spill curve, local memory, allocation granularity | M2, M7 | 3 min |
| B20 | `b20_tensor_detail` | FP8, shapes, tensor latency, `ldmatrix` feed, ISA features | M7 tensor | 3 min |
| B21 | `b21_compute_block` | non-FMA rates, inner-loop efficiency with SASS mix | M7 | 3 min |
| B22 | `b22_clock_governor` | warm-up, clock by load type, governor limit, gap behaviour | M10 | 4 min |
| B23 | `b23_unified_mechanics` | first-touch by page size, prefetch, translation overhead, host-resident BW | M11 | 3 min |
| B24 | `b24_atomics_reductions` | atomic throughput/latency by scope and pattern | M12 (new) | 2 min |
| B25 | `b25_multi_kernel` | kernel gap, graph gap, concurrency, L2 persistence | M9 | 2 min |
| B26 | `b26_functional_probes` | async-copy visibility and window, barrier/fence semantics, FP rounding, bank map, warp divergence | functional reading of M3–M7 | 3 min |

**Total: about 55 minutes of Spark time with repetitions**, in one `--micro` run. The writing is the cost: 14 CUDA programs, most under 150 lines each in the existing `common.cuh` style. B13, B16, B17 and B26 are the ones that must be read carefully before trusting; B19.3 and parts of B20.5 may legitimately come back `not measured`.

**Priority if time is short:** B16, B15, B13, B17, B26 first (the two known misses and the race window), then B22 and B18, then the rest.

---

## 3. What changes elsewhere

- **`agentic/schema/chip.json`:** every new field above, each optional, with `method`, `interval`, `raw` provenance as today; a new `functional:` block; `l2_bytes_effective` gains a `structure_confirmed` flag set by B13.
- **`fit.py`:** one fitter per benchmark; B13 and B18 do inference (bit masks, channel count) and must write their reasoning into `detail` (the latency/bandwidth tables they inferred from), so a reader can check the inference.
- **`preflight.md`:** P0 status (profiler counters), `cuobjdump`/`cublasLt` availability, cluster-launch support.
- **Console `/calibration` page:** the new rows, grouped by mechanism (M1–M12) rather than listed flat, with the functional block shown as a separate table of yes/no answers.
- **`notes/mechanistic_simulator_design.md` §2:** each mechanism row gains the B13+ benchmarks in its parameters column; §4's calibration table gains rows for B13–B26 with the tolerances above; G2's structural choices for L2 are no longer free: they come from B13.
- **Runner:** warm-up policy from B22; the reference-check tolerance from B26.5.
- **Paper §6.1 / Appendix:** the measured contract table extends; the "undocumented L2" point becomes a measured structure with the inference method stated.

---

## 4. Hand-off line for Claude Code

> Read `calibration/suite/B13_plus.md`. First do P0 on the Spark (profiler counters) if you have sudo there, and record the result in `preflight.md`. Then implement B13–B26 as `calibration/suite/bench/b13_*.cu` … `b26_*.cu` in the B1–B12 style, extend `driver.py` (`--micro`), `fit.py`, `chip.json` and the calibration page, run the suite on spark-6951, and commit the raw records and the extended `contract.yaml`. Report cycles beside seconds everywhere. For each benchmark, put the inferred structure *and the table it was inferred from* in the record. Flag anything that comes back `not measured` with the reason. Do not start G0 of the simulator until the contract has the B13–B17 fields. No address or key in the repository.

---

## 5. References

[1] Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza, *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*, arXiv:1804.06826; Turing follow-up arXiv:1903.07486.
[2] W. Luo et al., *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture*, IPDPS 2024.
[3] M. Khairy et al., *Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling*, ISCA 2020.
[4] X. Mei, X. Chu, *Dissecting GPU Memory Hierarchy through Microbenchmarking*, IEEE TPDS 2017.
[5] V. Volkov, *Understanding Latency Hiding on GPUs*, PhD thesis, UC Berkeley, 2016.
[6] NVIDIA, *PTX ISA* (asynchronous copy, mbarrier, memory consistency model), and *CUDA C++ Programming Guide* (L2 persistence, unified memory on Grace-class systems).
