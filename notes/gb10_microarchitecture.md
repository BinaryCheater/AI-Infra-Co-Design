# The GB10 GPU: A Microarchitecture Reference from Measurement

*The NVIDIA GB10 in the DGX Spark, as seen from user space through 29 microbenchmarks and the hardware counters. Every number in this book comes from a benchmark named in brackets, e.g. [B5]; the kernels are in `calibration/suite/bench/`, the raw records in `calibration/spark-6951/raw/`. Where a statement rests on NVIDIA's public documentation instead of a measurement, the text says so. Written October 2026 for the host spark-6951: driver 580.159.03, CUDA 13.0, `sm_121`.*

---

## How to read this book

A GPU kernel's time is set by a chain of stages. Instructions are fetched and issued by warp schedulers. Math instructions go to arithmetic pipes. Memory instructions go through a queue to a load/store unit, then to shared memory or L1, then to L2, then to DRAM. Between every pair of stages there is a queue, and a kernel is slow when one of those queues is full.

The chapters follow that chain from the chip down to DRAM. Each chapter gives the structure, the numbers, and how the numbers were obtained. The last chapter lists what is not known.

Clock conversions use the sustained clock under load, 2.44–2.48 GHz [B22]. One nanosecond is about 2.46 cycles.

---

## Chapter 1. The chip

**What it is.** The GB10 is a Blackwell-generation part with **48 streaming multiprocessors** (SMs), compute capability 12.1, in a package that shares **128 GB of LPDDR5X** with a 20-core Arm CPU [B1]. There is no separate graphics memory: the GPU's "DRAM" is the same LPDDR5X the CPU uses, behind a 256-bit bus at 8533 MT/s, which the datasheet rates at 273 GB/s. The GPU reaches 235 GB/s of it on reads (Chapter 8).

**Clocks.** The SM clock idles at 2.50 GHz under a light load and settles to 2.44–2.48 GHz under sustained load, depending on the load type: 2.46 GHz for FMA chains, 2.44 for a GEMM inner loop, 2.48 for a DRAM stream or tensor work [B22]. There is no power or thermal cap in play: 60 s of full-chip FMA draws 79 W and the clock does not fall [B11, B22]. The clock needs no warm-up; the first kernel after a 1 s idle runs at 2.48 GHz [B22]. Clocks cannot be locked from user space, so every timed number carries a 1.6 % run-to-run spread [B11].

**Launching work.** A kernel launch costs **2.0 µs** on the device side and 1.8 µs on the host [B10]. Two dependent kernels on one stream are separated by **2.5 µs** from the end of one to the start of the next; inside a CUDA graph that gap is **0.5 µs** [B25]. Kernels on two streams run concurrently, even when each has 48 blocks [B25].

**Block dispatch.** The hardware places the blocks of a grid onto SMs in a fixed order for the first wave [B16]. For grids of up to 8 blocks, blocks go to SMs 0, 1, 2, … in order. For larger grids the order is SM pairs, round-robin over six groups of eight SMs: block 0 to SM 0, block 1 to SM 1, block 2 to SM 8, block 3 to SM 9, then 16, 17, 24, 25, 32, 33, 40, 41, then back to SMs 2, 3, and so on. This is the same on every launch. The six groups of eight are almost certainly the chip's GPCs; that is an inference, the GPC count is not published for this part.

Once the first wave is placed, later blocks go wherever a block finishes; only 10 % of launches repeat an identical order after the first wave [B16]. The gap between a block leaving an SM and the next block entering it is **192 ns** [B16]. A grid of 48 equal 50 µs blocks takes 103 µs; 49 blocks take 154 µs; the staircase step is one wave of 48 [B16].

**Residency.** An SM holds at most 48 warps (1536 threads) at once, which is six blocks of 256 threads, and at most 24 blocks of any size [B1, B9]. Shared memory per SM is 100 KB, and a block may ask for up to 99 KB of it with the opt-in attribute [B1]. Residency falls with the shared-memory request: 6 blocks of 256 threads at ≤ 8 KB each, 5 at 16 KB, 3 at 32 KB, 2 at 48 KB, 1 at 64 KB or more [B9]. The register file is 64K 32-bit registers per SM; a block of 256 threads using 71 registers fits six to an SM [B9, B19].

---

## Chapter 2. The SM front end

**Structure.** Each SM has **four sub-partitions**, each with one warp scheduler that issues **one warp-instruction per cycle** [B17]. Resident warps are divided among the four schedulers. An SM therefore issues at most 4 warp-instructions per cycle; the counters show 3.65 per active cycle with 16 warps running independent FMAs, 2.9 with 4 warps, 0.7 with one [ncu, `issue_by_warps`].

**Dependency latencies.** A warp cannot issue an instruction until its operands are ready. The time from issuing an instruction to its result being usable, in SM cycles [B17, B27]:

| Instruction | Latency (cycles) |
|---|---|
| FFMA, FADD, FMUL | 4 |
| IMAD | 4 |
| `ld.shared` (LDS) | 27 for the load alone; 33 when the chain includes the address arithmetic of a pointer chase |
| global load, L1 hit | 44 |
| `mma.sync` m16n8k16 FP16 | 29 |

**Hiding latency.** With one warp per scheduler and a single dependent FFMA chain, the scheduler is busy 1 cycle in 4. Four independent chains in one warp fill it; so do two warps with two chains each [B27]. For shared loads the rule is different, because the loads also pass through a queue into a unit that the four schedulers share (Chapter 5): one warp with one outstanding load takes 33 cycles per load; with `a` independent loads and `w` warps per scheduler the time per iteration is 8·a·w cycles once a·w ≥ 4 [B27]. That 8 is not a scheduler property. It is the SM's load/store unit serving one 32-bit load every 2 cycles, divided among four schedulers. ILP and TLP both hide the 27-cycle latency, and neither raises the rate past the unit's limit.

**What a scheduler stalls on.** The hardware counters classify every cycle a warp does not issue. For the FFMA-dense GEMM inner loop (tile 128×256×16, 8×16 register blocking) the picture per issue-active cycle is: 0.84 warps waiting because another warp was picked (`not_selected`), 0.31 waiting for the memory queue (`mio_throttle`), 0.15 on a fixed-latency dependency, 0.11 on a short scoreboard; the FMA pipe is busy 62 % of cycles and 87 % of issued instructions are FFMA [ncu, `inner_loop`]. For a load-heavy loop (32×32×32, 2×2 blocking) the order flips: 1.52 short-scoreboard, 1.15 wait, 1.03 `mio_throttle`, 0.45 `not_selected`, and the FMA pipe is busy 18 % of the time [ncu]. These two profiles are the two regimes of Chapter 4.

**Divergence.** Threads of a warp do not interleave across divergent branches; a divergent `if/else` runs one side to completion before the other [B26.7]. The compiler contracts `a*b+c` into FFMA by default and does not flush denormals [B26.5].

**Register pressure.** The SASS of a GEMM inner loop that needs 71 registers runs at full speed. Capped at 64 registers it spills 216 bytes per thread and runs **7× slower**; at 48 registers, **33×**; at 32, **170×** [B19]. Spills go to local memory, which lives in L1 (15 ns latency) [B19], but the spill traffic competes with everything else in Chapter 5.

---

## Chapter 3. The arithmetic pipes

**FP32.** Each SM sustains **0.59 TFLOP/s** of FP32 FMA, 28.3 TFLOP/s for the chip, measured with dependent chains and eight-way ILP [B2]. At 2.46 GHz that is 120 FMAs per cycle per SM, consistent with 128 FP32 lanes (32 per sub-partition) and the issue limits of Chapter 2.

**Other instructions**, as a fraction of the FFMA rate [B21]:

| Instruction | Rate relative to FFMA |
|---|---|
| FADD, FMUL | 0.95, 0.93 |
| IMAD | 0.57 |
| SHF (shift) | 0.46 |
| LOP3 | 0.38 |
| I2F / F2I | 0.30 |
| IADD3 | 0.28 |
| MUFU rsqrt | 0.15 |

Integer address arithmetic is therefore two to four times as expensive per instruction as the FMAs it supports. In the generated GEMM kernels the non-loop instructions per phase (address, guard, wait, barrier, loop control) number 280–660 depending on the tile [B21, SASS].

**Tensor cores.** Per SM, through `mma.sync` [B20]:

| Precision, shape | TFLOP/s per SM | Chip (cuBLAS, 8192²) |
|---|---|---|
| FP8 e4m3, m16n8k32 | 5.13 | 198 (e5m2: 200) |
| INT8, m16n8k32 | 2.90 TOP/s | — |
| BF16, m16n8k16 | 2.57 | 95 |
| FP16, m16n8k16 | 2.57 | 92 |
| TF32, m16n8k8 | 1.28 | 40 |
| FP32 (no tensor core) | 0.59 | 21.6 |

The chip-level cuBLAS numbers are 75–80 % of 48 × the per-SM peak, the usual loss to feeding the tensor cores. A dependent `mma.sync` chain shows **29 cycles** of latency per operation [B20]. Accumulation into FP32 rounds to within 1.04 ulp for BF16 and FP16 inputs and 1.5 ulp for TF32 [B26.5].

**Feeding the tensor core.** `ldmatrix` from shared memory delivers **351 GB/s per SM** when the tile is laid out with the swizzle that avoids bank conflicts, and **44 GB/s** without it: an 8× penalty for the wrong layout [B20].

**ISA features on `sm_121`.** Thread-block clusters and distributed shared memory work (`map_shared_rank` reads a peer block's shared memory correctly). `wgmma`, `tcgen05` and `setmaxnreg` are rejected by `ptxas`. This is a consumer-class Blackwell SM, not the datacenter one [preflight P0].

---

## Chapter 4. The GEMM inner loop as a case study

The purpose of Chapters 2 and 3 is to explain this table. A tiled GEMM inner loop, with its tile already in shared memory, reaches only a fraction of the FFMA bound [B21]:

| Tile (M×N×K), register blocking | LDS per thread per phase | FFMA per thread per phase | Cycles per phase | FFMA bound | Efficiency |
|---|---|---|---|---|---|
| 128×256×16, 8×16 | 384 | 2048 | 7 186 | 3 317 | 46 % |
| 128×128×32, 8×8 | 512 | 2048 | 7 361 | 3 317 | 45 % |
| 64×256×16, 4×16 | 320 | 1024 | 4 867 | 1 658 | 34 % |
| 64×64×32, 4×4 (2 blocks/SM) | 256 | 512 | 5 343 | 829 | 16 % |
| 32×32×32, 2×2 (4 blocks/SM) | 128 | 128 | 4 214 | 207 | 5 % |

The pattern: efficiency tracks the ratio of FFMAs to shared loads. With 8×8 blocking each `LDS.128` feeds 16 FFMAs and the loop is scheduler-bound (`not_selected`). With 2×2 blocking each load feeds one FFMA and the loop waits on the memory queue (`mio_throttle`, `short_scoreboard`). The next chapter is about that queue.

---

## Chapter 5. Shared memory and the load/store path

**Banks.** Shared memory has **32 banks of 4 bytes**. A warp's 32-bit accesses are served in one pass when they touch 32 distinct banks. 64-bit accesses are served per half-warp, 128-bit per quarter-warp, so a warp-wide `LDS.128` always takes at least four passes [B26.6]. The cost of a bank conflict, as a multiplier on a conflict-free access, by word stride [B6]:

| stride (words) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| penalty | 1.0 | 1.0 | 1.9 | 3.8 | 7.6 | 15.3 |

Stride 2 is free, so the bank datapath is effectively 8 bytes wide per bank pair; from stride 4 up the penalty is stride/2.

**Throughput of the shared-memory pipe.** With independent loads and no conflicts, one SM sustains, per cycle [B21]:

| Instruction | Warp-instructions per cycle per SM | Bytes per cycle per SM |
|---|---|---|
| LDS.32 | 1.05 | 135 |
| LDS.64 | 0.99 | 254 |
| LDS.128 | 0.49 | 253 |
| STS.32 | 0.91 | 117 |

So the pipe moves about 256 bytes per cycle per SM, 0.63 TB/s at 2.46 GHz, and reaches it only with 64- or 128-bit accesses. A dependent chain of 32-bit loads gets a quarter of that, 154 GB/s [B6], because each load waits 27 cycles for the previous one.

**The queue in front of the pipe.** Shared loads and stores, global loads and stores, and `cp.async` all pass from the sub-partition's scheduler through a queue (NVIDIA's counters call it MIO) into the load/store unit. When the unit cannot accept more, the scheduler reports `mio_throttle`. The structure, from B30:

- **Dispatch.** A sub-partition can hand one shared load to the queue every **4.2 cycles**. With two warps on the same sub-partition each gets one every 10.1 cycles, so the slot belongs to the sub-partition, not the warp [B30].
- **No visible depth.** One warp issued 64 shared loads back to back and never stalled. What limits issue is the drain, not a full queue [B30].
- **Drain.** The unit retires **one wavefront per cycle per SM**, and never more than one instruction per 2 cycles. A conflict-free 32-bit load is one wavefront, 64-bit two, 128-bit four, each multiplied by the bank-conflict degree; the profiler's wavefront counts match the clock on all ten width-and-stride cases [B30]. So an SM sustains at most 0.5 conflict-free 32-bit loads per cycle and 0.25 128-bit loads, and the rate does not change between 4 and 48 resident warps; only the `mio_throttle` count grows, from 0.1 to 8.5 stalled warps per issue cycle [B30].
- **Broadcast is cheap.** A load whose 32 threads read one address is one wavefront whatever its width. This is why a loop of broadcast `LDS.128` reaches one instruction per cycle [B28], and why the A-fragment loads of a register-blocked GEMM, which are broadcast across the threads of a row, cost a quarter of what their width suggests [B21].
- **Who shares the queue.** `ldmatrix` waits behind shared loads (12 cycles alone, 36 after a burst). `cp.async` does not: issuing a 16-byte chunk costs **6.8 cycles** with or without shared loads in flight [B30]. Adding copy traffic to a loop of shared loads still slows the loads by 6–13 % [B28], so the two share the entry to the unit, not its queue.
- **Arbitration.** With several blocks resident, the blocks do not progress at equal rates; per-block times spread by 30 % [B28].

In short: a shared load costs the unit `wavefronts = passes(width, addresses) × conflict degree` cycles, and a kernel is load-bound when the sum of those over its resident warps exceeds the cycles its FFMAs need. The 16×32×8 GEMM tile looked like such a kernel; it is not. Its two A addresses per warp sit in different banks, the counter says 2.5 wavefronts per k-step, and what made it slow in the model was an instruction count, not a conflict (gap 19). The load-bound kernels in the test set are the 2×2 and 4×4 register-blocked ones, where each load feeds one or four FFMAs.

**A rule for template writers.** `ldmatrix` waits behind shared loads: 12 cycles alone, 36 after a burst of loads. `cp.async` issue does not: 6.8 cycles per chunk with or without a burst of loads in front of it [B30.4]. So a tensor kernel's fragment loads and its plain loads compete for one queue, while a kernel's copies can be issued into a busy load stream for free.

**An unexplained number.** In a loop that mixes shared loads with `cp.async`, one SM completes one 4 KB copy tile every **520 cycles**, about 19.6 GB/s per SM, whether one block issues the copies or six [B28]. The copy-only kernel gets 44 GB/s per SM [B8]. Issue is not the cause (B30.4), and L2 is not (the source is L2-resident). The cause has not been measured. The hypothesis, and it is only that: a copy's data is written into shared memory as wavefronts through the same one-per-cycle pipe as the loads, so a loop of loads leaves the fills the pipe's leftovers. The counter that settles it is `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts` beside `_op_ld` on B28's mixed run (gap 25).

**Asynchronous copies (`cp.async`).** Each thread issues a 16-byte copy from global memory straight into shared memory, bypassing registers. The copy does not allocate in L1, with either `.cg` or `.ca` [B14]. Issuing one 16-byte chunk costs the warp **6.8 cycles** [B30]; the per-chunk address and bounds arithmetic around it in a generated kernel adds far more, about 120 cycles in the GEMM kernels [B15, SASS]. A tile then completes after a latency that grows with its size, median cycles with one block per SM [B15]:

| tile | 4 KB | 8 KB | 16 KB | 32 KB | 48 KB |
|---|---|---|---|---|---|
| p50 | 549 | 621 | 931 | 1 765 | 2 754 |
| p99 | 1 185 | 1 035 | 1 279 | 2 169 | 3 530 |

The slope is about 23 ns per KB, i.e. one SM pulls a tile at roughly 44 GB/s while it is the only consumer; 48 SMs together reach **1.96 TB/s** from L2-resident data [B8, B15]. With six blocks per SM sharing the path, the 16 KB median rises to 4 512 cycles and the tail to 86 000 [B15]. At least 22 copy groups can be outstanding per block before the issue of the next one stalls [B15].

The data of a copy lands in shared memory **in issue order**, 16 bytes at a time, and no 16-byte chunk is ever observed half-written [B26.1]. After `wait_group N`, the N most recent groups may or may not be complete; in 155 of 2000 trials a later group had already landed [B26.2]. This is what makes a kernel that reads a tile before its wait a race rather than a certain failure: the window between the first and last chunk of a 32 KB tile is about 13 000 cycles when swept by a polling warp, and the first chunk lands after about 960 [B26.1].

**TMA (`cp.async.bulk`).** A 16 KB 1-D bulk copy with an mbarrier completes in **500 ns**; a 2-D tensor-map copy in 455 ns; an mbarrier arrive-and-wait with nothing pending costs 46 ns [B8, B15]. TMA copies reach **1.90 TB/s** chip-wide, and running TMA and `cp.async` together gives 1.67 TB/s, less than either alone, so the two share a path [B15]. TMA does not allocate in L1 [B14]. Bytes of a TMA tile can be read in shared memory before the mbarrier reports completion, about 30 bytes per 16 KB tile on average [B26.4]; a program that polls the destination instead of the barrier will see partial data.

**Barriers.** `__syncthreads` costs 11 ns for 128 threads, 15 for 256, 23 for 512, 40 for 1024 (27, 37, 57, 97 cycles) [B7]. A store to shared memory is visible to another warp after a barrier, after a named-barrier `arrive`/`sync` pair, or after `__threadfence_block` alone, the last within 243 cycles [B26.3].

**Atomics.** On shared memory, an `atomicAdd` returns in 54 ns; with every thread of the SM hitting one address, 2.5 G atomics/s per SM; on distinct addresses, 40 G/s [B24]. Global atomics are performed at L2 (Chapter 7).

---

## Chapter 6. L1 and the carve-out

L1 and shared memory are one 112 KB array per SM, split by the kernel's shared-memory request [B14]. L1 is **64 KB** when the block asks for 16 KB of shared memory or less, 32 KB at 48 KB, 16 KB at 64 KB and above [B14, counters]. Latency is **17 ns**, 42 cycles, and a dependent global load that hits in L1 returns in 44 cycles [B5, B17].

None of the copy instructions allocate in L1 (Chapter 5), so for a GEMM that stages through shared memory the L1 sees only the epilogue stores, register spills, and local memory. Local-memory accesses hit in L1 at 15 ns [B19].

The pointer chase that found 32 KB of L1 under-read it: a latency rise at 48 KB turned out not to be a miss. The counter `l1tex__t_sector_hit_rate` gives the 64 KB figure above, and where a counter and a timing inference disagree, this book uses the counter.

---

## Chapter 7. L2

**Capacity.** The L2 is **24 MiB**, as declared, and all of it is usable: a stream that cycles through 24 MiB keeps an 86 % hit rate, 28 MiB 83 %, 32 MiB 73 %, 48 MiB 24 % [counters, B13]. A single warp chasing pointers through 24 MiB keeps 91 % hits [B13]. The capacity edge is soft, which means the replacement policy is not pure LRU: a sequential cyclic pattern larger than the cache still hits part of the time, where LRU would miss everything [B13].

**The 16 MiB knee.** A bandwidth test with all 48 SMs streaming sees L2 bandwidth hold to 8 MiB and fall toward DRAM bandwidth from 16 MiB [B4]. That knee is where many concurrent streams start evicting each other faster than the cache can serve them; it is a bandwidth effect, not the capacity. A model that uses 16 MiB as the capacity will under-predict reuse for a single well-behaved stream.

**Lines, sectors, sets.** Lines are 128 bytes made of four 32-byte sectors; a miss fills one sector, so a stride-128 chase pulls 32 bytes per request from DRAM and a stride-32 chase pulls 128 [counters]. No set conflicts appear for up to 64 lines at any power-of-two stride, so the cache is either at least 64-way or, more likely, indexes its sets with a hash of many address bits [B13]. No single address bit selects a visibly slower slice: the latency spread across address bits is 6 ns around a mean of **136 ns** (132 near, 138 far) [B5, B13]. If the L2 is split into two partitions with a crossbar, as on larger Blackwell parts, the far side costs no more than that.

**Bandwidth.** Streaming reads of an L2-resident working set sustain **1.02 TB/s** chip-wide with a grid of simple blocks [B4], and 1.36 TB/s with 48 warps per SM issuing 128-bit loads [B17]. The bandwidth grows with resident warps and needs about 24 warps per SM to saturate [B17]; by Little's law that is about 30 outstanding requests per SM.

**Across kernels.** L2 contents survive a kernel boundary; a second kernel reading what the first wrote runs at L2 speed [B25]. The persisting-access-window API is accepted [B25].

**Atomics.** Global atomics execute at the L2. Same-address `atomicAdd` from the whole chip runs at **0.74 G/s**, one every 1.35 ns, and the rate does not change between 1 and 48 blocks contending, so the serialization is at the L2 slice, not in the SMs. Atomics to distinct addresses run at 25–28 G/s; `red` (no return value) is no faster than `atom`. Latency of a global atomic is 162 ns [B24].

---

## Chapter 8. DRAM: LPDDR5X behind the L2

**Bandwidth.** Reads **235 GB/s**, writes **198 GB/s**, copies 217 GB/s, against a datasheet 273 GB/s [B4]. Latency **300 ns** [B5]. The read bandwidth needs only 8 warps per SM to saturate, or about 12 outstanding requests per SM [B17].

**No page locality.** Random 128-byte reads run as fast as sequential ones, 186 GB/s each; random 4 KB chunks reach 229 GB/s and 16 KB chunks 235 [B18]. The memory controller hides DRAM row structure completely at this granularity; the effective "page" from the GPU's point of view is 128 bytes. Mixing reads and writes costs nothing beyond the write-bandwidth ceiling [B18].

**Channels.** Forcing any single address bit constant keeps at least 97 % of the bandwidth, so the channel is selected by a hash of several bits and the channel count cannot be read from user space by a stride sweep [B18].

**Sharing with the CPU.** When the CPU streams through memory at the same time, the GPU's read bandwidth falls from 235 to **157 GB/s** [B18]. Over 60 s of sustained streaming the bandwidth drifts by 5 % [B18].

**Counters.** The GB10 exposes no `dram__*` metrics to the profiler. DRAM traffic is read as L2 lookup misses × 32 bytes [preflight P0].

**Unified memory.** Allocation kind decides the GPU's read bandwidth [B12, B23]:

| Allocation | GPU read bandwidth |
|---|---|
| `cudaMalloc` (device) | 238 GB/s |
| `cudaMallocHost` (pinned host) | 200 GB/s |
| `cudaMallocManaged`, any first toucher, with or without prefetch | 163 GB/s |
| host-resident pages read coherently | 163 GB/s |

Managed memory costs a constant **1.45×** against device memory, with the CPU idle, whichever side touched it first and whether or not it was prefetched. So the penalty is address translation on the managed path, not migration and not bandwidth sharing. A managed allocation can be as large as 116 GB [B23].

---

## Chapter 9. What is not known

In order of how much it matters to predicting a kernel's time on this chip.

1. **The copy return path.** One 4 KB `cp.async` tile per SM every 520 cycles under a load-heavy loop, 19.6 GB/s against 44 for copies alone (Chapter 5). The copy-bound and barrier-bound GEMM designs in the test set wait on this, and the model has no mechanism for it. Four counters would show where the time goes: `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts`, `_op_ld`, `_op_ldsm`, and `l1tex__m_xbar2l1tex_read_bytes`. If the first is non-zero and fills plus loads sit near one wavefront per cycle, the fills share the shared-memory pipe; if not, the limit is between L2 and the SM.
2. **The scheduler's choice** among eligible warps: oldest-first, round-robin, or something else. The `not_selected` counts are consistent with either.
3. **Register-file bank conflicts.** Operand register numbers cannot be controlled from CUDA C++ or PTX, so the penalty has not been measured [B19.3].
4. **L2 internals:** the set hash, the number of slices, the replacement policy beyond "not LRU", and whether there are two partitions. The chip behaves like one 24 MiB cache with a 6 ns latency spread; that is all the outside view gives.
5. **DRAM channel count and controller queues.** The channel hash hides the count; the controller hides page structure. Only the aggregate bandwidth and the CPU-sharing loss are known.
6. **Outstanding-miss limits** beyond the Little's-law estimates (12 DRAM, 30 L2 requests per SM).
7. **The front end** above the scheduler: instruction cache size and fetch bandwidth. No kernel so far has been limited by them.
8. **The tensor core's operand path:** `ldmatrix` shares the queue with `LDS` [B30]; unknown is the operand collector depth that sets how many independent `mma.sync` a warp needs in flight (29 cycles of latency says at least 2–4).
9. **Clock behaviour under heavier thermal load** than a 60 s FMA burst at 79 W; no cap was reached, so the governor's limits are unseen.

Each of these is one benchmark. None of them changes the numbers in Chapters 1–8; they bound how far those numbers can be extrapolated.

---

## Appendix A. Benchmark index

| # | Name | Chapter | What it fixes |
|---|---|---|---|
| B1 | device properties | 1 | declared SM count, limits |
| B2 | FMA chains | 3 | FP32 rate |
| B3 | tensor (WMMA + cuBLAS) | 3 | tensor rates per precision |
| B4 | bandwidth vs working set | 7, 8 | L2/DRAM bandwidth, the 16 MiB knee |
| B5 | pointer chase | 6, 7, 8 | L1/L2/DRAM latency |
| B6 | shared-memory strides | 5 | bank-conflict penalty |
| B7 | barrier | 5 | `__syncthreads` cost |
| B8 | async copy | 5 | `cp.async`/TMA latency and throughput |
| B9 | occupancy | 1 | residency vs shared memory and registers |
| B10 | launch | 1 | launch cost |
| B11 | clocks under load | 1 | sustained clock, variance |
| B12 | unified memory | 8 | bandwidth by allocation kind |
| B13 | L2 structure | 7 | sectors, sets, slices, replacement, capacity |
| B14 | L1 carve-out | 6 | L1 size per configuration, copy allocation |
| B15 | copy queue | 5 | issue cost, latency distribution, TMA sharing |
| B16 | block scheduler | 1 | dispatch order, refill, waves |
| B17 | latency hiding | 2, 7, 8 | warps to saturate, schedulers, op latencies |
| B18 | DRAM | 8 | locality, channels, CPU sharing, drift |
| B19 | registers and local memory | 2 | spill cost |
| B20 | tensor detail | 3 | shapes, FP8, mma latency, ldmatrix |
| B21 | compute block | 3, 4, 5 | instruction rates, shared-load rates, inner-loop efficiency |
| B22 | clock governor | 1 | clock by load, warm-up, gaps |
| B23 | unified-memory mechanics | 8 | translation overhead, prefetch, limits |
| B24 | atomics | 5, 7 | throughput and latency by scope |
| B25 | multi-kernel | 1, 7 | kernel gaps, concurrency, L2 persistence |
| B26 | functional probes | 2, 3, 5 | copy ordering and visibility, rounding, bank map, divergence |
| B27 | dependency vs ILP vs TLP | 2, 5 | scoreboard latencies, load dispatch cost |
| B28 | LSU contention | 5 | load/store rate vs occupancy, arbitration |
| B29 | inner-loop ground truth | 4 | the generated GEMM loops under counters |
| B30 | scheduler-to-LSU queue | 2, 5 | dispatch slot, drain rate, wavefront costs, who shares the queue |

## Appendix B. Public documentation used

Only for names and for facts no benchmark can reach: the LPDDR5X bus width and rate and the 273 GB/s figure (NVIDIA DGX Spark datasheet); the 128-byte line and 32-byte sector organisation of L1 and L2 and the name of the MIO queue (NVIDIA Nsight Compute kernel profiling guide, where the stall reasons are defined); the PTX semantics of `cp.async`, `cp.async.bulk` and mbarrier (PTX ISA). Everything else is from the benchmarks above.
