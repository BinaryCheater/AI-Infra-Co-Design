# The GB10 GPU: Microarchitecture and Instruction Set, from Measurement

*The NVIDIA GB10 in the DGX Spark, as seen from user space through 40 microbenchmarks, three composite kernels and the hardware counters. Every number in this book comes from a benchmark named in brackets, e.g. [B5]; the kernels are in `calibration/suite/bench/`, the raw records in `calibration/spark-6951/raw/`. Where a statement rests on NVIDIA's public documentation instead of a measurement, the text says so. Written October 2026 for the host spark-6951: driver 580.159.03, CUDA 13.0, `sm_121`.*

---

## How to read this book

A GPU kernel's time is set by a chain of stages. Instructions are fetched and issued by warp schedulers. Math instructions go to arithmetic pipes. Memory instructions go through a queue to a load/store unit, then to shared memory or L1, then to L2, then to DRAM. Between every pair of stages there is a queue, and a kernel is slow when one of those queues is full.

Part I (Chapters 1–9) follows that chain from the chip down to DRAM. Each chapter gives the structure, the numbers, and how the numbers were obtained, and Chapter 9 lists what is not known. Part II (Chapters 10–15) is the instruction set: how to read the machine code the compiler emits, how dependencies work on a machine without register renaming, what each instruction costs, three real kernels read line by line, and what the compiler does on the way.

Clock conversions use the sustained clock under load, 2.44–2.48 GHz [B22]. One nanosecond is about 2.46 cycles.

---

# Part I. The machine

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

**The tensor path, instruction by instruction.** One `mma.sync` occupies its sub-partition's tensor pipe for **16.1 cycles**, the same for FP16, BF16, TF32 and FP8 e4m3 (four different SASS instructions, one rate; the FLOP rates differ only because the shapes do) [B31]. Its result is ready **30.1 cycles** after issue [B31]; INT8 m16n8k32 is the odd one at 19.6 cycles of pipe and 81.6 of latency [B31]. Two independent `mma` in flight per warp reach the pipe's rate, and a second warp on the sub-partition doubles the time per warp, so the pipe, not the warp, is the resource [B31]. The warp that issued an `mma` issues nothing else for those 16 cycles: 16 FFMAs placed between two `mma` add their full cost, 1.1 cycles each [B34]. A second warp's FFMAs do not fill that gap either: with one warp issuing `mma` and another issuing FFMAs on the same sub-partition, the FFMA warp keeps 95 % of its rate and the `mma` warp loses 43 % [B34].

`ldmatrix.x4` returns **47 cycles** after issue in a chain through `mma` (81 per link less the `mma` latency and the address arithmetic) [B32]; a loop that loads the fragments one step ahead already runs at 90 % of the pipe (71 cycles per step of four `mma` and two `ldmatrix.x4`), and deeper register double-buffering buys nothing [B32]. The unswizzled layout costs 7 % inside that loop, not the 8× of the bare `ldmatrix` [B32]. A load-heavy block on the same SM, LDS.128 loops or `cp.async` streams or both, costs the tensor block at most 8 % [B33]. The accumulators are the register budget: a tensor loop with 128 accumulators per thread built at a 128-register cap runs 3.6× slower, with 192 accumulators 12× slower, from spills; 64 accumulators fit anywhere [B34]. The FP32-to-BF16 store epilogue of a 128 x 128 tile costs about 1 700 cycles per warp [B34].

**The SFU.** `ex2`, `rcp`, `rsqrt`, `lg2` and `tanh` (`.approx`) all issue at **one warp instruction per 8 cycles per sub-partition** (four lanes per cycle) by the pipe counter, with a latency of 18 cycles (`rcp` 20) [B38]. The FP32-to-BF16 converts `F2F` and the packed `F2FP` go through the same pipe at the same rate; `I2F` has a 4.6-cycle latency [B38]. One warp alone issues an `ex2` every 8.3 cycles, so a single warp already saturates its sub-partition's SFU [B38]. FFMAs between SFU instructions from the same warp: a few hide in the SFU's shadow (32 FFMAs after 8 `ex2` cost 19 cycles), many do not (128 cost their full 155) [B38]. The clock64 throughput table of [B21] that gave `MUFU rsqrt` 0.15 of FFMA, and the same pattern in [B38], disagree with the counter by 1.6×; the counter's 8 cycles per instruction is the number to use (gap 34).

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

**A copy that is slow to land.** In B28's copy-only kernel one SM completes one 4 KB copy tile every **536 cycles** at one to three resident blocks, and every 398–428 cycles at four and six [B28]. Two causes are ruled out. Issue is not it: a chunk issues in 6.8 cycles even behind loads [B30.4]. The copy code is not it either: in B28 the address arithmetic, commit, wait and barrier around the copy are about ten instructions per warp per tile, worth about 40 cycles, and running the kernel's real instruction stream through the simulator gives 321–338 cycles per tile, 200 short of the card [gap 40]. The shared-memory fill path is not visible to the wavefront counters (`op_ldgsts` reads zero, gap 25). So about 200 cycles per 4 KB tile are spent between the copy being issued and the data being usable, in a part of the path no benchmark has opened yet.

**Asynchronous copies (`cp.async`).** Each thread issues a 16-byte copy from global memory straight into shared memory, bypassing registers. The copy does not allocate in L1, with either `.cg` or `.ca` [B14]. Issuing one 16-byte chunk costs the warp **6.8 cycles** [B30]; the per-chunk address and bounds arithmetic around it in a generated kernel adds far more, about 120 cycles in the GEMM kernels [B15, SASS]. A tile then completes after a latency that grows with its size, median cycles with one block per SM [B15]:

| tile | 4 KB | 8 KB | 16 KB | 32 KB | 48 KB |
|---|---|---|---|---|---|
| p50 | 549 | 621 | 931 | 1 765 | 2 754 |
| p99 | 1 185 | 1 035 | 1 279 | 2 169 | 3 530 |

The slope is about 23 ns per KB, i.e. one SM pulls a tile at roughly 44 GB/s while it is the only consumer; 48 SMs together reach **1.96 TB/s** from L2-resident data [B8, B15]. With six blocks per SM sharing the path, the 16 KB median rises to 4 512 cycles and the tail to 86 000 [B15]. At least 22 copy groups can be outstanding per block before the issue of the next one stalls [B15].

The data of a copy lands in shared memory **in issue order**, 16 bytes at a time, and no 16-byte chunk is ever observed half-written [B26.1]. After `wait_group N`, the N most recent groups may or may not be complete; in 155 of 2000 trials a later group had already landed [B26.2]. This is what makes a kernel that reads a tile before its wait a race rather than a certain failure: the window between the first and last chunk of a 32 KB tile is about 13 000 cycles when swept by a polling warp, and the first chunk lands after about 960 [B26.1].

**TMA (`cp.async.bulk`).** A 16 KB 1-D bulk copy with an mbarrier completes in **500 ns** with one block per SM [B8], 1 176 cycles median and 1 305 at the 99th percentile when every SM issues one [B35]. The completion grows with residency because the chip's copy bandwidth is shared, not because of a per-SM queue: 16 KB tiles take 1 943 cycles median at two resident blocks, 2 728 at four and 4 792 at six, while the chip moves 1.5 to 1.9 TB/s throughout [B35]. By tile size at one block: 681 cycles for 4 KB, 810 for 8, 1 176 for 16, 1 989 for 32, 2 988 for 48 [B35]. A 2-D tensor-map copy costs the same as a 1-D copy of the same bytes (ratio 1.00) [B36]. TMA does not allocate in L1 [B14]. Bytes of a TMA tile can be read in shared memory before the mbarrier reports completion, about 30 bytes per 16 KB tile on average [B26.4]; a program that polls the destination instead of the barrier will see partial data.

**TMA under load.** TMA and `cp.async` do not share an issue path: four 8 KB TMA streams per SM move 1.48 TB/s, four `cp.async` streams 1.74, and two of each 1.94 TB/s, more than either alone [B36]. (B15's earlier reading of 1.67 TB/s for the mix, below both pure streams, is not reproduced; what the two engines share is the chip's 2 TB/s.) In the mix the TMA blocks take 1 976 cycles per 8 KB tile against the `cp.async` blocks' 1 008, so the bulk engine yields to the per-thread copies [B36]. TMA beside LDS.128 loop blocks loses 2.5 cycles per tile, nothing [B36]. TMA stores (shared to global) reach 0.94 TB/s and, run beside TMA loads, slow the loads from 1 098 to 2 193 cycles per tile [B36]. An mbarrier arrive-and-wait round trip costs **75 cycles** with 1 or 32 arriving threads and 111 with 256 (B15's 46 ns = 114 cycles) [B37]; with a 4 KB transaction pending it returns after 621 cycles, the TMA latency itself [B37]. Cluster multicast launches and completes but takes 500× longer than two unicast loads and reads the L2 twice; it saves nothing on this chip [B37]. Clusters and distributed shared memory launch; the DSMEM latency and bandwidth readings (3 648 cycles, 572 GB/s) do not fit one mechanism and are not relied on [B37] (gap 37).

**Shuffles.** `shfl.sync` is a load/store-unit instruction: the counter reads one shared-memory wavefront per shuffle, and shuffles stall on `mio_throttle` like loads [B39]. One warp issues a shuffle every **8.7 cycles**, and the sub-partition's rate with many warps is the same, one per 8 cycles [B39]. Latency: 27 cycles for `bfly` and `up`, 35 for `idx` [B39]. A 32-lane butterfly reduction (5 shuffles and 5 FADDs, dependent) takes 171 cycles [B39]; `redux.sync` compiles on `sm_121` and a dependent chain of them runs at 53 cycles per link [B39]; a 256-thread block reduction through shared memory takes 352 cycles [B39]. Beside LDS.128 loop warps on the same sub-partition a shuffle slows from 8.7 to 13.3 cycles [B39].

**The copy that seemed not to overlap.** A 128 x 128 x 32 BF16 tensor tile kernel with `cp.async` double-buffering runs its compute alone in 1 126 cycles per k-step and its copies alone in 937; together they take 1 579, so the copies add 453 cycles on top of the compute [C1]. That looked like a copy engine that stalls while the loop runs. It is not. The copies cost what their *instructions* cost: each 16-byte chunk needs address and bounds arithmetic on the slow integer pipe plus the 6.8-cycle hold of the `LDGSTS` itself, and all of it competes with the compute for the same issue slots. A simulator that runs the kernel's real SASS predicts 437 cycles of add-on at one block and 597 at two, against 453 and 560 measured; one that counted only the copy's bytes predicted 99 and 108 [gap 29]. The lesson for a template writer: the price of `cp.async` on this chip is the integer code around it, not the copy. The same T5 FP32 tile with 2-D TMA loads, where one thread issues the whole tile, still runs 18–21 % slower than the model (C2, gap 30), so the TMA engine's completion under compute stays open.

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

## Chapter 9. Measured for GEMM and attention; open

**What is measured.** Every stage a GEMM or an attention kernel passes through has a number from a benchmark in this book and a mechanism in the simulator that was checked as a prediction against a kernel it was not built from (`calibration/spark-6951/coverage.md` is the table): the schedulers and the scoreboard [B17, B27, B29], the queue to the load/store unit [B30, B28], shared-memory banks and wavefronts [B6, B26.6, B30], `cp.async` and TMA completion [B8, B15, B35, B36], the tensor pipe and `ldmatrix` [B31-B34, C1], the SFU and the converts [B38], shuffles [B39], the register budget [B9, B19, B34, B40], L1 [B5, B14], L2 [B4, B13], DRAM [B4, B18]. The compute halves of the composites predict within 5 % (GEMM T5 with TMA, C2) to 15 % (the BF16 tensor tile at one block, C1), the copy halves within 5 to 9 %.

**Open, in order of how much it costs a prediction.**

1. **Copy completion.** About 200 cycles per 4 KB `cp.async` tile land later on the card than any measured stage explains [B28, gap 40], and the TMA tile kernel is 18–21 % slower than predicted [C2, gap 30]. The instruction cost of copies is explained (Chapter 5, gap 29); what remains is between issue and data-ready. The next measurement is a `clock64` stamp at the copy issue and at the return of the wait, per phase and per block, with every block's start and end time: it says whether the copy is late or the warp is late to the wait, and it also settles how blocks share an SM when only two of three fit (gap 41).
2. **Issue cycles another warp can take.** An `mma` holds its warp 16 cycles and an FFMA warp beside it takes 8 of them [B34]; an SFU instruction holds 8 and a short run of FFMAs hides in it while a long one does not [B38]. The model holds the warp and gets the ends right and the middle wrong (gaps 27, 35). What the arbiter does between an eligible FFMA and an `mma` or `MUFU` in dispatch is the open question, and it is the one attention's softmax warps beside tensor warps will ask.
3. **Software pipelining by the compiler** is solved by giving the simulator the kernel's real instruction stream (Part II): the softmax rows went from +67 % to within 7 % at every residency [C3, gap 39]. B27's one-warp streaming points are still 2× off (gap 17).
4. **Shuffles beside loads.** A shuffle is one LSU wavefront, yet beside loads it slows more than the queue charges (13.3 against 9.2 predicted, gap 33).
5. **The scheduler's choice** among eligible warps: the `mma` warp loses to the FFMA warp [B34], so it is not oldest-first; the rule is not measured.
6. **Register-file bank conflicts.** Operand register numbers cannot be controlled from CUDA C++ or PTX, so the penalty has not been measured [B19.3].
7. **L2 internals:** the set hash, the number of slices, the replacement policy beyond "not LRU", and whether there are two partitions.
8. **DRAM channel count and controller queues.** The channel hash hides the count; the controller hides page structure.
9. **DSMEM and multicast.** Multicast is 500× slower than unicast and saves no traffic [B37]; the DSMEM readings do not fit one path (gap 37). Neither matters until a template uses clusters.
10. **Clock behaviour under heavier thermal load** than a 60 s FMA burst at 79 W.

Each of these is one benchmark. None of them changes the numbers in Chapters 1-8; they bound how far those numbers can be extrapolated.

---

# Part II. The instruction set

Part I described the machine. Part II describes what the machine is asked to do: the instructions the compiler emits for it, as NVIDIA's disassembler prints them. NVIDIA does not publish this instruction set. What follows is built from three sources, always named: the text that `cuobjdump -sass` prints for the 44 kernels we compiled for `sm_121`, the benchmarks of Part I that put a cost on each instruction, and, for the hidden control bits, published reverse engineering of earlier generations. An instruction that none of our kernels uses is not described.

---

## Chapter 10. Reading SASS

**One line, one instruction.** Every instruction is 16 bytes. The disassembler prints it on one line:

```
/*1090*/        FFMA R20, R48, R86, R20 ;
/*0550*/        LDGSTS.E.BYPASS.128.ZFILL [R13], desc[UR8][R8.64], P2 ;
/*0af0*/  @!P1  STG.E desc[UR10][R12.64], R55 ;
```

Left to right: the byte address inside the kernel; an optional **guard predicate** (`@P0` runs the instruction in lanes where P0 is true, `@!P0` where it is false); the **opcode** with its **modifiers** after dots (`LDS.128` is a 128-bit shared load, `HMMA.16816.F32.BF16` is a tensor multiply of shape m16n8k16 with BF16 inputs and FP32 accumulators); then the operands, destination first.

**Operands.** The kinds that appear in our kernels:

| Written as | What it is |
|---|---|
| `R0` … `R254`, `RZ` | the per-thread registers; `RZ` reads as zero and discards writes |
| `R48.reuse` | the same register, with a hint to keep it in the operand cache for the next instruction; no change in meaning |
| `-R11`, `\|R4\|` | negated or absolute operand, free |
| `P0` … `P6`, `PT` | predicate registers, one bit per lane; `PT` is always true |
| `UR4`, `URZ`, `UP0` | the **uniform** registers and predicates: one value per warp, not per lane |
| `c[0x0][0x160]` | a word in a constant bank: kernel arguments and launch constants |
| `[R36+UR5+-0x2000]` | a shared-memory address: register plus uniform register plus offset |
| `desc[UR8][R8.64]` | a global address: the 64-bit register pair R8:R9, through a memory descriptor held in UR8 |
| `0x1f`, `1.4426950216293334961` | immediates |

A wide operation names only its first register. `LDS.128 R48` writes R48, R49, R50 and R51. `HMMA.16816.F32.BF16 R64, R80, R62, R64` writes four accumulator registers R64–R67, reads four A registers from R80, two B registers from R62 and the four accumulators.

**Two datapaths.** Most instructions run once per lane, 32 lanes per warp. The uniform instructions (`UIADD3`, `UMOV`, `ULEA`, `USHF`, `UISETP`, `R2UR`, `S2UR`, `LDCU`) compute one value for the whole warp on a separate, narrow datapath. The compiler uses them for addresses that every lane shares: tile bases, loop counters, TMA descriptors. They take an issue slot and leave the per-lane pipes free.

**The control word.** Each 16-byte instruction carries, besides its operation, about 21 bits that the compiler fills to tell the hardware how to schedule it. `cuobjdump` prints them only as part of the raw hex. They are not documented by NVIDIA. Published reverse engineering of the Volta and Turing generations (Jia et al. 2018, 2019) found this layout in the top bits of the second 8 bytes:

| Field | Bits | Meaning |
|---|---|---|
| stall | 4 | cycles to wait before issuing the next instruction of this warp |
| yield | 1 | a hint about switching warps |
| write barrier | 3 | which of six dependency counters this instruction's result will release (7 = none) |
| read barrier | 3 | which counter releases when this instruction has read its source registers |
| wait mask | 6 | which counters must be clear before this instruction issues |
| reuse | 4 | operand-cache flags, the `.reuse` in the text |

We decoded a handful of `sm_121` words with that layout and they are self-consistent. In the copy loop of a GEMM kernel, `LDGSTS ... [R13]` sets read barrier 1, and the very next instruction, `IADD3 R13, ...`, which overwrites R13, has wait mask bit 1 set: the hardware will not let the address register be overwritten until the copy has read it. A constant load `LDCU` sets write barrier 1 because its latency is not fixed. That is enough to say the scheme survives on Blackwell; it is not a full decode. The simulator does not read these bits (Chapter 11).

---

## Chapter 11. Dependencies without renaming

**The register in the text is the register in the hardware.** A GPU has no register renaming. When an instruction writes R5, it writes the physical R5 of its thread, and a later instruction that reads R5 depends on it. Two values that are alive at the same time must be in different registers, and the compiler, not the hardware, makes sure of that (Chapter 14). For a reader this is good news: every true dependency is visible in the printed register names.

**Two kinds of latency.** Instructions fall into two groups by how their results become ready:

- **Fixed latency.** FP32 and integer math, moves, compares, predicate logic. The result is ready a fixed number of cycles after issue, 4 on this chip [B17, B27]. The compiler knows the number and encodes the wait in the *stall* field; no counter is used. When a warp waits on one of these, the profiler calls it `wait` (the stall count) or `short_scoreboard` (for the shorter variable-latency group below).
- **Variable latency.** Everything that goes through a queue: shared and global loads, `ldmatrix`, shuffles, constant loads, the SFU, conversions, the tensor pipe, copies. The result time depends on what else is in the queue, so the compiler cannot count cycles. It gives the instruction a **write barrier**, and puts that barrier in the *wait mask* of the first consumer. The profiler calls the wait `short_scoreboard` for the on-SM units (shared memory, SFU, shuffles) and `long_scoreboard` for anything that leaves the SM (global and local memory).

**Write-after-read.** A variable-latency instruction may read its source registers late. Overwriting a source before it has been read would corrupt the operation, so the instruction also gets a **read barrier**, and the next writer of that register waits on it. The `LDGSTS`/`IADD3` pair of Chapter 10 is this case.

**Predicates** are registers too. `ISETP` writes P0 four cycles after issue, a guarded instruction reads it.

**Memory order** inside shared memory is not tracked by registers. Between warps it is enforced only by `BAR.SYNC` (all threads of the block arrive, then all proceed, 37 cycles at 256 threads [B7]) and, for copies, by waiting on copy groups (`DEPBAR`) or mbarriers (`SYNCS`). Within a warp a load after a store to the same address sees the store. Global and local memory follow the same pattern with fences.

**How the simulator uses this.** It does not read the control bits. It rebuilds every dependency from the register names, charges the measured latency of the producer's class (Chapter 12), and lets the four schedulers pick among ready warps. Where the compiler chose a stall longer than the dependency needs, the card is a little slower than the model; that is visible as a small positive error and is not corrected by fitting.

---

## Chapter 12. The instructions, by pipe

Every opcode our 44 kernels contain, grouped by the unit that executes it. **Cost** is the cycles one warp instruction keeps that unit busy on one sub-partition; **latency** is issue to result ready. Both are in SM cycles at the sustained clock. The benchmark is in brackets; "not measured" means the simulator uses the stated stand-in.

**FP32 and the FMA pipe.**

| Opcode | Does | Cost | Latency |
|---|---|---|---|
| `FFMA` | a·b+c in FP32 | 1.07 [B2] | 4.2 [B27] |
| `FADD`, `FMUL` | add, multiply | 1.13, 1.16 [B21] | 4.2 |
| `FMNMX` | min or max | as `FADD`, not measured | 4.2 |
| `FSETP`, `FSEL` | compare to predicate, select | as `FADD`, not measured | 4.2 |
| `HFMA2` | two FP16 FMAs in one register; the compiler also uses `HFMA2.MMA R, -RZ, RZ, 0, 0` to zero a register | as `FFMA` | 4.2 |
| `IMAD` | integer multiply-add, also used for moves (`IMAD.MOV`) and shifts (`IMAD.SHL`) | 1.88 [B21] | 4.2 |

**Integer and address arithmetic.**

| Opcode | Does | Cost | Latency |
|---|---|---|---|
| `IADD3` | three-input add | 3.9 [B21] | 4.2 |
| `LEA`, `LEA.HI.X` | shift-and-add for addresses; the `.HI.X` half carries into the upper word of a 64-bit address | as `IMAD` | 4.2 |
| `LOP3` | any three-input bitwise function, given by an 8-bit table | 2.8 [B21] | 4.2 |
| `SHF` | funnel shift | 2.3 [B21] | 4.2 |
| `ISETP`, `PLOP3` | integer compare to predicate; predicate logic | as `IADD3` | 4.2 |
| `SEL`, `VIMNMX`, `PRMT` | select, integer min/max, byte permute | as `IADD3`, `LOP3` | 4.2 |
| `MOV`, `CS2R`, `S2R` | move; read a special register (`SR_TID.X`, the clock) | 1, not measured | 4.2 |

These rates came from B21's throughput kernels. The same kernel pattern over-read the SFU and shuffle rates by 1.6× and 16× because not all of its blocks ran at once (gap 34); the integer rates have not yet been re-read from the pipe counters, and may be high in the same way.

**The uniform datapath.** `UMOV`, `UIADD3`, `ULEA`, `USHF`, `UISETP`, `ULOP3`, `UIMAD`, `R2UR` (register to uniform register), `S2UR`, `CS2UR`, `VOTEU`. One issue slot each; no per-lane pipe. Not measured beyond that.

**The SFU and conversions (XU pipe).**

| Opcode | Does | Cost | Latency |
|---|---|---|---|
| `MUFU.EX2`, `.RCP`, `.RSQ`, `.LG2`, `.TANH` | 2ˣ, 1/x, 1/√x, log₂, tanh, approximate | 8 [B38] | 18; `RCP` 20 [B38] |
| `MUFU.RCP64H` | the seed of an FP64 reciprocal | not measured | not measured |
| `F2FP.BF16.F32.PACK_AB` | two FP32 to two BF16 in one register | 8 [B38] | about 10 [B38] |
| `F2F`, `F2I`, `I2F` | conversions | 8 [B38] | `I2F` 4.6 [B38] |
| `I2FP` | integer to float on the ALU, not the SFU | as `IADD3` | 4.2 |

A warp that issues a `MUFU` cannot issue again for about 8 cycles [B38], so one warp alone fills its sub-partition's SFU.

**The tensor pipe.**

| Opcode | Does | Cost | Latency |
|---|---|---|---|
| `HMMA.16816.F32.BF16` and `.F16` | m16n8k16 multiply-accumulate, FP32 accumulators | 16.1 [B31] | 30.1 [B31] |
| `HMMA.1688.F32.TF32` | m16n8k8 in TF32 | 16.1 [B31] | 30.1 |
| `QMMA` (FP8 e4m3, m16n8k32) | | 16.1 [B31] | 30.1 |
| `IMMA` (INT8, m16n8k32) | | 19.6 [B31] | 81.6 [B31] |

The cost is also the time the issuing warp waits: an `HMMA` holds its warp for 16 cycles, and another warp's FFMAs on the same sub-partition take some of those cycles from it (Chapter 3, [B34]). Two independent `HMMA` in flight per warp reach the pipe's rate [B31]. The datacenter Blackwell's `wgmma` and `tcgen05` instructions do not exist on `sm_121` [preflight].

**Shared memory, through the load/store queue (Chapter 5).**

| Opcode | Does | Cost | Latency |
|---|---|---|---|
| `LDS`, `LDS.64`, `LDS.128` | shared load, 4, 8 or 16 bytes per lane | one dispatch per 4.2 cycles per sub-partition, then 1, 2 or 4 wavefronts × conflict degree at one wavefront per cycle per SM [B30] | 27 [B27] |
| `STS`, `STS.128` | shared store | as loads | — |
| `LDSM.16.M88.4` | `ldmatrix.x4`: four 8×8 BF16 fragments into four registers per lane, in the layout `HMMA` wants | in the same queue; 12 cycles alone, 36 behind a burst of loads [B30] | 47 [B32] |
| `SHFL.BFLY`, `.UP`, `.IDX` | exchange a register between lanes | one shared-memory wavefront; one per 8 cycles per sub-partition [B39] | 27; `IDX` 35 [B39] |
| `ATOMS` | shared-memory atomic; `ATOMS.CAS` as a spin lock | not measured as an instruction | 54 ns ≈ 133 [B24] |

**Global, local and constant memory.**

| Opcode | Does | Latency |
|---|---|---|
| `LDG.E` | global load | 44 on an L1 hit, ≈ 336 from L2, ≈ 740 from DRAM [B5, B17] |
| `STG.E`, `STG.E.64` | global store | — |
| `LDL`, `STL` | local memory: spills and stack; lives in L1 | ≈ 38 [B19] |
| `LDC`, `LDCU` | constant bank to a register or a uniform register | not measured; the simulator uses the L1 hit latency |

**Copies.**

| Opcode | Does | Cost |
|---|---|---|
| `LDGSTS.E.BYPASS.128.ZFILL [Rs], desc[UR][Rg.64], P` | `cp.async`: 16 bytes from global into shared memory, past the registers and past L1; `ZFILL` writes zeros where the predicate turns the copy off | 6.8 cycles of the warp per chunk [B30]; completion from the copy engine (Chapter 5) |
| `LDGDEPBAR` | `cp.async.commit_group`: closes the current group | an issue slot |
| `DEPBAR.LE SB0, n` | `cp.async.wait_group n`: wait until at most n groups are pending | waits |
| `UTMALDG.2D [URs], [URd]` | a TMA load: one instruction, one thread, a whole tile, described by a tensor map | an issue slot; completion from the TMA engine (Chapter 5) |
| `SYNCS.ARRIVE.TRANS64`, `SYNCS.PHASECHK.TRANS64.TRYWAIT`, `SYNCS.EXCH.64` | mbarrier arrive with an expected byte count, test-and-wait on its phase, initialize | round trip 75 cycles at 1 or 32 arrivals, 111 at 256 [B37] |

**Synchronization and control.**

| Opcode | Does | Cost |
|---|---|---|
| `BAR.SYNC.DEFER_BLOCKING 0x0` | `__syncthreads` | 27–97 cycles by block size [B7] |
| `BSSY Bn`, `BSYNC Bn` | mark and close a region where lanes may diverge, so they reconverge | an issue slot |
| `WARPSYNC`, `ELECT`, `VOTE`, `VOTEU` | warp-level sync; pick one lane (TMA issue uses `ELECT`); vote | an issue slot |
| `BRA`, `EXIT`, `CALL`, `RET`, `LEPC`, `BREAK` | branch, end, call, return, read the program counter | an issue slot; branch cost not measured |
| `NOP`, `YIELD` | nothing; a hint to switch warps | an issue slot |

---

## Chapter 13. Three kernels, read line by line

**The FP32 GEMM inner loop** (the T5 tile, 8×8 register block per thread, here the TMA variant C2). A k-step is sixteen shared loads, a mix of 32-bit `LDS` and 128-bit `LDS.128`, followed by the 64 FFMAs of an 8×8 outer product:

```
LDS      R86, [R30+-0x7c0]            ; 32-bit loads ...
LDS.128  R48, [R36+UR5+-0x2000]       ; ... and 128-bit loads, interleaved
...
FFMA     R20, R48, R86, R20           ; the first FFMA waits 27 cycles for its loads
FFMA     R18, R48.reuse, R84, R18     ; R48 stays in the operand cache
...                                   ; 64 FFMAs per k-step
```

The compiler issues all the loads of a step before the first FFMA, so their 27-cycle latencies overlap. Counted from the real addresses, the step costs 12 wavefronts on the shared-memory pipe, and the counter agrees exactly [gap 20]: several of the loads are broadcasts, so far fewer than their widths suggest. Twelve wavefronts against 64 FFMAs per warp: the FFMAs win, and this loop is scheduler-bound (`not_selected`, Chapter 2).

**The BF16 tensor loop** (C1). One k-step of the 128×128×32 tile, per warp:

```
LDSM.16.M88.4  R84, [R76+0x4000]      ; four A fragments
HMMA.16816.F32.BF16 R64, R80, R62, R64
LDSM.16.M88.4  R60, [R0+0x6400]       ; next B fragments, one step ahead
MOV  R80, R72
MOV  R81, R74
HMMA.16816.F32.BF16 R56, R84, R80, R56
...
```

In the loop body we read, 17 `HMMA`, 5 `LDSM` and **30 `MOV`**. The MOVs are the compiler's doing: `HMMA` wants its two B registers adjacent, and the fragments `LDSM` returned are not, so it copies them into place before each multiply. Seventeen `HMMA` hold the warp for 17 × 16 = 272 cycles; the thirty MOVs cannot issue during those holds from the same warp, so they add about 30 cycles, 11 % of the step. That may be part of C1's compute-only miss at one block (+15 %, gap 28). A layout that lands the fragments where `HMMA` wants them would remove them.

**The softmax row** (C3). One row of 128 values, four per lane:

```
SHFL.BFLY PT, R10, R9, 0x2, 0x1f      ; row max: 5 shuffle-and-FMNMX steps
FMNMX     R10, R9, R10, !PT
...
FADD  R4, R4, -R11                    ; x - max, four values
FMUL  R4, R4, 1.4426950216293334961   ; times log2(e)
MUFU.EX2  R4, R4                      ; 2^(...), four of these
FADD  R3, R4, R5                      ; local sum
SHFL.BFLY PT, R3, R2, 0x10, 0x1f      ; row sum: 5 shuffle-and-FADD steps
...
MUFU.RCP  R12, R12                    ; 1 / sum
FMUL  R4, R4, R12                     ; normalize
F2FP.BF16.F32.PACK_AB R5, R5, R4      ; to BF16
STG.E desc[UR8][R2.64], R5
```

Two dependent chains of five shuffles each: 10 × (27 + 4) ≈ 310 cycles of latency per row with nothing to overlap it in one row. Four `MUFU.EX2` at 8 cycles each plus one `RCP`. The compiler, given two rows per loop pass, interleaves them, so the second row's shuffles fill the first row's waits. A model that ran the rows one after another was 60 % slow at one block and close at six, where other warps fill the waits instead; given the compiled instruction stream, it is within 7 % at every residency [C3, gap 39].

The binary also holds a double-precision reciprocal sequence (`MUFU.RCP64H` and `DFMA`), the compiler's expansion of a division in the source it could not prove was single precision. FP64 is not measured on this chip.

---

## Chapter 14. What the compiler does between CUDA and SASS

A template writes CUDA; `nvcc` turns it into PTX; `ptxas` turns PTX into SASS. Everything in the list below was seen in our own binaries, and each one changes the numbers the simulator must use.

1. **Unrolling.** Fixed-count loops become straight code. The T5 k-loop body is one k-step with 64 FFMAs; the copy loops are unrolled by the number of chunks per thread.
2. **Address strength reduction.** Row × stride becomes a pointer plus a constant. What remains is still a lot of integer code: around each `cp.async` chunk in the GEMM kernels it is about 120 cycles of instructions, against 6.8 for the copy itself [B15, B30]. In the BF16 tensor tile this code is the whole reason the copies seemed not to overlap the compute: 437 of the 453 cycles they add are its instructions [C1, gap 29].
3. **Load widening and keeping values in registers.** Adjacent 32-bit loads become one `LDS.128`. In the 32×128×32 tile the compiler kept some B values in registers across k-steps and issued 5.7 shared loads per step where the template has 8.5 (gap 20).
4. **Scheduling.** Loads move as early as dependencies allow, so their latencies overlap (Chapter 13, the GEMM loop). Independent work is interleaved (the softmax rows).
5. **Register allocation and spills.** The compiler assigns every value a physical register. When the block's register cap is too low it spills to local memory (`STL`, `LDL`), and the loop falls off a cliff: 7× slower for the FP32 GEMM at 64 registers, 3.6× for the tensor loop with 128 accumulators at a 128 cap [B19, B34]. `nvcc` ignores `-maxrregcount` when the kernel has a `__launch_bounds__`.
6. **Register shuffling for operand layout.** The 30 MOVs per step in the tensor loop (Chapter 13).
7. **Instruction selection.** `IMAD` is used for plain moves and shifts; `HFMA2.MMA` zeroes registers; division becomes a reciprocal and a correction.
8. **Padding and dead code.** Before each copy loop the compiler emits several `@!PT LDS RZ, [RZ]`, loads guarded by "never", which take an issue slot and do nothing. A TMA kernel carries a fallback path (`LEPC`, `CALL.ABS.NOINC`) that never runs in practice. Synchronous copy code the template never selects is still compiled in unless the template removes it; for small tiles that was half the static instruction count (gap 19).
9. **Dependency hints.** The control word of every instruction (Chapter 10).

For the simulator this means one rule: model the SASS, never the CUDA. The template says what was meant; the SASS says what the chip is asked to do.

---

## Chapter 15. The instruction set: what is not known

1. **The integer rates.** IMAD, IADD3, LOP3, SHF, I2F rates come from the B21 pattern that over-read the SFU and shuffles (gap 34). They should be re-read from `sm__inst_executed_pipe_*` per cycle.
2. **Unmeasured opcodes**: `MOV`, `CS2R`, `S2R`, the uniform datapath, `LDC`/`LDCU` latency, the FP compares and min/max, `ATOMS` as an instruction, branches, FP64 (`DFMA`, `MUFU.RCP64H`). The simulator uses stated stand-ins for each.
3. **The control-word layout on Blackwell.** Only spot-checked. A full decode would show where the compiler's stalls are longer than the dependencies need, which is the one place the simulator is expected to be too fast.
4. **Which pipe each integer opcode uses.** We follow the profiling guide's FMA/ALU split; only IMAD on the FMA pipe is consistent with our rates.
5. **The operand cache** (`.reuse`): its size and whether a missed reuse costs a register-bank conflict. Register-bank conflicts are not measurable from CUDA or PTX [B19.3].
6. **The arbiter** between an eligible FFMA warp and an `HMMA` or `MUFU` warp on one sub-partition (gaps 27, 35).

Each of these is one benchmark or one counter read.

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
| B31 | mma issue and latency | 3 | tensor pipe occupancy per `mma` by shape, latency, in-flight count |
| B32 | ldmatrix to mma | 3 | `ldmatrix` latency, pipeline depth, swizzle cost in a loop |
| B33 | tensor loop beside loads | 3, 5 | tensor block under LDS and `cp.async` blocks |
| B34 | mixed issue | 2, 3 | FFMA beside `mma`, the accumulator spill cliff, the epilogue |
| B35 | TMA residency | 5 | TMA completion by tile and residency, chip copy bandwidth |
| B36 | TMA contention | 5 | TMA beside `cp.async`, LDS, stores; 2-D vs 1-D |
| B37 | mbarrier and clusters | 5 | mbarrier cost by arrivals, multicast, DSMEM |
| B38 | SFU and converts | 3 | SFU rate and latency per op, FFMA overlap |
| B39 | shuffles | 5 | shuffle latency and rate, butterfly, beside loads, `redux` |
| B40 | register budget | 2 | attention-loop spill curve, local memory by warps, residency by registers |
| C1 | BF16 tensor tile | 3, 5 | the group-1 composite: compute, copies, both |
| C2 | T5 tile with TMA | 5 | the group-2 composite |
| C3 | softmax rows | 3, 5 | the group-3 composite |

## Appendix B. Public documentation used

Only for names and for facts no benchmark can reach: the LPDDR5X bus width and rate and the 273 GB/s figure (NVIDIA DGX Spark datasheet); the 128-byte line and 32-byte sector organisation of L1 and L2 and the name of the MIO queue (NVIDIA Nsight Compute kernel profiling guide, where the stall reasons are defined); the PTX semantics of `cp.async`, `cp.async.bulk` and mbarrier (PTX ISA). Everything else is from the benchmarks above.
