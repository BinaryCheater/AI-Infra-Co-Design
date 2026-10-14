# A Mechanistic Hardware Model: Event-Driven Functional and Timing Simulation

**Design document, v0.1 · October 15, 2026**
*Li Shang, drafted with Claude. Follows `hardware_calibration_design.md` (F0–F5 done on spark-6951). Written for Claude Code to execute.*

---

## 0. Why the calibrated cost model is not enough

The calibration round replaced invented constants with measured ones. It did not change what the model *is*: two aggregate terms, `T_comp` and `T_mem`, combined by a `max`. On the Spark that model missed by −33 % on the 4096³ GEMM and +67 % on the skinny one, and the errors are not coefficients. The 4096³ case re-reads A and B through a 16 MiB L2 the model does not know exists. The skinny case puts 32 blocks on 48 SMs and is latency-bound, which a bandwidth term cannot say. Fitting the numbers harder will not fix a model that lacks the mechanism.

What is needed is a **physically grounded model of the mechanisms that produce the timings**: a load is a request that enters a queue, waits for a port, pays a conflict if another request wants the same bank, hits or misses in L2, and returns after a latency that depends on where it was served; a barrier releases warps when the last one arrives; a kernel's time is what falls out of those events. Statistics about the hardware (bandwidths, throughputs) become *outputs* of this model and *checks* on it, not its content.

Three facts make this the natural next layer for this project rather than a side project:

1. **The framework already says the bottom level is this.** Paper §3.1, level 5: "an access becomes requests to banks, ports and bounded queues." We have been evaluating level-5 claims with a level-2 evaluator.
2. **The verifier already has half of it.** `smem_verify.py`'s two-warp event model is an untimed event-driven simulator: loads, barriers, computes, interleavings. A mechanistic performance model is the same event model with time and resource contention added. The right architecture is therefore **one event semantics per template, read two ways**: the verifier explores all admissible interleavings for correctness; the evaluator runs the arbitrated one for time. That is condition K1 again, at the simulator level, and a stronger story than a separate simulator bolted on.
3. **The microbenchmarks become calibration and validation, not the model.** Each of B1–B12 isolates one mechanism. A simulator that has the mechanism must reproduce the benchmark; one that does not cannot be tuned into it.

---

## 1. Scope

**In scope for this round:** one SM-centric model of a Blackwell-class GPU at the level the tiling template and its successors need: warps, the register file as a capacity, shared memory with banks and ports, the L1/L2/DRAM hierarchy with capacities, latencies and bandwidths, asynchronous copies, barriers, block scheduling onto SMs, and occupancy. Calibrated to the GB10 first, then the RTX 5090 with the same mechanism set and different parameters.

**Explicitly out of scope:** instruction-level pipeline modeling (issue slots, dependency stalls beyond a per-warp latency), warp schedulers' exact policies, tensor-core internals (the tensor core is a functional unit with a measured shape and rate), TLBs, power. These can be added as mechanisms later; the architecture must allow it, this round does not build it.

**Fidelity target:** within 10 % of the Spark on every design the tiling template can produce (the 722-design menu on three workloads), with the sign of each regime (memory-, latency-, compute-bound) correct. Today's model is at −33 % to +67 %.

---

## 2. The mechanism inventory

Each mechanism is one Python class with (a) a functional behavior the verifier can explore and (b) a timing behavior the evaluator runs. Parameters come from the measured contract; the column on the right says which benchmark fixes each.

| # | Mechanism | Functional behavior (verifier) | Timing behavior (evaluator) | Parameters (from) |
|---|---|---|---|---|
| M1 | **Warp** | a sequence of operations: issue load, wait for a copy group, barrier, compute, store | each operation has an issue cost; a warp blocks on waits and barriers; up to `max_warps_per_sm` resident | `warp_size`, `max_blocks_per_sm_measured` (B1, B9); `warps_to_hide_*_latency`, `per_warp_outstanding_loads`, `schedulers_per_sm`, `op_latency_cycles` (B17) |
| M2 | **Register file** | capacity only: a block's registers × warps must fit | occupancy limit; determines resident blocks per SM | `regs_per_sm`, the register cliff (B9); `spill_penalty_curve`, `local_mem_*` (B19) |
| M3 | **Shared memory** | an array of slots; a write makes a value visible after completion; reads return the current value (this is the verifier's `lifetime`/`readiness` state) | `banks` banks, `ports` ports; a warp access is split into bank requests; conflicts serialize with the measured per-stride penalty; bandwidth cap per SM | `smem_bytes_per_sm`, `smem_bytes_per_s_per_sm`, `bank_conflict_penalty[stride]` (B6) ; `l1_bytes_by_carveout`, `l1_latency_ns`, `l1_smem_pool_bytes` (B14); `ld_shared_bytes_per_s_per_sm` by width (B21); `functional.smem_bank_map` (B26) |
| M4 | **Asynchronous copy (`cp.async`, TMA)** | a load is *issued* and later *completes*; a wait on a copy group blocks until completion; between issue and completion the destination slot is undefined (this is what makes the T6 race real) | issue cost; completion after latency(bytes) + queueing behind earlier copies on the same SM; throughput cap | `async_copy_latency_s[bytes]`, `async_copy_bytes_per_s`, `tma_bulk_latency_s` (B8) ; `cp_async_groups_in_flight`, `cp_async_issue_cycles_per_16B`, `cp_async_latency_dist`, `tma_cp_async_shared_path`, `mbarrier_s`, `tma_2d_latency_s` (B15); `cp_async_*_allocates_l1` (B14); `functional.cp_async_chunk_order`, `.cp_async_window_cycles`, `.tma_early_bytes_observed`, `.wait_group_overcomplete_observed` (B26) |
| M5 | **L1 / L2 / DRAM hierarchy** | functional: addresses map to lines; a load returns the value (the verifier treats memory as a map) | a request is looked up in L1 (per SM), then L2 (shared, `l2_bytes_effective`, sliced), then DRAM; each level has latency, bandwidth and a queue; L2 uses LRU over sets at line granularity; DRAM has a bandwidth cap and a read/write asymmetry | `latency_ns[l1,l2,dram]`, `l2_bytes_effective`, `l2_bytes_per_s`, `dram_bytes_per_s`, `dram_bytes_per_s_write`, line size (B4, B5) ; `l2_sector_bytes`, `l2_ways`/`l2_sets` (set conflicts), `l2_slice_hash_bits`, `l2_replacement`, `l2_chase_capacity_bytes` (B13); `dram_channels`/`dram_channel_select`, `dram_locality_bytes_per_s`, `dram_rw_turnaround_penalty`, `dram_drift_60s` (B18); `per_sm_outstanding_requests` (B17) |
| M6 | **Barrier** | all warps of a block must arrive before any proceeds (the verifier's barrier event) | release at last arrival + `barrier_s[threads]` | `barrier_s` (B7); `functional.threadfence_visibility_cycles` (B26) |
| M7 | **Compute unit** | performs the tile multiply–accumulate; functional value is the reference arithmetic | FP32: `fp32_flops_per_s_per_sm` with ILP-limited issue; tensor: `tensor_flops_per_s_per_sm[precision]` per `tensor_shapes` | B2, B3; `inst_throughput_per_sm`, `compute_block_efficiency` with the SASS mix (B21); per-shape tensor rates, `tensor_op_latency_cycles`, `ldmatrix_bytes_per_s_per_sm` (B20); `functional.fma_contraction_default`, `.ftz_default`, `.tensor_accumulate_ulp_bound` (B26) |
| M8 | **Block scheduler** | blocks of the grid are assigned to SMs; a block occupies an SM until it finishes | greedy: a block starts on the first SM with free capacity (M2, M3, M1 limits); this is what makes the skinny case latency-bound (32 blocks, 48 SMs, no overlap partner) | `sms`, M1–M3 limits (B1, B9); `block_dispatch_order` (SM pairs round-robin over 6 groups of 8, deterministic first wave), `block_refill_latency_s`, `block_dispatch_rate_per_s`, `wave_staircase` (B16) |
| M9 | **Launch** | — | fixed cost per kernel | `launch_s` (B10); `kernel_gap_s`, `graph_kernel_gap_s`, `concurrent_kernels`, `l2_persists_across_kernels` (B25) |
| M10 | **Clock** | — | all cycle counts convert to time at the sustained clock; variance reported, not modeled | `clock_hz_sustained`, variance (B11); `clock_warmup_s`, `clock_hz_sustained_by_load`, `clock_governor_limit`, `clock_after_gap_hz` (B22) |
| M11 | **Unified memory (Spark only)** | functional: same map | a page's first touch from the GPU pays the migration cost; pinned-host and managed allocations have their own bandwidth caps | `unified_memory`, `unified_memory_penalty` (B12); `um_first_touch_s`, `um_prefetched_bytes_per_s`, `um_translation_overhead`, `um_host_resident_bytes_per_s` (B23); `dram_bytes_per_s_under_cpu_load` (B18) |
| M12 | **Atomics** (new, B24) | an atomic returns the pre-value and applies the op at L2 (global) or in shared memory | serialized per address at the measured same-address rate; distinct addresses at the measured per-thread rate; latency `atomic_latency_ns` | `atomic_throughput_per_s`, `atomic_contention_curve`, `atomic_latency_ns` (B24) |

The two missing mechanisms behind the Spark errors are **M5's L2** (the 4096³ case) and **M8 with M4's latency** (the skinny case). Both are in the first build stage.

---

## 3. One event semantics, two readers

### 3.1 The event model

`agentic/sim/events.py` defines the event vocabulary shared by both readers:

```
ISSUE_COPY(warp, src_region, dst_slot, bytes, group)     # cp.async / TMA issue
COPY_DONE(group)                                          # completion (ordering: per-SM queue)
WAIT(warp, group)                                         # cp.async.wait_group
BARRIER_ARRIVE(warp, id) / BARRIER_RELEASE(id)
LOAD(warp, addr, bytes) / STORE(warp, addr, bytes)        # register-level global access (coarse templates)
COMPUTE(warp, slot_reads: [...], flops, unit)             # a tile MAC step; names the slots it reads
```

A template's `program` (already a string in the contract, e.g. Listing 1's loop) is compiled into a per-warp event sequence by the template's `events.py`. This replaces, and must agree with, the ad-hoc `build_program` in `smem_verify.py`.

### 3.2 The verifier reads it untimed

`agentic/sim/explore.py`: given the event sequences and the *admissible reorderings* (a `COPY_DONE` may occur any time after its `ISSUE_COPY` and before the matching `WAIT` returns; warps interleave arbitrarily between barriers), enumerate interleavings (bounded) or encode them for Z3, and check the schedule obligations (`readiness`, `lifetime`, `terminal_agreement`) as today. This is `smem_verify.py` refactored onto the shared event model. Its verdicts must not change: the S1 tests (24-point family exactness, the T6 race) are the regression gate.

### 3.3 The evaluator reads it timed

`agentic/sim/run.py`: a discrete-event simulator. Each mechanism M1–M11 is a resource with a queue and a service rule; events are scheduled on a global time heap; the arbitrated interleaving is the one the resources produce. Outputs:

- total kernel time (cycles and seconds at the sustained clock);
- per-resource utilization computed from *simulated* activity (bytes actually served by DRAM after L2 hits, not modeled traffic), which fixes the regime-label artifact in `cuda.py`;
- a **trace**: every event with its timestamp, which the console's schedule view draws directly (round 4's "modeled durations" become simulated ones, per phase, not aggregates);
- the **critical path**: for each phase, which resource the slowest warp waited on; this is the regime, mechanistically (`dram`, `l2`, `smem_bank`, `copy_latency`, `barrier`, `compute`, `scheduler`), a finer vocabulary than compute-/memory-bound.

### 3.4 The consistency condition

The timed run is one interleaving among those the untimed reader explores. Therefore: **if the verifier proves an obligation over all admissible interleavings, the simulated run satisfies it**; and conversely, a simulated run that violates `lifetime` is a bug in one of the two readers. Test: for every design in the 24-point family, run the timed simulator and check the obligations on its trace; they must agree with the family verdict. This is K1 at the simulator level, and it is the test that keeps the two readers from drifting apart.

---

## 4. Calibration and validation by microbenchmark

Each benchmark becomes a *simulator test*: the simulator runs the benchmark's own kernel (as an event sequence) and must reproduce the measured number within the benchmark's CI plus a stated tolerance.

| Benchmark | What it fixes | What it validates | Tolerance |
|---|---|---|---|
| B2 FMA | M7 rate | the compute unit alone | 2 % |
| B4 bandwidth vs working set | M5 capacities and bandwidths | L1/L2/DRAM plateaus **and the knee** (the simulator must place the knee at 16 MiB without being told) | 5 % per plateau; knee within one working-set step |
| B5 latency | M5 latencies | pointer chase at each rung | 5 % |
| B6 bank conflicts | M3 banks/ports | penalty vs stride curve | 5 % per stride |
| B7 barrier | M6 | cost vs block size | 5 % |
| B8 async copy | M4 latency and throughput | latency vs tile size, one in flight; throughput with many in flight | 5 % |
| B9 occupancy | M1–M3 limits | resident blocks vs smem and registers | exact |
| B12 unified memory | M11 | managed vs device bandwidth | 10 % |
| B13 L2 structure | M5 L2 fill granularity, set structure, slicing, replacement | sector 64 B fills; no set conflict up to 64 lines at any power-of-two stride (hashed sets or >= 64 ways); slice latency spread 5 ns (no near/far); LRU-like, chase capacity 24 MiB vs B4's 16 MiB streaming knee | structure: exact classification; latencies 5 % |
| B14 L1 carve-out | M5 L1 capacity per smem request, M4 L1 allocation of copies | 32 KB L1 up to 48 KB smem, 16 KB at 64–96 KB; cp.async.cg/.ca and TMA do not allocate in L1 | capacity within one step; booleans exact |
| B15 copy queue | M4 pending groups, issue cost, completion distribution, shared TMA/cp.async path | >= 22 groups pending without an issue stall; 127 cycles per 16 B per warp issue; p50/p99 per tile and residency; cp.async 1.89 TB/s, TMA 1.85, mixed 1.62 (shared path) | 5 % on medians, 25 % on p99 |
| B16 block scheduler | M8 order, rate, refill | first wave: SM pairs round-robin over 6 groups of 8, deterministic; later waves follow completion order; refill 192 ns; staircase at 48 blocks | order exact; refill 10 % |
| B17 latency hiding | M1 warps-to-hide, MLP, schedulers, op latencies | 8 warps/SM saturate DRAM, 24 for L2; FMA/IMAD 4 cycles, LDS 33, L1 hit 44 | 5 %; warps within one step |
| B18 DRAM | M5 DRAM channel hash, locality, R/W turnaround, CPU contention, drift | no single address bit selects a channel (hashed); random 128 B = sequential; turnaround 2.7 %; 159 of 235 GB/s under CPU load; 5.7 % drift | 5 % |
| B19 registers/local | M2 spill curve, local memory | no spill at 71 regs; 64 regs 7x slower, 32 regs 170x; local latency 15.5 ns (L1) | 10 % |
| B20 tensor detail | M7 per-shape rates, FP8, tensor latency, ldmatrix | fp8 e4m3 5.09 TFLOP/s/SM, bf16 2.55, tf32 1.27, int8 2.88; mma latency 29 cycles; ldmatrix swizzled 349 GB/s/SM vs 44 unswizzled | 5 % |
| B21 compute block | M7 opcode rates, the generated inner loop | the T5/T1 inner loops run at 45–46 % of the FFMA bound with the smem tile pre-filled (LDS + FFMA issue), R2 at 14 % | 5 % |
| B22 clocks | M10 warm-up, load dependence | no warm-up (clock starts at 2480 MHz and settles to 2437–2483 by load); no power or thermal cap; after a 1 s gap the first kernel runs at the idle 2483 | 1 % |
| B23 unified memory | M11 first touch, prefetch, host-resident | first touch per page; prefetched 161 GB/s; host-resident 162 GB/s; managed/device 1.5x with the CPU idle | 10 % |
| B24 atomics | M12 | same-address global 0.74 G/s (serialized at L2), per-thread 27 G/s; shared same-address 2.5 G/s; latency 159 ns global, 54 ns shared | 10 % |
| B25 multi-kernel | M9 gaps, concurrency, L2 persistence | 2.46 us between dependent kernels, 0.51 us in a graph; two 24-block kernels co-run; L2 persists across launches | 10 % |
| B26 functional | the verifier's event semantics (M3–M7) | cp.async chunks land in issue order, never torn; wait_group N leaves later groups possibly complete; smem stores visible after syncthreads, named barrier, or fence alone (243 cycles); TMA bytes visible before the mbarrier completes (30 B per 16 KB tile on average); a*b+c contracts, no FTZ; tensor accumulate rounding bound per precision; 32 x 4 B banks with half/quarter-warp wavefronts; divergent branches did not interleave | exact |

**Validation on real kernels, the gate for the round:** the five accepted templates of the `spark-001` run, plus the 722-design menu sampled at 30 points across the three workloads, simulated and compared with Spark measurements (the runner from F3, with clocks sampled during the run). Targets: median absolute error ≤ 10 %, worst ≤ 20 %, and the critical-path resource agrees with `ncu`'s stall reasons on the 30 sampled designs for at least 24 of them. The two known failures are the headline tests: **4096³ within 10 % (needs L2 reuse), skinny within 10 % (needs latency-bound single wave).**

Calibration is **not** fitting. Parameters are taken from the contract; the only free choices are structural (L2 set associativity and slicing, copy-queue depth, scheduler policy), and each is chosen by the benchmark that exposes it (B4's knee shape, B8's throughput curve, B9). Since B13–B26 (`calibration/suite/B13_plus.md`) the L2 choices are no longer free: B13 found 64 B fills, no observable set conflicts (hashed sets), no slice latency spread and LRU-like replacement with a 24 MiB single-warp capacity against B4's 16 MiB streaming knee, so G2's phenomenological LRU-over-regions stands with those facts stated; B16 fixes the scheduler's first-wave order (SM pairs round-robin over six groups of eight); B15 fixes the copy window (at least 22 groups pending per block, 127 cycles per 16 B per warp to issue) and recorded in `sim/config/<host>.yaml` with the benchmark that justified it. If a design cannot be matched without changing a measured parameter, that is a missing mechanism, and it is logged as such.

---

## 5. Configurability: the chip as a design space

The point of a mechanistic model is that its parameters are physical, so they can be *varied*. `sim/config/<host>.yaml` has three kinds of field:

| Kind | Examples | Can be varied? | How the orchestrator treats a template that changes it |
|---|---|---|---|
| **Measured** | latencies, bandwidths, rates, penalties | yes, but the result is extrapolation | regime strength capped at `observed`; narration says "simulated, not measured" |
| **Structural** | banks, ports, L2 sets/slices, copy-queue depth, SM count, smem per SM | yes; these are the chip-level decisions of paper levels 6–7 | same cap; and the *verifier* must re-prove any template whose obligations mention the changed structure (banks and ports appear in level-5 obligations) |
| **Fixed by physics** | line size, warp size, bytes per element | no | a change is rejected as malformed |

This is what makes the GB10 model a *starting point* rather than a description: the same simulator with `sms: 96, dram_bytes_per_s: 4.7e11` is a hypothetical chip, evaluated with the same mechanisms, and the console labels every number from it as simulated. The Ventus and CGRA testbeds then become the places where such a hypothetical chip can be built and the simulator's extrapolation checked. The RTX 5090 is the first check that is free: same mechanisms, parameters from the same suite, measurements from the same runner.

---

## 6. Interfaces

- **Evaluator backend `sim:<host>`** beside `roofline` and `cuda:<host>`: `measure(T, x)` compiles the design's events, runs `sim/run.py` with `sim/config/<host>.yaml`, returns a `measurement` with `simulator: "sim:spark-6951-gb10"`, utilizations from simulated activity, the critical-path resource per phase, and the trace artifact id.
- **Verifier**: `schedule_obligations` in `smem_verify.py` reads events from `sim/events.py` instead of `build_program`. Verdicts unchanged.
- **Console**: the schedule view (round 4) takes per-phase durations from the trace when a `sim:` or `cuda:` measurement exists; the resource strip gains the critical-path resource per phase as a colored band; the narration's regime clause uses the finer vocabulary ("latency-bound: warps waited on copy completion in 29 of 32 phases").
- **Orchestrator**: the regime handler's `probe()` uses the simulator over the menu when the backend is `sim:`; `regime_of()` maps critical-path resources to the catalog's regimes.
- **Paper**: §6.1 gains the simulator and its validation table; §6.6's modeled 58–63 % error is replaced by the measured −33 % / +67 % of the aggregate model and the simulator's error after this round.

---

## 7. Build order and acceptance tests

| Stage | Deliverable | Acceptance test |
|---|---|---|
| **G0** Shared events | `sim/events.py`; `templates/smem_tiling/events.py`; `smem_verify.py` refactored onto it | S1 tests pass unchanged: 24-point family exactness, T6 refuted with the same obligation, T7 coverage refuted. `build_program` deleted. |
| **G1** Timed core | `sim/run.py` with M1, M6, M7, M8, M9, M10 (warps, barriers, compute, scheduler, launch, clock) | Reproduces B2 within 2 % and B7 within 5 %; a compute-only kernel's time equals `flops / rate + barriers` within 2 %; the consistency test (§3.4) passes on the 24-point family. |
| **G2** Memory hierarchy | M5 with L1, L2 (LRU, sets, slices), DRAM queues | Reproduces B4's three plateaus within 5 % and places the knee at 16 MiB; B5 latencies within 5 %; **the 4096³ GEMM within 10 % of the Spark** (was −33 %). |
| **G3** Shared memory and async copy | M3 banks/ports, M4 copy queue | B6 penalty curve within 5 % per stride; B8 latency and throughput within 5 %; **the skinny GEMM within 10 %** (was +67 %); the unsound design's race appears in the timed run when a warp is delayed by the race-amplifier (`sim` and `cuda` agree). |
| **G4** Validation set | 30 sampled designs × 3 workloads, measured on the Spark with in-run clock sampling, simulated, compared; `calibration/spark-6951/validation.json` | Median error ≤ 10 %, worst ≤ 20 %; critical path agrees with `ncu` stall reasons on ≥ 24/30; the console shows simulated vs measured per design. |
| **G5** Configurability | `sim/config/` schema with measured/structural/fixed kinds; orchestrator extrapolation rule; console labels | A template on a hypothetical chip (`sms: 96`) runs end to end with every number labeled "simulated"; the verifier re-proves a template whose scope changes `banks`; the RTX 5090 config, when available, passes G1–G3 with its own contract and no code change. |
| **G6** Occupancy and unified memory | M2 register cliff; M11 | B9 exact; B12 within 10 %; a design using managed memory is simulated with the penalty. |

**Review gate:** for the 4096³ and the skinny GEMM, Li can read the critical-path band in the schedule view and say, in one sentence each, what the hardware is waiting on; and the simulator's explanation agrees with `ncu`.

---

## 8. What this is not

It is not Accel-Sim or GPGPU-Sim. Those model the instruction pipeline and take minutes per kernel; this models the mechanisms the templates reason about and must run in milliseconds, because it sits inside the optimization loop (the regime probe sweeps 722 designs). The level of detail is set by the templates' vocabulary: a mechanism enters the simulator when a template's decisions or obligations refer to it. Tensor-core internals enter with the tensor-core template; instruction scheduling enters with the level-5 scheduling template; NoC routers enter with the system-architecture template on the CGRA. The architecture (§3) is designed so that each addition is one class and one benchmark.

---

## 9. Questions for Li

1. **Tolerance vs speed.** 10 % median error is the proposed gate. Tighter needs instruction-level detail and will cost the millisecond budget. Is 10 % the right number for the regime probe, with the real GPU as the final check?
2. **L2 structure.** The GB10's L2 slicing and set associativity are not public. G2 infers them from B4's knee shape. If the inferred structure is wrong but reproduces B4, do we accept it as a phenomenological model of L2, labeled as such, or hold the gate until it is confirmed by a dedicated microbenchmark (set-conflict sweeps)?
3. **Who owns the critical-path vocabulary.** The catalog's regimes are compute-/memory-bound. The simulator will produce seven resources. Should the catalog gain the finer regimes now, or should the orchestrator map them to the two until the catalog is revised?
