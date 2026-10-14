# Calibration suite, part 3: B27–B29 and the warp issue model

*v0.1 · October 15, 2026 · Li Shang, drafted with Claude. Companion to `B13_plus.md` and `notes/mechanistic_simulator_design.md`. Written for Claude Code to execute after the G4 result of commit `e7e969b`.*

---

## 0. Where G4 stopped, and why

G4 ran the 30-design validation set on spark-6951 and did not pass its gate: median |error| 13.9 %, worst 65 %, 9 of 30 within 10 %, 24 of 30 within 20 %. The errors are not scattered. They sit in two corners of the menu:

- **Tiny tiles at high occupancy** (16-wide tiles, four or more resident blocks per SM): the simulator is **too slow**, +47 % to +65 % (`16×32×8 S2 P0` on the skinny GEMM, `256×16×32 S2 P0` on the ragged one).
- **Many-block mid tiles** (`32×64×32 S2 P1` and `32×128×32 S3 P0` on 4096³, 8192 blocks, four resident per SM): the simulator is **too fast**, −20 % to −42 %.

`ncu` says what the hardware is doing in both: the dominant stall reason is `not_selected`, with `mio_throttle` and `short_scoreboard` next. Warps are eligible and the four warp schedulers per SM are the limiter, not the FMA pipe and not the LSU bandwidth. The limiter is instruction issue, which §1 of the simulator note put out of scope and gap #6 then had to bring in as G4a.

**Diagnosis of G4a's issue model.** It charges each warp-instruction an issue cost on one of four schedulers and sums them. Two mechanisms are absent, one fit is present:

1. **No per-warp dependencies.** The model sees issue slots and assumes eligible warps fill them. With a 16-wide tile the inner loop is short: each k-step is a handful of `ld.shared` followed by the FFMAs that consume them, and with the measured latencies (LDS 33 cycles, FFMA 4) there is not enough independent work in the loop to cover the load latency. Each warp stalls on its own scoreboard, and a stalled warp costs the scheduler nothing: other warps issue in its place. The model has no scoreboard, so it charges every instruction as scheduler occupancy, including the ones that on the card are merely *waiting*; with many warps per scheduler it sums costs that the hardware overlaps. Net effect: too slow on tiny tiles. (The fitted overlap factor was an attempt to paper over exactly this.)
2. **No shared LSU / shared-memory port arbitration across blocks.** Issue is modelled per SM, but the LSU queue and the shared-memory ports are charged per block. With four resident blocks the card queues their `ld.shared` and `cp.async` traffic on one LSU (`mio_throttle` rises); the model lets each block proceed at its single-block rate. Net effect: too fast on many-block mid tiles.
3. **A fitted number.** `issue_overlap_by_warps_per_scheduler: {6: 0.70, 8: 0.75}` in `sim/config/spark-6951-gb10.yaml` was set from two B21 points at 3–4 blocks per SM and labelled phenomenological. It is a fit against the quantity the model then predicts, which §4 of the design note forbids. It goes.

So the round is: measure the two missing mechanisms (B27, B28), replace the issue model with a warp model whose parameters come only from those measurements and B17/B21, check it against ground truth the model has never seen (B29), and only then rerun the 30 designs.

Conventions as in B13+: one CUDA program each under `calibration/suite/bench/`, JSON lines, cycles beside seconds, `driver.py --micro3`, `fit_micro.py` fitters, provenance in the contract, no address or key in the repository. Counters are available (P0 done), so each benchmark carries its `ncu` cross-check.

---

## 1. Benchmarks

### B27 `b27_dependency_ilp` — dependency chains vs ILP vs TLP: the scoreboard parameters

**Measures:** how many independent instructions a warp needs in flight to hide the `ld.shared` latency, and how much thread-level parallelism substitutes for instruction-level parallelism on one scheduler.

**Mechanism:** M1 (warp) revised: per-warp scoreboard; M7 issue.

**Kernel sketch and procedure.**
- One block per SM; the block's warps are pinned by count so that there are exactly `w` warps per scheduler, `w` ∈ {1, 2, 4, 8} (block of 4·`w` warps, one block per SM; verify with `%smid` and `ncu smsp__warps_active`).
- The inner loop mimics the template's k-step: `a` independent accumulators, each updated by an FFMA that depends on an `ld.shared` issued at the top of the iteration (`acc[i] = fma(s[i], b, acc[i])`), loop length `L` iterations unrolled, `a` ∈ {1, 2, 4, 8, 16}, `L` ∈ {4, 8, 16, 32, 64}. Shared memory pre-filled; the loads are conflict-free (stride 1) so the LSU is not the limiter at `w` = 1.
- Three variants: (i) load→FFMA dependent as above; (ii) FFMA-only chain (no loads) at the same `a` — the pure-latency reference, which must reproduce B17's 4-cycle FFMA latency and B2's throughput at `a` ≥ 4; (iii) loads only, `a` outstanding — must reproduce B17's LDS 33 cycles.
- Record cycles per iteration per warp (`clock64`), and warp-instructions issued per SM cycle.

**Pass criterion.** Variant (ii) at `a` = 1 gives 4 cycles per FFMA ± 0.5; at `a` ≥ 4, `w` = 1 reaches ≥ 90 % of one instruction per cycle per scheduler. Variant (iii) at `a` = 1 gives 33 ± 3 cycles. Variant (i) shows a monotone curve in `a` and in `w`, and the (`a`, `w`) product at which it reaches 90 % of the issue bound is the headline number: the instructions-in-flight needed to hide LDS latency.

**What it yields.** The warp model's two parameters: the per-warp scoreboard (latency per op class from B17, confirmed here) and the issue rule (a warp is eligible when its next instruction's operands are ready; the scheduler issues one eligible warp-instruction per cycle). Nothing else: TLP-vs-ILP is then a *prediction* of the model, and the measured curve for variant (i) over the whole (`a`, `w`, `L`) grid is the first check of it. **This curve replaces `issue_overlap_by_warps_per_scheduler`.**

**Contract:** `scoreboard_latency_cycles` (fma, lds, confirms B17), `issue_bound_fraction[a,w,L]` (the full grid, raw), `ilp_to_hide_lds` (the headline).
**Spark time:** 3 min. **ncu:** `smsp__issue_active.avg.pct_of_peak_sustained_active`, `smsp__warp_issue_stalled_short_scoreboard_per_warp_active`, `..._not_selected_...`, `..._wait_...`.

### B28 `b28_lsu_port_contention` — LSU and shared-memory ports shared across resident blocks

**Measures:** the per-SM limit on shared-memory and asynchronous-copy traffic when several blocks share one SM, and how the hardware divides it among them.

**Mechanism:** M3 (ports), M4 (copy queue), M1: resources arbitrated per SM across all resident blocks, not per block.

**Kernel sketch and procedure.**
- Each block runs the template's real per-warp pattern: per k-step, `ld.shared.128` × 4 and `ld.shared.32` × 8 per thread (the T5 widths from B21's SASS mix), plus one 16 B `cp.async` per thread per phase into a second buffer with `wait_group 1`, 256 threads. Tile in shared memory pre-filled; global source L2-resident (8 MiB) so DRAM is not the limiter.
- Sweep resident blocks per SM `b` ∈ {1, 2, 3, 4, 6, 8} (force with dynamic shared-memory padding; confirm with `ncu sm__ctas_launched` / B9's table). Grid = 48·`b`, all blocks the same work, one wave.
- Measure per block: cycles per k-step; per SM: aggregate shared-memory bytes per cycle and aggregate `cp.async` bytes per second.
- Three mixes: LDS only; `cp.async` only; both (the real pattern). The "both" case against the sum of the two pure cases says whether loads and copies share the LSU path.
- Arbiter classification: at the plateau, the per-block rates. Equal shares within 10 % → `fair-share`; one block at its single-block rate and the others starved → `priority`/`age`; report the per-block distribution either way.

**Pass criterion.** At `b` = 1 the LDS rate matches B21's `ld_shared_instr_per_s_per_sm` within 5 % and the `cp.async` rate matches B8/B15 within 5 %. The aggregate plateaus for `b` ≥ some `b*` (report `b*`); the plateau is the port limit.

**Contract:** `smem_port_bytes_per_cycle_per_sm` (plateau), `lsu_shared_by_copies` (bool, from the mixed case), `copy_bytes_per_s_per_sm_by_residency[b]`, `smem_arbiter` (enum + per-block distribution), `lds_rate_by_residency[b]`.
**Spark time:** 3 min. **ncu:** `l1tex__data_pipe_lsu_wavefronts_mem_shared`, `smsp__warp_issue_stalled_mio_throttle_per_warp_active`, `l1tex__throughput`.

### B29 `b29_inner_loop_ground_truth` — the generated kernels at the bad corners, under counters

**Measures:** nothing new in the contract. This is the **check set**: the loops the warp model must reproduce without any parameter changed.

**Mechanism:** none; validation of M1/M3/M7 together.

**Kernel sketch and procedure.**
- Take the generated kernels (from `codegen`) for the four worst G4 designs and the two spark-001 references: `16×32×8 S2 P0` (skinny), `256×16×32 S2 P0` (ragged), `32×64×32 S2 P1` and `32×128×32 S3 P0` (4096³), plus T5 `128×128×32` and T1 `128×256×16`. Strip the global loads (tile pre-filled, as B21 does), keep everything else: the `ld.shared` widths, the FFMA block, the per-phase overhead instructions, the barriers.
- Run each at `b` ∈ {1, 2, 4, 8} resident blocks per SM (force with padding), one wave.
- Record per (kernel, `b`): cycles per k-step per block, and the full `ncu` issue and stall breakdown: `smsp__issue_active`, `smsp__inst_executed`, `smsp__warp_issue_stalled_{not_selected, short_scoreboard, long_scoreboard, mio_throttle, wait, math_pipe_throttle, barrier}_per_warp_active`, `sm__pipe_fma_cycles_active`, `l1tex__data_pipe_lsu_wavefronts_mem_shared`.
- Then the same six kernels with the global loads **kept** (the real kernels), same `b` sweep, same counters: this is what G4 measured end to end, now with per-block-residency resolution.

**Pass criterion for the benchmark itself:** T5 and T1 at `b` = 1 reproduce B21's `compute_block_efficiency` within 5 %.

**Pass criterion for the warp model (the point of B29):** with parameters from B27/B28/B17/B21 only, the simulated cycles per k-step are within 10 % of B29 on all 24 (kernel, `b`) points, and the simulated dominant wait family matches the `ncu` dominant stall family on ≥ 20 of 24. **A miss is a gap-log entry, not a knob.**

**Contract:** `inner_loop_ground_truth` block (the 24-point tables, raw), under `counter_crosscheck:`.
**Spark time:** 5 min. **ncu:** as listed.

---

## 2. Warp issue model (M1/M7 revision)

Replaces `sim/issue.py`'s additive occupancy with a mechanism. One class, `sim/warp.py`, used by the timed reader for every compute phase; the untimed reader is unchanged (it does not care about timing).

**State per warp:** a program counter into the phase's instruction sequence (from the template's `events.py`, which already carries the per-k-step counts: `ld.shared` by width, FFMA, per-phase overhead from the SASS), and a scoreboard: for each pending result, the cycle at which it becomes available.

**Instruction classes and their latencies** (from B17, confirmed by B27): FFMA 4, IMAD 4, `ld.shared` 33 (+ bank-conflict degree × wavefront cost from B6/B26.6), L1-hit load 44, barrier release from M6, `cp.async` issue 124 cycles per 16 B per warp (B15.3) with completion from M4.

**Schedulers:** four per SM (B17), each owning a fixed quarter of the resident warps (warp id mod 4, as the hardware does); each cycle a scheduler issues one instruction from one *eligible* warp (operands ready on the scoreboard; the structural resource below free), choosing the oldest eligible (loose round-robin is acceptable; B27's TLP curve decides between the two if they differ).

**Shared per-SM resources** (this is what fixes the many-block corner): the LSU accepts `ld.shared` wavefronts and `cp.async` chunks at the B28 plateau rate, arbitrated among all resident blocks by the B28 arbiter rule; the FMA pipe accepts one FFMA warp-instruction per scheduler per cycle (B2). A warp whose instruction needs a busy resource is not eligible that cycle (`mio_throttle` in `ncu` terms).

**Outputs per phase:** cycles, and the wait-family histogram per warp (`scoreboard`, `not_selected`, `lsu`, `barrier`, `copy`) so the critical-path family is predicted from the same counters `ncu` reports, not inferred afterwards.

**Cost:** this is cycle-stepped for the compute phase only (a k-step is ~100 warp-instructions; a phase ~10³ cycles; 10³ phases per design), well within the millisecond budget if written with integer arrays rather than per-instruction objects; the memory and copy mechanisms stay event-driven. If it proves too slow for the 722-design regime probe, compute the per-(design, residency) phase cost once with the warp model and cache it; the design space is finite.

**Parameters:** B17, B21 (instruction counts), B27, B28 only. `issue_overlap_by_warps_per_scheduler` and `issue_model: occupancy` are removed from `sim/config/spark-6951-gb10.yaml`. `issue_overhead_per_chunk/per_phase` stay, as they are counts read from SASS, not fits; B29 checks them.

---

## 3. Other fixes required in this round

1. **L2 capacity.** `sim/config`'s `l2_policy: lru-fully-associative` with `l2_bytes_effective` = 16 MiB came from B4's streaming knee. B13 (single-warp chase: 91 % hits at 24 MiB, collapse at 32) and the counter cross-check (87 % L2 hits through 24 MiB, 73 % at 32) show the capacity is the declared 24 MiB with hashed sets and a soft edge, and that the 16 MiB knee is a bandwidth effect of many concurrent streams. Set the model's capacity to `l2_chase_capacity_bytes` (24 MiB), keep LRU over regions, and let the bandwidth queue (not the capacity) produce the 16 MiB streaming knee; B4 must still be reproduced within 5 % with the knee in place, which is now a prediction, not an input. Remove the 16 MiB choice and the "phenomenological" label from the capacity; the label stays on set/slice structure only.
2. **Critical-path agreement criterion.** G4 reports 25/30 agreement, but the simulator predicts `compute` on all 30 designs and `ncu` says compute on 25: the agreement is the base rate. The criterion becomes per family: for each family `ncu` reports as dominant (compute, copy, barrier), the fraction of those designs where the simulator agrees, reported separately, and the gate requires agreement on at least half of the non-compute designs as well as ≥ 80 % overall. With the warp model's wait histogram this is directly comparable.
3. **Copy-queue depth units.** `copy_queue_depth: 370` is in 16 B chunks per SM, derived from the B8 slope; B15.1 measured "≥ 22 pending 4 KB groups per block with no issue stall", a different quantity (groups, per block, and a lower bound). State both in the config in their own units, derive the chunk window implied by B15 (22 × 4 KB / 16 B = 5 632 chunks, so the issue window is not the limit and 370 is a *throughput* window), and rename `copy_queue_depth` to `copy_throughput_window_chunks` so nobody reads it as a queue length.
4. **Gap #2** (B8 copy latency vs tile size not linear in the window model: 8–32 KB +9 % to +20 %). With B15.3's issue cost of 124 cycles per 16 B per warp now measured, charge issue explicitly (it was absorbed in the slope) and re-fit nothing: check whether the mid sizes fall within 5 %. If not, the gap stays open with that result recorded.

---

## 4. Build order

| Step | Deliverable | Acceptance |
|---|---|---|
| 1 | B27, B28 implemented, run on spark-6951, fitted into the contract with provenance | each benchmark's own pass criterion (§1); contract validates |
| 2 | Fixes §3.1, §3.3, §3.4 applied | B4 reproduced within 5 % with the knee as a prediction; G1–G3 tests still pass |
| 3 | `sim/warp.py`; `issue.py` and the overlap factor removed; wait-family histogram wired to the trace and the console | B27's variant (i) grid reproduced within 10 % (it is a prediction now); `tests/test_g3_smem_copy.py`'s 4096³ headline no longer xfail |
| 4 | B29 run; warp model checked against it, **no parameter changed** | 24 points within 10 %, family agreement ≥ 20/24; any miss → `notes/sim_gaps.md` |
| 5 | §3.2 criterion implemented; G4 rerun on the same 30 designs (`--resimulate` first, then re-measure only if the kernels changed) | median ≤ 10 %, worst ≤ 20 %, per-family agreement as in §3.2 |
| 6 | Report | the G4 table, the per-family agreement, the gap log diff |

**Do not start G5** (the simulator inside the regime probe) until step 5 passes.

---

## 5. Hand-off for Claude Code

> Read `calibration/suite/B27_plus.md`. Implement and run B27 and B28 on spark-6951 first, then apply the fixes in §3 (L2 to the measured 24 MiB, copy-window units, gap #2 with the measured issue cost), then replace `sim/issue.py` with the warp model of §2 and delete the overlap factor from the config. Run B29 and check the warp model against it without changing any parameter; a miss goes into `notes/sim_gaps.md`. Implement the per-family agreement criterion, rerun G4 on the same 30 designs, and stop and report with the table, the per-family agreement and the gap-log diff. Do not start G5 unless the median is under 10 % and the worst under 20 %. Parameters are never fitted. No address or key in the repository.

---

## 6. References

B17, B21 and the counter cross-check in `calibration/spark-6951/contract.yaml`; V. Volkov, *Understanding Latency Hiding on GPUs*, PhD thesis, UC Berkeley 2016 (ILP vs TLP curves, the method of B27); Z. Jia et al., arXiv:1804.06826 (dependent-chain latencies); NVIDIA Nsight Compute documentation (stall-reason metrics).
