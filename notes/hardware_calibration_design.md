# Grounding the Hardware Model: Calibration on Real Blackwell GPUs

**Design document, v0.1 · October 14, 2026**
*Li Shang, drafted with Claude. Follows `schedule_view_design.md` (console round 4) and the S4 item of the v1 review. Written for Claude Code to execute; it runs the measurements, since only it can reach the machines.*

---

## 0. Why

Every number the system has produced so far comes from `illustrative-fp32-gpu`: 128 SMs, 1.8 TB/s, 80 TFLOP/s, a 40-cycle barrier. Those are the paper's Listing 1 parameters, chosen to be plausible, not measured. The evaluator's only backend re-evaluates the template's cost model, so "measured equals predicted by construction" and the optimism record can never go dirty. The console's schedule view draws modeled durations and says so.

Two real machines are available:

| Machine | Where | What it is |
|---|---|---|
| **DGX Spark** (`spark-6951`) | Li's NUS office, reachable over SSH from the Mac | NVIDIA GB10 Grace Blackwell: a Blackwell GPU and a 20-core Arm CPU sharing 128 GB of LPDDR5X unified memory at about 273 GB/s |
| **8 × RTX 5090** | remote server | Eight consumer Blackwell GPUs, each with 32 GB GDDR7 at about 1.8 TB/s |

They are the same GPU generation with very different memory systems, roughly a 6× bandwidth gap. The same benchmark suite on both gives two calibrated points that differ mainly where the structure catalog says it matters most: a design that is memory-bound on the Spark may be compute-bound on the 5090. That is a measured regime boundary.

This document specifies three things, in the order they should be built:

1. **A microbenchmark suite** that measures the quantities the hardware contract needs and fits them into a `measured` hardware contract with provenance.
2. **A real evaluator backend** that compiles a solved tiling design into a CUDA kernel, runs it on the machine, checks the result against a reference, and records measured time beside the prediction.
3. **Console changes** so the schedule view and the narration show measured against modeled, and the optimism record becomes meaningful.

A distinction governs all three. A model fitted to a real card is trustworthy near the points measured. **Software knobs** (tiles, slots, prefetch, barriers) can be varied freely, because the card checks every design. **Hardware knobs** (SM count, shared-memory capacity, bandwidth) cannot be varied on the card, so a template whose scope changes them is extrapolating, and the orchestrator must mark its regime assumptions as *observed*, never *proved*. The real GPUs are ground truth for the software side and for calibration; Ventus remains the testbed where the chip itself is a decision.

Access note: Claude Code reaches `spark-6951` with the SSH key already configured on the Mac. No address or key is written into the repository; the host alias lives in the Mac's SSH config, and the suite takes it as a parameter.

---

## 1. Pre-flight (before any code)

Already known from Li's own login (October 14): the Spark runs DGX OS 7.5.0 on kernel 6.17.0-1026-nvidia, aarch64, with a 3.67 TB root disk at 18 % use and a pending firmware upgrade (`fwupdmgr get-upgrades`); SSH from the Mac works with the existing key. The firmware upgrade should be applied before the suite runs, since GPU firmware can change clock behavior, and the version recorded in provenance.

Claude Code, over SSH, records the answers in `calibration/<host>/preflight.md`:

1. `nvidia-smi` and `nvidia-smi -q -d CLOCK,MEMORY,PERFORMANCE`: driver, GPU name, memory, current and max clocks, whether clocks can be locked (`-lgc`; on the Spark this may be unavailable, in which case the suite must report clock variance).
2. `nvcc --version`; CUDA toolkit present and its compute capability target (GB10 is `sm_121`; the 5090 is `sm_120`; confirm with `deviceQuery` or `torch.cuda.get_device_capability()`).
3. `python3 -c "import torch; print(torch.cuda.is_available(), torch.__version__)"`.
4. CUTLASS available (`/usr/local/cuda/include/cutlass` or a clone); cuBLAS present (needed for the reference in §3).
5. `deviceQuery` or the equivalent through `cudaGetDeviceProperties`: SM count, shared memory per block and per SM, registers per SM, L2 size, warp size, max threads per block. These are the *declared* values; the suite measures the *effective* ones.
6. Disk and permissions: where the repo lives on the Spark, whether `rsync` works, and that the account can run a 10-minute job unattended.

If 2 or 3 fail, stop and report; the rest of this document assumes a working toolchain.

---

## 2. The microbenchmark suite

`calibration/suite/`: one small CUDA program per quantity, a Python driver, and a fitter. Each benchmark writes a JSON record `{quantity, value, unit, ci95, n, method, clocks, timestamp}`. The driver runs everything, repeats each measurement until the 95 % confidence interval is under 2 % or 50 repetitions, and writes `calibration/<host>/raw/*.json`. The fitter turns raw records into one hardware contract (§2.3).

### 2.1 What to measure, and which contract field it feeds

| # | Quantity | Method | Feeds |
|---|---|---|---|
| B1 | SM count, warp size, smem per block and per SM, registers per SM, L2 size, max threads per block | `cudaGetDeviceProperties` | `sms`, `smem_bytes_per_block`, `smem_bytes_per_sm`, `regs_per_sm`, `l2_bytes` |
| B2 | FP32 FMA throughput per SM | a kernel of dependent FMAs with enough ILP, one block per SM, timed with `cudaEvent` | `fp32_flops_per_s_per_sm` |
| B3 | Tensor-core throughput per precision and shape (TF32, BF16, FP16, FP8; `mma.sync` shapes) | `cublasGemmEx` at large square sizes, and a `mma.sync` microkernel for the per-SM peak | `tensor_flops_per_s_per_sm[precision]`, `tensor_shapes[precision]` |
| B4 | Global (DRAM) bandwidth: read, write, copy, as a function of working set | streaming kernels over 1 MB … 8 GB; the knee gives the effective L2 capacity | `dram_bytes_per_s`, `l2_bytes_effective`, `l2_bytes_per_s` |
| B5 | Memory latency at each rung: L1, L2, DRAM | pointer chasing over working sets from 4 KB to 1 GB with random stride | `latency_ns[l1,l2,dram]`, line size |
| B6 | Shared-memory bandwidth and bank-conflict penalty | `ld.shared` throughput for strides 1, 2, 4, 8, 16, 32 words | `smem_bytes_per_s_per_sm`, `bank_conflict_penalty[stride]` |
| B7 | Barrier cost | `__syncthreads` in a loop, as a function of block size (128, 256, 512, 1024 threads) | `barrier_s[threads]` (replaces the invented 40 cycles) |
| B8 | Asynchronous copy: `cp.async` and, where supported, `cp.async.bulk` (TMA) latency and throughput for tile sizes 4 KB … 64 KB | issue-to-completion timing with `cp.async.wait_group` | `async_copy_latency_s`, `async_copy_bytes_per_s`, `tma_supported` |
| B9 | Occupancy limits: blocks per SM vs registers per thread and smem per block | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` plus a measured sweep | `max_blocks_per_sm`, the register cliff |
| B10 | Kernel launch overhead | 1000 empty launches | `launch_s` |
| B11 | Clock behavior under sustained load | B2 run for 60 s with `nvidia-smi dmon` sampling | `clock_hz_sustained`, variance; whether `-lgc` held |
| B12 | **Spark only:** unified-memory access cost from the GPU for CPU-resident and GPU-resident allocations, and page-migration cost | streaming over `cudaMallocManaged` buffers touched by CPU first vs GPU first | `unified_memory_penalty`, `coherent_access_bytes_per_s` |

B1–B7 are the minimum for the tiling template's cost model (`cost_model` uses `sms`, `smem_bytes_per_block`, `dram_bytes_per_s`, `fp32_flops_per_s_per_sm`, `barrier_s`, `bytes_per_element`). B8 is what makes the prefetch model honest, since the template's loads are asynchronous. B9–B12 are needed before the model is trusted on a new workload or on the Spark's memory system.

### 2.2 Prior art, so the suite is citable

The method is the one used for every recent generation: Jia et al. dissected Volta through Ampere by microbenchmark; Luo et al. did Hopper; Accel-Sim's tuner calibrates a simulator the same way. There is no public dissection of the GB10 or of the consumer Blackwell parts yet, so `calibration/<host>/` is itself a publishable artifact. Each benchmark's README cites the method it follows.

### 2.3 The measured hardware contract

The contract schema gains a `measured` variant. `scope.chip` of a template may name either:

```yaml
name: spark-6951-gb10            # or rtx5090-measured
kind: measured                    # vs. illustrative | ventus
sms: 48                           # from B1; every value below carries its source
smem_bytes_per_block: 101376
dram_bytes_per_s: 2.41e11         # B4, effective, not the datasheet 273 GB/s
fp32_flops_per_s_per_sm: ...      # B2
barrier_s: {256: 1.9e-7, 512: ...}   # B7, by block size
async_copy_latency_s: ...         # B8
tensor_flops_per_s_per_sm: {bf16: ..., tf32: ..., fp8: ...}   # B3
latency_ns: {l1: ..., l2: ..., dram: ...}                      # B5
provenance:
  suite_commit: <sha>
  driver: ...
  toolkit: ...
  clocks: {locked: false, sustained_hz: ..., variance: 0.03}
  date: 2026-10-xx
  raw: calibration/spark-6951/raw/
fixed: [sms, smem_bytes_per_block, dram_bytes_per_s, ...]   # cannot be varied on this machine
```

The numbers above are placeholders; the fitter writes the real ones. The `fixed` list is what the orchestrator reads: a template whose `scope.chip` differs from a measured contract in any `fixed` field is extrapolating (§4.2).

The illustrative contract stays in the repo for the demo run and the tests; the measured contracts are added beside it.

---

## 3. The real evaluator backend

`agentic/agents/evaluator/cuda.py`: a second backend after `roofline`, selected by `evaluator_backend="cuda:spark-6951"`.

### 3.1 From a solved design to a kernel

The tiling template's `program` is already a kernel skeleton: for each block, prefetch P tiles into S rotating shared-memory slots, then per k-phase load tile p+P, leading barrier, compute from slot p mod S, optional trailing barrier. `templates/smem_tiling/kernel.cu.j2` is that program as a CUDA kernel template with `Tm, Tn, Tk, S, P, beta` as compile-time constants, `cp.async` for the loads (B8 says whether TMA is also available), `__syncthreads` for barriers, and FP32 FMA for the compute (the template is FP32; a tensor-core variant is a later template, not a flag on this one). Grid rounding and guard follow `scope.structure`, so the T7 floor-grid design compiles to a kernel that really drops the tail, which is what the reference check must catch.

`codegen(T, x) -> str` renders the kernel; the rendered source is stored as an artifact (`kernel`, content-addressed) and linked from the measurement, so a measured number always points at the exact code that produced it.

### 3.2 Running it

The evaluator does not run CUDA locally. It calls a runner on the machine over SSH:

```
agentic-runner --kernel kernel.cu --m 1000 --n 1000 --k 1000 --reps 50 --json out.json
```

The runner (`calibration/runner/`, Python plus `nvcc`) compiles with `-arch=sm_121` (or `sm_120`), allocates A, B, C, fills A and B from a fixed seed, runs the kernel `reps` times after 5 warm-ups, records the median and the 95 % interval from `cudaEvent` timing, and **checks C against `cublasSgemm`** on the same inputs with a relative tolerance of 1e-4. The check result is part of the measurement: `reference_agrees: true|false`, with the first disagreeing index when false.

The evaluator service copies the kernel to the host, runs the command, reads the JSON, and returns a `measurement` artifact with `simulator: "cuda:spark-6951"`, `measured_cost` = median time in µs, `predicted_cost` = the solver's objective, `discrepancy`, the confidence interval, the clocks during the run (B11), the kernel artifact id, and `reference_agrees`.

### 3.3 What the orchestrator does with it

- `measured` is now a real number; the `optimism` record updates from it, and the first design that runs faster than predicted makes the record dirty for real, demoting every `skip_regime` that depended on a model ceiling (design doc §2.4). This is the first time that path can execute.
- `reference_agrees: false` is **not** a performance obligation. It means the certified design computed the wrong answer on real hardware, which can only happen if the verifier's model of the hardware was wrong (a race the two-warp event model did not capture, a `cp.async` ordering assumption). The orchestrator records a `diagnosis` with source `trusted_base` and stops the run: this is a bug in the verifier's hardware contract, the most important thing the real GPU can find.
- Utilizations for regime detection come from the measured time and the cost model's traffic and FLOP counts (bytes moved / measured time against `dram_bytes_per_s`, and likewise for compute). A proper counter-based measurement (`ncu`) is a later refinement; the division is honest as long as the counts come from the design, not the model's time.

---

## 4. Extrapolation and the regime rule

### 4.1 Two kinds of knob

| Knob | On a measured contract | Orchestrator treatment |
|---|---|---|
| Tiles, slots, prefetch, barrier, layouts, instruction schedule (software) | freely variable; every design runs on the card | measured; regime may be **proved** by a bound over the menu when the card confirms the dominant resource on every probed point |
| SM count, shared-memory capacity, bandwidth, tensor-unit shape (hardware) | fixed on this machine | any template whose scope differs in a `fixed` field is **extrapolating**: its regime strength is capped at `observed`, its measurements are the model's, and the narration says so |

### 4.2 Implementation

In `Orchestrator._iterate`, after the proposal is validated: if `T.scope.chip.kind == "measured"` and any field in `T.scope.chip` differs from the measured contract's value for a name in `fixed`, append `extrapolating` with the differing fields and set `evaluator_backend` for this iteration to `roofline` regardless of the run's backend. The story clause: "This template changes the chip (smem 96 → 128 KiB), which the Spark cannot run; costs are modeled, and the regime is only assumed."

### 4.3 Two machines, one suite

The same suite and the same runner produce `rtx5090-measured`. A template run on both contracts gives the first measured regime boundary: the orchestrator's `regime_bound` for the Spark says memory-bound on every feasible design, and for the 5090 says the regime varies. Both are recorded in the population as structure–regime evidence (paper §2.6), which is the regime map the design doc asked the curator to build.

---

## 5. Console

Small changes to round 4:

- **Schedule view, resource strip:** two bars per resource when a measured contract is in play, *modeled* and *measured*, with the discrepancy. The caption changes from "modeled, not measured" to "measured on spark-6951, 50 runs, ±1.2 %".
- **Schedule view, pipeline:** durations stay modeled per phase (there is no per-phase measurement), but the total block time is scaled to the measured kernel time, and the caption says "structure modeled; total measured".
- **Evaluator badge:** "measured 61.2 µs · predicted 53.1 (+15 %)".
- **Narration:** "Measured on the Spark: 61.2 µs, 15 % slower than predicted; the model is now known to be optimistic in this regime." If `reference_agrees` is false: "**Stopped:** the design computed a wrong answer on real hardware; the verifier's hardware assumptions need review."
- **Pool card:** the chip name on each card, so "ragged 1000³ GEMM on spark-6951" and "… on illustrative FP32 GPU" are distinguishable.
- **A calibration page** (`/calibration`): one table per host from `calibration/<host>/contract.yaml`, each row a quantity with its value, interval, method and date, and a link to the raw JSON. This is where Prof. Peh's students will look first.

---

## 6. Build order and acceptance tests

| Stage | Deliverable | Acceptance test |
|---|---|---|
| **F0** Pre-flight | `calibration/spark-6951/preflight.md` | All six items answered; toolchain compiles and runs a hello-world kernel on the Spark; capability recorded. |
| **F1** Suite, minimum | B1, B2, B4, B5, B6, B7 and the fitter; `calibration/spark-6951/contract.yaml` | Every value has an interval under 2 % or 50 reps; `dram_bytes_per_s` is within the datasheet envelope (0.6–1.0 × 273 GB/s); `barrier_s[256]` is positive and under 1 µs; the fitter's output validates against the extended chip schema. |
| **F2** Suite, full | B3, B8–B12 | `tma_supported` recorded; `unified_memory_penalty` measured; B11 reports clock variance. The README for each benchmark cites its method. |
| **F3** Runner + codegen | `templates/smem_tiling/kernel.cu.j2`, `codegen`, `calibration/runner/` | For T5's design on the ragged GEMM, the generated kernel compiles, runs, and `reference_agrees` is true. For T7's floor-grid design, `reference_agrees` is **false** with a disagreeing index in the tail. For an `unsound` precondition design (S = 2, P = 1, β = 0), the kernel either disagrees or agrees by luck; the test asserts only that the runner reports a result, and records it. |
| **F4** Evaluator backend | `evaluator/cuda.py`; backend selection; `extrapolating` event; story clauses | A run of the scripted formulator on `spark-6951-gb10` produces real measurements for the five accepted templates; the ledger holds a kernel artifact per measurement; the optimism record's history has five real entries; a template that changes `smem_bytes_per_block` is marked `extrapolating` and measured by the roofline backend. |
| **F5** Console | §5 changes | The schedule view for a Spark-measured template shows both bars and the measured caption; the narration carries the measured clause; `/calibration` lists the Spark's contract with links to raw JSON. |
| **F6** Second machine | same suite on the 5090 box; `rtx5090-measured.yaml` | The two contracts differ by more than 4× in `dram_bytes_per_s`; running T5 on both yields different dominant resources, recorded as two `regime_bound` events with different `regime` values. |

**Review gate:** Li reads `calibration/spark-6951/contract.yaml` and the T5 measurement and can say whether the 15 % (or whatever it is) gap is bandwidth, barrier cost or something the model does not have. If the answer is "something the model does not have", that is the next cost-model term, and it goes into the paper's §6.6 as a measured limitation replacing a modeled one.

---

## 7. What this changes in the paper

- §6.1 gains a measured chip beside the illustrative one, with the calibration method and the provenance block.
- §6.2's "certified in 25 ms" gains "and measured on a GB10 at x µs against a predicted y".
- §6.6, "When the formulation is wrong," gains the first measured model-vs-hardware gap, replacing the modeled 58–63 % from the KV study as the headline example.
- §8's third next step, "carry one template through to execution on a real GPU," is done; the open item becomes the tensor-core template.

---

## 8. Out of scope

- A tensor-core (BF16/FP8) tiling template: a new template with its own program and obligations, not a flag on the FP32 one. The suite measures B3 now so the template can be written later.
- Counter-based utilization (`ncu`): later refinement of §3.3.
- Multi-GPU across the eight 5090s: a system-architecture-level template (NoC/NVLink), which is Prof. Peh's territory and round 5 of the console.
- Power: the Spark reports package power through `nvidia-smi`; record it in B11 but do not model it yet.

---

## 9. Questions for Li

1. Can clocks be locked on the Spark for the suite (`nvidia-smi -lgc`)? If not, every measurement carries the B11 variance and the acceptance intervals in F1 widen to 4 %.
2. Is the 5090 box shared? The suite needs one idle GPU for ten minutes; the runner needs one GPU per evaluation call.
3. For the reference check, cuBLAS SGEMM at 1e-4 relative tolerance, or an exact-order CPU reference for small shapes? cuBLAS is faster; the CPU reference is closer to the verifier's token replay. My recommendation: cuBLAS for timing runs, CPU reference once per template family in F3.
