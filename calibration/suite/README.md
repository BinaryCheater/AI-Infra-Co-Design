# Calibration suite

Microbenchmarks that measure what the hardware contract needs and fit them into a `measured`
contract with provenance (`notes/hardware_calibration_design.md` §2). Each benchmark is one CUDA
program under `bench/` that prints JSON lines `{quantity, unit, samples, detail}`; `driver.py`
builds and runs them on the GPU host, repeating until the 95 % interval of every quantity is under
the target (4 % where clocks cannot be locked) or 50 repetitions, and writes `raw/*.json`;
`fit.py` turns the raw records into `calibration/<host>/contract.yaml`, which validates against
`agentic/schema/chip.json`.

```bash
calibration/suite/remote.sh <ssh-alias>                      # minimum set B1 B2 B4 B5 B6 B7 -> calibration/<host>/raw
calibration/suite/remote.sh <ssh-alias> --full               # adds B3, B8-B12
python calibration/suite/fit.py --raw calibration/<host>/raw --out calibration/<host>/contract.yaml --datasheet-bw 273e9
```

The host is an SSH alias from the local SSH config; no address or key is stored in the repository.

| # | Benchmark | Measures | Method |
|---|---|---|---|
| B1 | `b1_props` | declared properties | `cudaGetDeviceProperties`, `cudaDeviceGetAttribute` |
| B2 | `b2_fma` | FP32 FMA throughput per SM | dependent FMA chains with ILP 8, 8 blocks/SM; Jia et al. 2018 [1] |
| B4 | `b4_bandwidth` | DRAM and L2 bandwidth (read/write/copy) vs working set; effective L2 capacity from the knee | STREAM-style grid-stride kernels; Accel-Sim tuner [3] |
| B5 | `b5_latency` | L1 / L2 / DRAM latency | fine-grained pointer chase over a random cyclic permutation of lines; Mei & Chu 2017 [4], Jia et al. 2018 [1] |
| B6 | `b6_smem` | shared-memory bandwidth per SM and bank-conflict penalty by stride | Jia et al. 2018 [1]; Luo et al. 2024 [2] |
| B7 | `b7_barrier` | `__syncthreads` cost by block size | barrier loop, one block per SM |
| B3 | `b3_tensor` | tensor-core throughput per precision: per-SM peak (WMMA `mma.sync` microkernel) and chip-level cuBLAS (`cublasGemmEx`, 8192²) for FP16, BF16, TF32 and FP32 | Jia et al. 2018 [1] (mma microkernel); Luo et al. 2024 [2] (per-precision sweep). FP8 needs cublasLt and is recorded as not measured |
| B8 | `b8_async_copy` | `cp.async` issue-to-completion latency (one tile in flight) and pipelined throughput (two in flight) for 4–48 KB tiles; `cp.async.bulk` + mbarrier (TMA) latency and whether it copies correctly | Luo et al. 2024 [2] (Hopper async-copy and TMA dissection) |
| B9 | `b9_occupancy` | blocks per SM vs dynamic shared memory (occupancy API and a measured resident-block count) and vs registers per thread (`__launch_bounds__` variants) | occupancy API cross-checked by measurement, as in the Accel-Sim tuner [3] |
| B10 | `b10_launch` | kernel launch overhead, device (events) and host (wall) per launch | 1000 empty launches, Jia et al. 2018 [1] |
| B11 | `b11_clocks` | clock, power and temperature under a 60 s FMA load; throughput drift | `nvidia-smi` sampled once per second (no clock lock available on the Spark) |
| B12 | `b12_unified` | GPU read bandwidth over `cudaMalloc`, `cudaMallocManaged` (GPU-first, CPU-first first pass and steady), `cudaMallocHost`; penalty relative to device memory | Spark only in intent; runs anywhere managed memory exists |
| B13 | `b13_l2_structure` | L2 fill granularity (bytes per request vs stride), set conflicts (n lines at power-of-two strides, `ld.cg` chase), slice-select bits (two-line chase per flipped address bit) and replacement (hit rate vs footprint, sequential vs random cyclic) | pointer-chase structure probes, Mei & Chu 2017 [4]; B13_plus.md |
| B14 | `b14_l1_carveout` | L1 capacity and latency per shared-memory request, line probe, whether `cp.async.cg` / `.ca` / TMA allocate in L1, the L1 + smem pool | chase at 128 B stride per carve-out; B13_plus.md |
| B15 | `b15_copy_queue` | `cp.async` groups in flight (issue cost vs pending groups), issue cycles per 16 B, completion distribution by tile and residency, cp.async vs TMA vs mixed aggregate throughput (shared path?), mbarrier round trip, 2-D TMA latency | B13_plus.md |
| B16 | `b16_block_scheduler` | block-to-SM assignment order (`%smid` + `%globaltimer` at entry), determinism over repeats, dispatch rate, refill latency, wave staircase | B13_plus.md |
| B17 | `b17_latency_hiding` | bandwidth vs resident warps (DRAM, L2) and warps-to-hide, per-warp outstanding loads (dependent iterations of U independent loads), issue rate vs warps, dependent-chain latencies (FMA, IMAD, LDS, L1 hit) | B13_plus.md; Jia et al. 2018 [1] |
| B18 | `b18_dram` | channel-select bits (random 128 B requests with one address bit forced), locality by run length, read/write turnaround, bandwidth under CPU load, 60 s drift | B13_plus.md |
| B19 | `b19_registers_local` | spill curve (`-maxrregcount` variants with ptxas spill bytes), local-memory latency and bandwidth, allocation granularity (from B9) | B13_plus.md |
| B20 | `b20_tensor_detail` | FP8 through cublasLt, per-shape `mma.sync` rates (fp16/bf16/tf32/int8/fp8), dependent `mma.sync` latency, `ldmatrix` feed swizzled vs not | B13_plus.md |
| B21 | `b21_compute_block` | per-opcode throughput per SM (ILP 8), `ld.shared` by width, the tiling template's inner loop vs the FFMA bound with its SASS mix (`cuobjdump`) | B13_plus.md |
| B22 | `b22_clock_governor` | clock warm-up from idle, idle clock, sustained clock by load type with power/temperature and cap flags, clock after idle gaps | `nvidia-smi` once per second; B13_plus.md |
| B23 | `b23_unified_mechanics` | first-touch cost by page size, prefetch, host-resident access, managed vs device with the CPU idle, allocation limits | CUDA 13 `cudaMemLocation` API; B13_plus.md |
| B24 | `b24_atomics_reductions` | atomic throughput by scope and pattern, `atom` vs `red`, same-address contention curve, latency | B13_plus.md |
| B25 | `b25_multi_kernel` | kernel-to-kernel gap, CUDA-graph gap, two-stream concurrency, L2 persistence across launches and the persisting-window API | B13_plus.md |
| B26 | `b26_functional_probes` | `cp.async` chunk order/window/torn chunks, `wait_group` semantics, fence visibility, TMA bytes before mbarrier completion, FMA contraction and FTZ defaults, tensor accumulate rounding, bank map, warp divergence interleaving | B13_plus.md |

References: [1] Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza, "Dissecting the NVIDIA Volta GPU
Architecture via Microbenchmarking", arXiv:1804.06826 (and the Turing follow-up, arXiv:1903.07486).
[2] W. Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", IPDPS 2024.
[3] M. Khairy et al., "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling",
ISCA 2020 (the tuner). [4] X. Mei, X. Chu, "Dissecting GPU Memory Hierarchy through Microbenchmarking",
IEEE TPDS 2017.


The structure suite B13–B26 (`calibration/suite/B13_plus.md`) runs with `driver.py --micro`; its fitters are in `fit_micro.py`, one per benchmark, and its fields land in the same contract under the `M1–M12` groups of the `/calibration` page, with B26's answers under `functional:`. Benchmarks that report a structure rather than a rate (B13, B16, B18, B22, B23, B26) run once per invocation (`repeat_policy: once` in their records).

After P0 (profiler counters unlocked), `driver.py --crosscheck` runs `ncu_crosscheck.py`: `bench/bx_counters.cu` under `ncu`, one record `counter_<check>[a=,b=]` per configuration, fitted into the contract's `counter_crosscheck:` block (L2 fill bytes per request, L2 hit rate vs footprint, issue rate vs warps, inner-loop stall reasons, shared-memory wavefronts, L1 hit rate by footprint and carve-out). A counter supersedes a timing inference of the same quantity.

Part 3 (`B27_plus.md`): `driver.py --micro3` runs B27 (`b27_dependency_ilp`: dependency chains vs ILP vs TLP with the FFMA/LDS latency references and the streaming-load grid) and B28 (`b28_lsu_port_contention`: shared-memory and copy traffic across 1-6 resident blocks); `b29_ground_truth.py` builds the generated kernels at the G4 corners, stripped and real, and measures them at 1-6 resident blocks under `ncu` (records `inner_loop_ground_truth[...]`, fitted under `counter_crosscheck`). `agentic/sim/check_warp.py` checks the warp model against B27 and B29 without changing a parameter.

Part 4 (`B30_plus.md`): `b30_mio_queue` measures the queue between the warp scheduler and the load/store unit (dispatch cycles per load per sub-partition, drain in wavefronts per cycle by width and stride with the `ncu` wavefront counts, cp.async and ldmatrix sharing). Its numbers are the warp model's only LSU parameters.
