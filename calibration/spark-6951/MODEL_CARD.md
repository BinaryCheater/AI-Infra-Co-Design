# GB10 model card, `gb10-model-v1`

The mechanistic simulator of the NVIDIA GB10 (DGX Spark, host spark-6951, `sm_121`, CUDA 13.0) as frozen at tag `gb10-model-v1`. Every parameter is a measurement in `contract.yaml` with its benchmark named beside it; nothing is fitted to a kernel. Structural choices and the benchmark behind each are in `agentic/sim/config/spark-6951-gb10.yaml`. The chip book is `notes/gb10_microarchitecture.md`; the open items are `notes/sim_gaps.md`.

## What it covers

One line per mechanism: the benchmarks that measured it and the error of the model as a prediction on the composite kernel that depends on it most (this is `coverage.md`, sorted worst first).

| mechanism | measured by | prediction error |
|---|---|---|
| cp.async completion under load | B8, B15, B28, B30.4, C1, C2 | copies alone within 5 % (C1); a copy beside a loop does not overlap it on the card: C1 -12 % / -15 %, C2 -14 % / -17 % at 1 / 2 blocks; GEMM copy-bound designs 2-3x (gaps 23-24, 29-30) |
| SFU and converts (ex2, rcp, rsqrt, lg2, tanh, F2F, F2FP) | B21, B38, C3 | one MUFU per 8 cycles per sub-partition (counter); ex2 + FFMA mixes -2 % to +18 %; softmax rows +67 % at 1 block to +11 % at 6 (gap 32) |
| LSU queue (scheduler to load/store unit) | B27, B28, B30 | B29 loops 16 of 19 within 10 %, worst +27 %; B28 load mix -24 % at 1 block, +56 % at 6 (gaps 21-22) |
| shuffles (in the LSU queue, one wavefront each) | B39, C3 | butterfly reduction -7 %; beside loads -31 % (gap 33) |
| TMA (bulk copies, 2-D maps, mbarrier) | B8, B15, B35, B36, B37, C2 | copies alone +20 % / +17 % / +3 %; the kernel as above; multicast 500x slower than unicast (gap 36) |
| tensor pipe, mma hold, ldmatrix | B20, B31-B34, C1 | B31 grid within 8 %, B32 loop within 6 %, B33 under load within 6 %; C1 compute-only +15 % / +2 %; an FFMA warp beside an mma warp -39 % (gap 27) |
| issue (four schedulers, in-order warps) | B17, B21, B27, B29 | B29 T1 / T5 within 6 %; B27 streaming grid 13 of 20 within 10 % |
| scoreboard latencies (FFMA 4, LDS 27, mma 30, ldmatrix 47, ex2 18, shfl 27) | B17, B27, B31, B32, B38, B39 | dependent chains within 2 cycles; one-warp streaming points 2x (compiler pipelining, gap 17) |
| L2 (24 MiB, hashed sets, LRU-like, 1.0 TB/s) | B4, B5, B13, counters | B4 plateaus within 5 %, the 16 MiB streaming knee predicted |
| shared-memory banks (32 x 4 B, wavefront rule) | B6, B26.6, B30.3 | wavefronts from addresses: 5 of 6 B29 kernels within 1 % |
| DRAM (235 GB/s read, hashed channels) | B4, B18 | B4 large plateau within 5 % |
| L1 (64 KB, bypassed by copies) | B5, B14, counters | pointer chase within 5 % |
| registers (allocation, spills, residency) | B9, B19, B34, B40 | residency by registers 9 of 10 points equal the occupancy API; spill cliffs measured, no composite depends on one |

The thirty-design GEMM set (`python -m agentic.sim.validate`): median 8.2 %, worst 41 %, 17 of 30 within 10 %; the five copy-bound and barrier-bound designs are the first row's miss.

## What it does not cover

By gap number in `notes/sim_gaps.md`:

- **23, 24, 29, 30.** A copy issued beside a loop does not overlap it; the model overlaps them. The first thing to measure next.
- **27, 35.** Issue cycles of an `mma` or `MUFU` that another warp's FFMAs can take; the model holds the warp.
- **17, 32.** The compiler's reordering across independent loop iterations at low occupancy.
- **33.** Shuffles beside loads slow more than their wavefront accounts for.
- **26.** The `ldmatrix.x4` latency has two readings 30 cycles apart; the chain's is used.
- **21, 22.** The load mix at six blocks and the 32x128x32 loop's instruction count.
- **25.** The `cp.async` fill path is invisible to the shared-memory wavefront counters.
- **36, 37.** Multicast and DSMEM: measured, not modeled, not used.
- **38.** One residency point reads one block below the occupancy API.
- Register-file bank conflicts, L2 set hash and slices, DRAM channel count: not measurable from user space so far (book, chapter 9).

Not covered at all: anything outside a 256-thread tiled GEMM or a one-warp-per-row softmax: no attention template yet, no persistent kernels, no clusters, no FP64, no unified-memory traffic inside a kernel.

## How to run it on a design

```bash
# a tiled-GEMM design on a workload, through the full simulator (events, memory hierarchy, warp model)
PYTHONPATH=. python -c "
from agentic.sim.config import load; from agentic.sim.run import SimConfig; from agentic.sim.memory import Hierarchy
from agentic.sim.warp import WarpModel; from agentic.sim.tiling import simulate_design
from agentic.templates.smem_tiling import TilingDesign, Workload
cfg = load('spark-6951-gb10'); C, S, F = cfg['contract'], cfg['structural'], cfg['fixed']
r = simulate_design(TilingDesign(128, 128, 32, 3, 1, True), Workload('square_4096', 4096, 4096, 4096), SimConfig(C, S, F, 'spark-6951-gb10'), Hierarchy(C, S, F), issue=WarpModel(C, S))
print(r['time_us'], r['dram_bytes'], r['critical_counts'])"

# the thirty-design GEMM set against the Spark measurements
PYTHONPATH=. python -m agentic.sim.validate
# the tensor path, the TMA engine, the SFU / shuffle classes against their benchmarks and the three composites
PYTHONPATH=. python -m agentic.sim.check_tensor
PYTHONPATH=. python -m agentic.sim.check_round2
# re-measure the chip (needs the Spark over the ssh alias dgx-spark) and refit the contract
calibration/suite/remote.sh dgx-spark --arch sm_121 --ci 0.04 --max-reps 50 --round2 --crosscheck
PYTHONPATH=. python calibration/suite/fit.py --raw calibration/spark-6951/raw --out calibration/spark-6951/contract.yaml --name spark-6951-gb10 --datasheet-bw 273e9
```

A new kernel shape is a program for `agentic/sim/tensor.py`'s runner (a list of `Instr` per warp, classes mma / ldsm / ffma / alu / lds / copy / wait / bar / sfu / shfl / cvt / gload / gstore) or events for `agentic/sim/run.py`; the parameters it will use are the contract's, and the coverage row each rests on is the label to carry with the number.
