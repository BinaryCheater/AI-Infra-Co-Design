# GB10 model card, `gb10-model-v2`

The mechanistic simulator of the NVIDIA GB10 (DGX Spark, host spark-6951, `sm_121`, CUDA 13.0) as released at tag `gb10-model-v2` (it supersedes `gb10-model-v1`). Every parameter is a measurement in `contract.yaml` with its benchmark named beside it; nothing is fitted to a kernel. Structural choices and the benchmark behind each are in `agentic/sim/config/spark-6951-gb10.yaml`. **The warp model's default input is the kernel's SASS stream** (`input: stream`); the per-step counts stay behind the switch (`input: counts`). It counts cycles; there is no power model and no clock model. The landing page is `README_GB10.md` at the repository root; the chip book is `notes/gb10_microarchitecture.md`; the open items are `notes/sim_gaps.md`; the opcodes are `opcodes.md`.

## Accuracy

| check | result |
|---|---|
| thirty GEMM designs measured on the Spark (stream input) | median error 8.6 %, 20 of 30 within 10 %, 27 within 20 %, worst 34 % |
| the same with the counts input | median 8.1 %, 17 within 10 %, worst 35 % |
| softmax rows, 1 / 2 / 4 / 6 blocks per SM | -2 / -4 / -4 / +7 % |
| BF16 tensor tile (C1), 1 / 2 blocks | +9 / +6 % |
| T5 tile with TMA (C2), 1 / 2 blocks | -21 / -18 % (the open row below) |

Source: `input_comparison.json`, produced by `python -m agentic.sim.compare_input --jobs 16`.

## What it covers

One line per mechanism: the benchmarks that measured it and the error of the model as a prediction on the composite kernel that depends on it most (this is `coverage.md`, sorted worst first). **The open row is first.**

| mechanism | measured by | prediction error |
|---|---|---|
| **cp.async and TMA copy completion (open)** | B8, B15, B28, B30.4, B35, B36, C1, C2 | the copy's instructions are explained (C1: copy cost on top of compute 453 measured, 437 stream, gap 29); the copy's completion is not: B28's 4 KB tile is about 200 cycles per SM slower than the model, -37 % to -40 % (gap 40); the TMA tile kernel -21 / -18 % (gap 30); GEMM copy-bound designs (gaps 23-24) |
| LSU queue (scheduler to load/store unit) | B27, B28, B30 | B29 loops 16 of 19 within 10 %; B28 load mix +32 % to +79 % at 3 to 6 blocks (gap 22) |
| shuffles (LSU instructions, one wavefront each) | B39, C3 | butterfly reduction -7 %; beside loads -31 % (gap 33) |
| tensor pipe, mma hold, ldmatrix | B20, B31-B34, C1 | C1 +9 / +6 %; B31 grid within 8 %, B32 loop within 6 %, B33 within 6 %; FFMA warp beside mma warp -39 % (gap 27); three blocks with a second wave +27 % (gap 41) |
| TMA engine, 2-D maps, mbarrier | B35, B36, B37, C2 | C2 compute-only -11 / -10 %; the kernel as in the first row; multicast 500x slower than unicast (gap 36) |
| SFU and converts (ex2, rcp, rsqrt, lg2, tanh, F2F, F2FP) | B21, B38, C3 | one MUFU per 8 cycles per sub-partition (counter, gap 34); softmax rows within 7 %; ex2 + FFMA mixes -2 % to +18 % (gap 35) |
| issue (four schedulers, in-order warps, SASS stream) | B17, B21, B27, B29, the SASS of each kernel | thirty designs as above; B29 T1 / T5 within 6 % |
| scoreboard latencies (FFMA 4, LDS 27, mma 30, ldmatrix 47, ex2 17, shfl 27) | B17, B27, B31, B32, B38, B39 | dependent chains within 2 cycles |
| L2 (24 MiB, hashed sets, LRU-like, 1.0 TB/s) | B4, B5, B13, counters | B4 plateaus within 5 %, the 16 MiB streaming knee predicted |
| shared-memory banks (32 x 4 B, wavefront rule) | B6, B26.6, B30.3 | wavefronts from addresses: 5 of 6 B29 kernels within 1 % |
| DRAM (235 GB/s read, hashed channels) | B4, B18 | B4 large plateau within 5 % |
| L1 (64 KB, bypassed by copies) | B5, B14, counters | pointer chase within 5 % |
| registers (allocation, spills, residency) | B9, B19, B34, B40 | residency by registers 9 of 10 points equal the occupancy API; spill cliffs measured |

## What it does not cover

By gap number in `notes/sim_gaps.md`, the open one first:

- **40, 30, 23, 24. Copy completion.** About 200 cycles per 4 KB `cp.async` tile per SM in B28, and the TMA tile 18 to 21 % slow, sit on the path that lands the data. Instruction cost is ruled out (gap 40). No counter on this chip shows the fill path (gap 25).
- **27, 35.** Issue cycles of an `mma` or `MUFU` that another warp's FFMAs can take; the model holds the warp.
- **41.** With more blocks than fit, the measured median block is faster than the two-block time; the benchmarks keep only the median, so the per-block schedule is not known.
- **33.** Shuffles beside loads slow more than their wavefront accounts for.
- **22.** The load mix at three to six blocks.
- **26.** The `ldmatrix.x4` latency has two readings 30 cycles apart; the chain's is used.
- **36, 37.** Multicast and DSMEM: measured, not modeled, not used.
- **38.** One residency point reads one block below the occupancy API.
- Register-file bank conflicts, L2 set hash and slices, DRAM channel count: not measurable from user space so far (book, Chapter 9).

Not covered at all: power, clock behaviour under load, FP64, clusters, unified-memory traffic inside a kernel, and kernels outside a tiled GEMM, a tensor tile or softmax rows. A design with no SASS dump on disk runs on the counts input, and its result says so.

## How to run it on a design

```bash
# a tiled-GEMM design on a workload (one of the thirty measured designs, so its SASS dump is on disk), through the full simulator
PYTHONPATH=. python -c "
from agentic.sim.config import load; from agentic.sim.run import SimConfig; from agentic.sim.memory import Hierarchy
from agentic.sim.warp import WarpModel; from agentic.sim.tiling import simulate_design
from agentic.templates.smem_tiling import TilingDesign, Workload
from agentic.chips import load_chip, scope_chip
cfg = load('spark-6951-gb10'); C, S, F = cfg['contract'], cfg['structural'], cfg['fixed']
r = simulate_design(TilingDesign(128, 128, 32, 1, 0, True), Workload('ragged_1000x1000x1000', 1000, 1000, 1000), SimConfig(C, S, F, 'spark-6951-gb10'),
                    Hierarchy(C, S, F), issue=WarpModel(C, S), chip=scope_chip(load_chip('spark-6951-gb10')))
print(r['time_us'], r['input'], r['critical_counts'])"

# thirty designs and C1-C3 through both inputs, in parallel
PYTHONPATH=. python -m agentic.sim.compare_input --jobs 16
# the tensor path, the TMA engine, the SFU / shuffle classes against their benchmarks and the composites
PYTHONPATH=. python -m agentic.sim.check_tensor
PYTHONPATH=. python -m agentic.sim.check_round2
PYTHONPATH=. python -m agentic.sim.check_b28_stream
# the opcode table from the pipe table and the contract
PYTHONPATH=. python -m agentic.sim.isa
# re-measure the chip (needs the Spark over the ssh alias dgx-spark) and refit the contract
calibration/suite/remote.sh dgx-spark --arch sm_121 --ci 0.04 --max-reps 50 --round2 --crosscheck
PYTHONPATH=. python calibration/suite/fit.py --raw calibration/spark-6951/raw --out calibration/spark-6951/contract.yaml --name spark-6951-gb10 --datasheet-bw 273e9
```

A new kernel shape is a SASS stream for `WarpModel.run_stream` (read with `agentic/sim/sass.py`) or events for `agentic/sim/run.py`; the parameters it uses are the contract's, and the coverage row each rests on is the label to carry with the number.
