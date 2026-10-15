# The GB10 model: start here

This is a measured model of one GPU: the NVIDIA GB10 in the DGX Spark, host spark-6951, `sm_121`. It has four parts. A book says what the chip does. Two specs say how the simulator is built. The simulator runs a kernel on the model. The measurements are what every number comes from. Release tag: `gb10-model-v2`.

## Scope, in one paragraph

The model covers the mechanisms a GEMM or an attention kernel touches: warp issue on four schedulers, the scoreboard, the queue into the load/store unit, shared-memory banks, `cp.async` and TMA copies, the tensor pipe, the SFU, shuffles, the register budget, L1, L2 and DRAM. Every parameter was measured on spark-6951; nothing is fitted to a kernel. It counts cycles. It has no power model and no clock model; cycles turn into seconds at the measured sustained clock. On thirty GEMM designs measured on the Spark, the median error is 8.6 %, 20 of 30 are within 10 %, and the worst is 34 %. Softmax rows are within 7 % at 1 to 6 blocks per SM.

**One mechanism is open: copy completion.** In B28, a 4 KB `cp.async` tile takes about 200 cycles per SM longer than the model says (gap 40). The TMA tile kernel is 18 to 21 % slow in the model (gap 30). This is not instruction cost: the stream input already charges the copy code's instructions, and that explained most of the gap on the tensor tile (gap 29). It is something on the path that lands the data, and no counter on this chip shows it yet. Any design that waits on its copies inherits this error.

The full table of what is measured and how well it predicts: [coverage.md](calibration/spark-6951/coverage.md). Everything known to be wrong, with the counter that shows why: [sim_gaps.md](notes/sim_gaps.md).

## Where to start

- **To understand the chip:** read the book, [notes/gb10_microarchitecture.md](notes/gb10_microarchitecture.md). Part I (Chapters 1–9) is the microarchitecture: schedulers, pipes, shared memory and the load/store path, L1, L2, DRAM. Part II (Chapters 10–15) is the instruction set: how to read SASS, the instructions by pipe, three kernels read line by line, what the compiler does. Chapter 9 and Chapter 15 list what is not known.
- **To understand the simulator:** read the two specs. [notes/mechanistic_simulator_design.md](notes/mechanistic_simulator_design.md) is the chip model: one event semantics read two ways, the mechanisms M1–M12, how each is checked against a benchmark. [notes/isa_simulator_design.md](notes/isa_simulator_design.md) is the instruction front end: SASS in, dependencies from register names, issue through the measured pipes. [notes/instruction_stream.md](notes/instruction_stream.md) is the one-page version of what was built.
- **To check a number:** go to [calibration/spark-6951/](calibration/spark-6951/). `contract.yaml` holds every measured parameter with the benchmark that produced it. `raw/` holds the raw records, one file per quantity. [MODEL_CARD.md](calibration/spark-6951/MODEL_CARD.md) is the one-page summary. [opcodes.md](calibration/spark-6951/opcodes.md) lists every SASS opcode the simulator knows, with its pipe, cost, latency and source.
- **To rerun a benchmark:** [calibration/suite/](calibration/suite/) holds B1–B40 and the composites C1–C3 (`bench/*.cu`), the driver, the counter cross-checks and the fitters. It needs the Spark.

## What to trust

- **A number in the book with a benchmark in brackets** is a measurement on this host. Trust it for this chip, this driver (580.159.03) and CUDA 13.0.
- **The contract** is measured, never fitted. Where a timing reading and a hardware counter disagreed, the counter won, and the timing reading is kept beside it.
- **The simulator** is as good as the coverage table says, row by row. Trust its regime (what a kernel waits on) and its ranking of designs more than its absolute time when a design waits on copies.
- **Not covered:** power, clock behaviour under load, other chips, FP64, clusters, multicast and distributed shared memory (measured, not modeled), unified-memory traffic inside a kernel, and kernels outside a tiled GEMM, a tensor tile or softmax rows.

## Running a design

The simulator is in [agentic/sim/](agentic/sim/). A tiled-GEMM design on a workload (one of the thirty measured designs, so its SASS dump is on disk; it measured 217.5 µs):

```bash
PYTHONPATH=. python -c "
from agentic.sim.config import load; from agentic.sim.run import SimConfig; from agentic.sim.memory import Hierarchy
from agentic.sim.warp import WarpModel; from agentic.sim.tiling import simulate_design
from agentic.templates.smem_tiling import TilingDesign, Workload
from agentic.chips import load_chip, scope_chip
cfg = load('spark-6951-gb10'); C, S, F = cfg['contract'], cfg['structural'], cfg['fixed']
r = simulate_design(TilingDesign(128, 128, 32, 1, 0, True), Workload('ragged_1000x1000x1000', 1000, 1000, 1000), SimConfig(C, S, F, 'spark-6951-gb10'),
                    Hierarchy(C, S, F), issue=WarpModel(C, S), chip=scope_chip(load_chip('spark-6951-gb10')))
print(r['time_us'], 'us;', 'input:', r['input'], '; what each phase waited on:', r['critical_counts'])"
```

**The input switch.** `input: counts | stream` in [agentic/sim/config/spark-6951-gb10.yaml](agentic/sim/config/spark-6951-gb10.yaml). `stream` is the default: the warp model runs the kernel's real SASS, read from `calibration/spark-6951/sass/<key>.sass`. `counts` builds each loop from per-step instruction counts in a hand-written order. A design with no SASS dump on disk runs on `counts`, and its result says so in `input`. To give a new design the stream, compile it on the GPU host and save `cuobjdump -sass` under its key.

The checks, each against measurements it was not built from:

```bash
PYTHONPATH=. python -m agentic.sim.compare_input --jobs 16   # thirty designs and C1-C3, counts and stream side by side
PYTHONPATH=. python -m agentic.sim.check_tensor               # the tensor path against B31-B34 and C1
PYTHONPATH=. python -m agentic.sim.check_round2               # TMA, SFU, shuffles against B35-B39, C2 and C3
PYTHONPATH=. python -m agentic.sim.check_b28_stream           # B28's copy runs through the stream
PYTHONPATH=. python -m pytest -q tests
```

The latest results of the first one are in [input_comparison.json](calibration/spark-6951/input_comparison.json). The optimizer can use the simulator as its evaluator: `--evaluator simulator` on a measured chip, and every number it returns names the coverage row it rests on.
