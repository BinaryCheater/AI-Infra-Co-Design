# agentic-runner

Runs on the GPU host. Takes a generated kernel (`agentic.templates.smem_tiling.codegen`), builds it
with `harness.cu` and cuBLAS, times it with `cudaEvent` (5 warm-ups, then `--reps` runs, median and
95 % interval), and checks the result against `cublasSgemm` on the same seeded inputs at a relative
tolerance (default 1e-4). `reference_agrees: false` comes with the first disagreeing index.

```bash
agentic_runner.py --kernel kernel.cu --m 1000 --n 1000 --k 1000 --reps 50 --json out.json
```

The evaluator backend (`agentic/agents/evaluator/cuda.py`) copies the kernel to the host, runs this
command over SSH and reads the JSON. Nothing here depends on the repository's Python package.
