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

References: [1] Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza, "Dissecting the NVIDIA Volta GPU
Architecture via Microbenchmarking", arXiv:1804.06826 (and the Turing follow-up, arXiv:1903.07486).
[2] W. Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", IPDPS 2024.
[3] M. Khairy et al., "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling",
ISCA 2020 (the tuner). [4] X. Mei, X. Chu, "Dissecting GPU Memory Hierarchy through Microbenchmarking",
IEEE TPDS 2017.
