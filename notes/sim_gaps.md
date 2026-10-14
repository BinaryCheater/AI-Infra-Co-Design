# Simulator gaps

Designs the mechanistic model could not match **without changing a measured parameter**, logged as
missing mechanisms per `notes/mechanistic_simulator_design.md` §4. Measured parameters come from
`calibration/spark-6951/contract.yaml` and are never fitted; structural choices are in
`agentic/sim/config/spark-6951-gb10.yaml` with the benchmark that justified each.

| # | Design / benchmark | Simulated vs measured | What is missing | Status |
|---|---|---|---|---|
| 1 | 4096³ GEMM, 128 × 256 × 16 tiles, S = 2, P = 1, β = 1 (spark-001, iteration 1): 9395.7 µs measured | G2 (L2 + DRAM only): 5057 µs (−46 %), critical path `barrier`/compute in every phase | **M3, shared-memory reads during compute.** Each k-step of the generated kernel issues 8 + 16 `ld.shared` per thread (393 KB per phase per block); at the measured 154 GB/s per SM that is 2.5 µs per phase against 1.8 µs of FMA. G2 has no shared-memory mechanism, so the phase is FMA-bound in the model and smem-bound on the card. | G3 adds M3 (banks, ports, per-SM bandwidth); residual after G3 recorded below |
