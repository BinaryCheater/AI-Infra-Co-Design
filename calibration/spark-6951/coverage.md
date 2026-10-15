# Coverage: mechanisms, the benchmarks that measured them, how the model does as a prediction

One row per mechanism. Error = the composite check of the model as a prediction (nothing fitted) on the kernel that exercises the mechanism most directly; the thirty-design GEMM gate is listed where it is the only end-to-end check. Sorted worst first; the top row is the next round's target. Updated as each round-2 group completes.

| mechanism | benchmarks | composite-check error (prediction) | templates that depend on it |
|---|---|---|---|
| cp.async (copy path under residency) | B8, B15, B28, B30.4 | B28 copy-only mix 2-3x too fast at every residency; B15 p50 within 15 % on 2 of 8 (gaps 23-25) | smem_tiling (every design with P = 0 or many resident blocks) |
| LSU queue (scheduler to load/store unit) | B27, B28, B30 | B29 stripped loops: 16/19 within 10 %, median 5 %, worst +27 % (gap 21); B28 load mix -24 % at 1 block, +56 % at 6 (gap 22) | smem_tiling, tensor (ldmatrix), attention (shuffles if in the queue) |
| tensor (mma pipe per sub-partition, latency, hold, ldmatrix path) | B20, B31, B32, B33, B34, C1 | C1 128x128x32 bf16 tile: -12 % at 1 block, -15 % at 2 (gap 29: copies and compute do not overlap on the card); compute-only +15 % / +2 % (gap 28); copy-only +5 % / +1 %; B31 grid within 8 %, B32 pipelined within 6 %, B33 under load within 6 %, B34 split -39 % (gap 27) | tensor, attention |
| issue (four schedulers, in-order warps) | B17, B21, B27, B29 | B29 T1/T5 within 6 %; B27 streaming grid 13/20 within 10 % (gap 17) | all |
| scoreboard (latencies FFMA 4, LDS 27) | B17, B27 | B27 dependent chains: latency points within 2 cycles; one-warp a = 1, 2 streaming points 2x (compiler pipelining, gap 17) | all |
| L2 (24 MiB, hashed sets, LRU-like) | B4, B5, B13, counters | B4 plateaus within 5 %, knee at 16 MiB predicted (G2 test); 4096³ headline within 10 % only with the warp model | smem_tiling |
| banks (32 x 4 B, wavefront rule) | B6, B26.6, B30.3 | wavefronts per load from addresses: 5 of 6 B29 kernels within 1 %, 32x128x32 +50 % (compiler, gap 20) | smem_tiling, tensor (ldmatrix) |
| DRAM (235 GB/s read, hashed channels) | B4, B18 | B4 large plateau within 5 % (G2 test); channel count not inferable (gap, B18) | smem_tiling |
| L1 (64 KB at <= 16 KB smem, bypassed by copies) | B5, B14, counters | pointer chase within 5 % (G2 test); capacity by counters, no composite | none directly (copies bypass it) |
| registers (allocation, spills) | B9, B19 | occupancy table exact (G1 test); spill curve measured, no composite yet (B40 adds the attention loop) | tensor, attention |
| TMA (bulk copy engine) | B8 (one block, 16 KB), B15 (2-D), B26.4 | not modeled under load: B35-B37 this round | tensor, attention |
| SFU (ex2, rcp, rsqrt) | B21 (rsqrt rate only) | not modeled: B38 this round | attention |
| shuffle (shfl.sync, reductions) | none | not measured: B39 this round | attention |
