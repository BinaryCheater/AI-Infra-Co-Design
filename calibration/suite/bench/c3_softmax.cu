// C3: the group-3 composite (round 2): online softmax over rows of 128 keys, FP32, one warp per row (4 values per lane from global,
// L2 resident), row max and row sum by 5-step butterflies, ex2 on the SFU, rcp, BF16 convert on the way out (8 B per lane stored).
// 128 threads per block at 1, 2, 4, 6 blocks per SM (residency by a shared-memory pad). Cycles per row per block.
#include "common.cuh"
#include <cuda_bf16.h>
#define EX2(x) asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x))
#define RCP(x) asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x))
__global__ void __launch_bounds__(128) softmax_rows(const float4* __restrict__ X, __nv_bfloat162* __restrict__ Y, int rows_per_warp, long long* cyc) {
  extern __shared__ char pad[]; int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; const int ROWS = 4096; asm volatile("" : "+r"(rows_per_warp)); long long t0 = clock64();
  for (int r = 0; r < rows_per_warp; r++) { int row = (blockIdx.x * 4 + warp + r * 97) & (ROWS - 1);
    float4 v = X[row * 32 + lane]; float m = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, d));
    float e0 = (v.x - m) * 1.442695f, e1 = (v.y - m) * 1.442695f, e2 = (v.z - m) * 1.442695f, e3 = (v.w - m) * 1.442695f; EX2(e0); EX2(e1); EX2(e2); EX2(e3);
    float s = e0 + e1 + e2 + e3;
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) s += __shfl_xor_sync(0xffffffff, s, d);
    RCP(s); __nv_bfloat162 p0 = __floats2bfloat162_rn(e0 * s, e1 * s), p1 = __floats2bfloat162_rn(e2 * s, e3 * s); Y[(size_t)row * 64 + lane * 2] = p0; Y[(size_t)row * 64 + lane * 2 + 1] = p1; }
  long long t1 = clock64(); if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0; if (pad[threadIdx.x & 7] == 123) cyc[0] = 0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); const int ROWS = 4096;
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  float4* X; CK(cudaMalloc(&X, (size_t)ROWS * 128 * 4)); CK(cudaMemset(X, 0, (size_t)ROWS * 128 * 4)); __nv_bfloat162* Y; CK(cudaMalloc(&Y, (size_t)ROWS * 128 * 2)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  CK(cudaFuncSetAttribute(softmax_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, softmax_rows));
  auto run = [&](int bps, bool emit) { int pad = std::max(1024, (96 * 1024) / bps); int grid = sms * bps, rows = 400; auto go = [&]() { softmax_rows<<<grid, 128, pad>>>(X, Y, rows, cyc); CK(cudaGetLastError()); };
    go(); CK(cudaDeviceSynchronize()); if (probe) { go(); CK(cudaDeviceSynchronize()); return; }
    std::vector<double> s; for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(grid); CK(cudaMemcpy(h.data(), cyc, grid * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[grid / 2] / rows); }
    if (!emit) return; char q[64], d[400]; snprintf(q, sizeof q, "softmax_cycles_per_row[bps=%d]", bps);
    snprintf(d, sizeof d, "{\"blocks_per_sm\": %d, \"warps_per_block\": 4, \"keys_per_row\": 128, \"regs\": %d, \"per_row_per_warp\": {\"ldg_128\": 1, \"fmax\": 8, \"shfl\": 10, \"fadd\": 8, \"fsub_fmul\": 8, \"ex2\": 4, \"rcp\": 1, \"fmul\": 4, \"cvt_bf16x2\": 2, \"stg_64\": 2}, \"method\": \"one warp per row, 4 keys per lane; cycles per row per block\"}", bps, fa.numRegs); emit_clocked(q, "cycles", s, d, mhz); };
  if (probe) { run(atoi(probe), false); return 0; }
  for (int bps : {1, 2, 4, 6}) run(bps, true);
  return 0;
}
