// B29 harness: one wave of 48 x b blocks of a generated tiling kernel (kernel.cu, included), residency forced by dynamic
// shared-memory padding; prints cycles per k-step per block from the wall time (cudaEvent) at the in-run clock.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"error\": \"%s at %s:%d\"}\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#include "kernel.cu"
__global__ void __clock_probe(double* out) { long long c0 = clock64(); unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0)); unsigned long long g; do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); } while (g - g0 < 200000); long long c1 = clock64(); out[0] = (double)(c1 - c0) / (double)(g - g0) * 1e3; }
static int argi(int argc, char** argv, const char* k, int d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return atoi(argv[i + 1]); return d; }
int main(int argc, char** argv) {
  int M = argi(argc, argv, "--m", 4096), N = argi(argc, argv, "--n", 4096), K = argi(argc, argv, "--k", 4096), b = argi(argc, argv, "--b", 1), reps = argi(argc, argv, "--reps", 5);
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); int sms = prop.multiProcessorCount;
  int gx, gy, threads, smem, use_async; smem_tiling_geometry(M, N, K, &gx, &gy, &threads, &smem, &use_async);
  int pad = std::max(smem, (96 * 1024) / b); if (pad > 99 * 1024) pad = 99 * 1024;
  CK(cudaFuncSetAttribute(smem_tiling_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
  float *A, *B, *C; CK(cudaMalloc(&A, (size_t)M * K * 4)); CK(cudaMalloc(&B, (size_t)K * N * 4)); CK(cudaMalloc(&C, (size_t)M * N * 4)); CK(cudaMemset(A, 0, (size_t)M * K * 4)); CK(cudaMemset(B, 0, (size_t)K * N * 4));
  int blocks = sms * b; int bx = std::min(gx, blocks), by = std::max(1, blocks / bx); dim3 grid(bx, by), block(threads);   // one wave: 48 x b blocks over the tile grid
  int KT = (K + TK - 1) / TK; double* dprobe; CK(cudaMalloc(&dprobe, 8));
  for (int w = 0; w < 3; w++) smem_tiling_gemm<<<grid, block, pad>>>(A, B, C, M, N, K, use_async, -1, 0);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); std::vector<double> t; double mhz = 0;
  for (int r = 0; r < reps; r++) { if (r == reps / 2) { __clock_probe<<<1, 1>>>(dprobe); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&mhz, dprobe, 8, cudaMemcpyDeviceToHost)); }
    CK(cudaEventRecord(e0)); smem_tiling_gemm<<<grid, block, pad>>>(A, B, C, M, N, K, use_async, -1, 0); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms * 1e-3); }
  std::sort(t.begin(), t.end()); double med = t[reps / 2];
  printf("{\"blocks\": %d, \"b\": %d, \"threads\": %d, \"smem_pad\": %d, \"KT\": %d, \"median_s\": %.6e, \"sm_mhz\": %.1f, \"cycles_per_step_per_block\": %.1f, \"cycles_per_phase\": %.1f}\n", blocks, b, threads, pad, KT, med, mhz, med * mhz * 1e6 / KT / TK, med * mhz * 1e6 / KT);
  return 0;
}
