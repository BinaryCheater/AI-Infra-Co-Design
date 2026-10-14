// Harness for one generated tiling kernel: fills A and B from a fixed seed, runs the kernel
// (5 warm-ups, then --reps timed with cudaEvent), checks C against cublasSgemm on the same inputs
// at a relative tolerance, and prints one JSON object. Built by agentic_runner.py together with the
// generated kernel (#include "kernel.cu").
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"error\": \"%s at %s:%d\"}\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
static int argi(int argc, char** argv, const char* k, int d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return atoi(argv[i + 1]); return d; }
static double argd(int argc, char** argv, const char* k, double d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return atof(argv[i + 1]); return d; }

int main(int argc, char** argv) {
  int M = argi(argc, argv, "--m", 1000), N = argi(argc, argv, "--n", 1000), K = argi(argc, argv, "--k", 1000);
  int reps = argi(argc, argv, "--reps", 50), seed = argi(argc, argv, "--seed", 12345); double tol = argd(argc, argv, "--tol", 1e-4);
  int gx, gy, threads, smem, use_async; smem_tiling_geometry(M, N, K, &gx, &gy, &threads, &smem, &use_async);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N), hR((size_t)M * N);
  unsigned s = seed; auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; };
  for (auto& v : hA) v = rnd(); for (auto& v : hB) v = rnd();
  float *A, *B, *C, *R; CK(cudaMalloc(&A, hA.size() * 4)); CK(cudaMalloc(&B, hB.size() * 4)); CK(cudaMalloc(&C, hC.size() * 4)); CK(cudaMalloc(&R, hR.size() * 4));
  CK(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(C, 0, hC.size() * 4));
  if (smem > 48 * 1024) CK(cudaFuncSetAttribute(smem_tiling_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(gx, gy), block(threads);
  for (int w = 0; w < 5; w++) smem_tiling_gemm<<<grid, block, smem>>>(A, B, C, M, N, K, use_async);
  cudaError_t le = cudaDeviceSynchronize();
  if (le != cudaSuccess) { printf("{\"error\": \"kernel: %s\", \"compiled\": true, \"ran\": false}\n", cudaGetErrorString(le)); return 3; }
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<double> t(reps);
  for (int r = 0; r < reps; r++) { CK(cudaEventRecord(e0)); smem_tiling_gemm<<<grid, block, smem>>>(A, B, C, M, N, K, use_async); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t[r] = ms * 1e3; }
  std::vector<double> ts = t; std::sort(ts.begin(), ts.end());
  double median = ts[reps / 2], lo = ts[(int)(0.025 * reps)], hi = ts[std::min(reps - 1, (int)(0.975 * reps))];
  // reference: cublasSgemm, row-major via the transpose identity C^T = B^T A^T
  cublasHandle_t h; cublasCreate(&h); float alpha = 1, beta = 0;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, R, N); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC.data(), C, hC.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hR.data(), R, hR.size() * 4, cudaMemcpyDeviceToHost));
  // |c - r| <= tol * (|r| + mean|r|): relative away from zero, absolute (scaled to the output's magnitude) near it
  double mean_abs = 0; for (size_t i = 0; i < hR.size(); i++) mean_abs += fabs((double)hR[i]); mean_abs /= hR.size();
  long bad = 0, first = -1; double worst = 0, max_abs = 0;
  for (size_t i = 0; i < hC.size(); i++) { double d = fabs((double)hC[i] - hR[i]), rel = d / (fabs((double)hR[i]) + mean_abs); if (rel > tol) { bad++; if (first < 0) first = (long)i; } worst = std::max(worst, rel); max_abs = std::max(max_abs, d); }
  FILE* f = popen("nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits", "r"); char clk[128] = "?"; if (f) { if (!fgets(clk, sizeof clk, f)) strcpy(clk, "?"); pclose(f); } for (char* c = clk; *c; c++) if (*c == '\n') *c = 0;
  printf("{\"compiled\": true, \"ran\": true, \"m\": %d, \"n\": %d, \"k\": %d, \"grid\": [%d, %d], \"threads\": %d, \"smem_bytes\": %d, \"cp_async\": %d, "
         "\"reps\": %d, \"warmups\": 5, \"median_us\": %.4f, \"ci95_us\": [%.4f, %.4f], \"min_us\": %.4f, "
         "\"reference\": \"cublasSgemm\", \"tolerance\": %g, \"reference_agrees\": %s, \"disagreeing\": %ld, \"first_disagreeing_index\": %ld, "
         "\"first_disagreeing_rc\": [%ld, %ld], \"worst_rel_error\": %.3g, \"max_abs_error\": %.3g, \"mean_abs_ref\": %.4g, \"clocks_after\": \"%s\"}\n",
         M, N, K, gx, gy, threads, smem, use_async, reps, median, lo, hi, ts[0], tol, bad ? "false" : "true", bad, first,
         first < 0 ? -1L : first / N, first < 0 ? -1L : first % N, worst, max_abs, mean_abs, clk);
  return 0;
}
