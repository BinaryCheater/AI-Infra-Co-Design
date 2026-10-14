// B3: tensor-core throughput per precision. Two methods: (a) cublasGemmEx at a large square size
// (what a library reaches on the whole chip), (b) a WMMA (mma.sync) microkernel with one block per
// SM and no memory traffic (the per-SM peak). FP8 needs cublasLt and is recorded as not measured.
// Method follows Jia et al. (2018) for the mma microkernel and Luo et al. (2024) for the per-precision sweep.
#include "common.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

template <typename T, typename Acc, int M, int N, int K>
__global__ void wmma_kernel(float* out, int iters) {
  wmma::fragment<wmma::matrix_a, M, N, K, T, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, M, N, K, T, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, M, N, K, Acc> c0, c1, c2, c3;
  for (int i = 0; i < a.num_elements; i++) a.x[i] = 1.0f;      // storage type is float for tf32, __half / bf16 convert from float
  for (int i = 0; i < b.num_elements; i++) b.x[i] = 0.5f;
  wmma::fill_fragment(c0, Acc(0)); wmma::fill_fragment(c1, Acc(0)); wmma::fill_fragment(c2, Acc(0)); wmma::fill_fragment(c3, Acc(0));
  for (int i = 0; i < iters; i++) { wmma::mma_sync(c0, a, b, c0); wmma::mma_sync(c1, a, b, c1); wmma::mma_sync(c2, a, b, c2); wmma::mma_sync(c3, a, b, c3); }
  // keep all four chains live: fold them into one accumulator before the store
  for (int i = 0; i < c0.num_elements; i++) c0.x[i] = c0.x[i] + c1.x[i] + c2.x[i] + c3.x[i];
  __shared__ Acc buf[4][M * N];
  int w = threadIdx.x / 32;
  wmma::store_matrix_sync(buf[w & 3], c0, N, wmma::mem_row_major);
  if (buf[w & 3][threadIdx.x & 15] == Acc(123456)) out[0] = 1;
}
template <typename T, typename Acc, int M, int N, int K>
static void run_wmma(const char* name, int reps, int iters, float* out) {
  int sms = sm_count(), threads = 128, blocks = sms * 4;     // 4 warps per block, 4 blocks per SM
  wmma_kernel<T, Acc, M, N, K><<<blocks, threads>>>(out, iters); CK(cudaDeviceSynchronize());
  Timer t; std::vector<double> s;
  for (int r = 0; r < reps; r++) { t.start(); wmma_kernel<T, Acc, M, N, K><<<blocks, threads>>>(out, iters); double ms = t.stop_ms();
    double flops = 2.0 * M * N * K * 4 /*chains*/ * (double)iters * (blocks * threads / 32); s.push_back(flops / (ms * 1e-3) / sms); }
  char d[160]; snprintf(d, sizeof d, "{\"shape\": \"m%dn%dk%d\", \"blocks\": %d, \"threads\": %d, \"iters\": %d, \"api\": \"wmma mma_sync\"}", M, N, K, blocks, threads, iters);
  char q[64]; snprintf(q, sizeof q, "tensor_flops_per_s_per_sm[%s]", name); emit(q, "flop/s", s, d);
}
static void run_cublas(const char* name, cudaDataType_t dt, cublasComputeType_t ct, int n, int reps, cublasHandle_t h) {
  size_t es = (dt == CUDA_R_32F) ? 4 : 2;
  void *A, *B, *C; CK(cudaMalloc(&A, (size_t)n * n * es)); CK(cudaMalloc(&B, (size_t)n * n * es)); CK(cudaMalloc(&C, (size_t)n * n * 4));
  CK(cudaMemset(A, 0, (size_t)n * n * es)); CK(cudaMemset(B, 0, (size_t)n * n * es));
  float alpha = 1, beta = 0;
  auto gemm = [&]() { return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, dt, n, B, dt, n, &beta, C, CUDA_R_32F, n, ct, CUBLAS_GEMM_DEFAULT); };
  if (gemm() != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx %s unsupported\n", name); cudaFree(A); cudaFree(B); cudaFree(C); return; }
  CK(cudaDeviceSynchronize());
  Timer t; std::vector<double> s;
  for (int r = 0; r < reps; r++) { t.start(); gemm(); double ms = t.stop_ms(); s.push_back(2.0 * n * n * (double)n / (ms * 1e-3)); }
  char d[128]; snprintf(d, sizeof d, "{\"n\": %d, \"api\": \"cublasGemmEx\", \"compute\": \"fp32 accumulate\"}", n);
  char q[64]; snprintf(q, sizeof q, "tensor_flops_per_s_chip[%s]", name); emit(q, "flop/s", s, d);
  cudaFree(A); cudaFree(B); cudaFree(C);
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 10), iters = argi(argc, argv, "--iters", 20000), n = argi(argc, argv, "--gemm-n", 8192);
  float* out; CK(cudaMalloc(&out, 4));
  run_wmma<__half, float, 16, 16, 16>("fp16", reps, iters, out);
  run_wmma<__nv_bfloat16, float, 16, 16, 16>("bf16", reps, iters, out);
  run_wmma<wmma::precision::tf32, float, 16, 16, 8>("tf32", reps, iters, out);
  cublasHandle_t h; if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "no cublas\n"); return 0; }
  run_cublas("fp16", CUDA_R_16F, CUBLAS_COMPUTE_32F, n, reps, h);
  run_cublas("bf16", CUDA_R_16BF, CUBLAS_COMPUTE_32F, n, reps, h);
  run_cublas("tf32", CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, n, reps, h);
  run_cublas("fp32", CUDA_R_32F, CUBLAS_COMPUTE_32F, n, reps, h);
  printf("{\"quantity\": \"tensor_fp8\", \"unit\": \"\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"FP8 GEMM needs cublasLt; not in this round\"}}\n");
  cublasDestroy(h); return 0;
}
