// B20: tensor detail. 1. FP8 (e4m3, e5m2) through cublasLt at 8192^2; 2. per-shape mma.sync rates
// (m16n8k16 fp16/bf16, m16n8k8 tf32, m16n8k32 int8 and fp8 if they compile); 3. dependent mma chain
// latency; 4. ldmatrix.x4 feed rate swizzled vs not; 5. ISA probes (recorded by preflight: wgmma, tcgen05, setmaxnreg unsupported; clusters supported).
#include "common.cuh"
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
template <int PREC> __global__ void mma_rate(float* out, int iters, long long* cyc) {
  // PREC 0: f16 m16n8k16, 1: bf16 m16n8k16, 2: tf32 m16n8k8, 3: s8 m16n8k32, 4: e4m3 m16n8k32
  unsigned a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0, b0 = 0x38003800u, b1 = b0; float c[4][4] = {}; long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (PREC == 0) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else if (PREC == 1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else if (PREC == 2) asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else if (PREC == 3) { int d0 = (int)c[j][0], d1 = (int)c[j][1], d2 = (int)c[j][2], d3 = (int)c[j][3]; asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); c[j][0] = d0; c[j][1] = d1; c[j][2] = d2; c[j][3] = d3; }
      else asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } }
  long long t1 = clock64(); float s = 0; for (int j = 0; j < 4; j++) for (int k = 0; k < 4; k++) s += c[j][k]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
__global__ void mma_latency(float* out, int iters, long long* cyc) {
  unsigned a0 = 0x3c003c00u, b0 = 0x38003800u; float c0 = 0, c1 = 0, c2 = 0, c3 = 0; long long t0 = clock64();
  for (int i = 0; i < iters; i++) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};\n" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(b0));
  long long t1 = clock64(); out[threadIdx.x] = c0 + c1 + c2 + c3; if (threadIdx.x == 0) cyc[0] = (t1 - t0) / iters;
}
template <int SWIZZLE> __global__ void ldmatrix_rate(float* out, int iters, long long* cyc) {
  __shared__ __align__(16) unsigned short sm[8192]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) sm[i] = i; __syncthreads();
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32; unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0; long long t0 = clock64();
  for (int i = 0; i < iters; i++) { int row = (lane % 16) + warp * 16, col = (lane / 16) * 8; int off = SWIZZLE ? (row * 64 + (col ^ ((row & 7) * 8))) : (row * 64 + col);   // 64 halfs per row: unswizzled rows 128 B apart hit the same banks
    unsigned addr = (unsigned)__cvta_generic_to_shared(sm + ((off + i) & 8191 & ~7)); unsigned x0, x1, x2, x3; asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(addr)); r0 ^= x0; r1 ^= x1; r2 ^= x2; r3 ^= x3; }
  long long t1 = clock64(); if ((r0 ^ r1 ^ r2 ^ r3) == 0x12345678u) out[0] = 1; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; float* out; CK(cudaMalloc(&out, (size_t)sms * 4 * 128 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 1024));
  // 2. per-shape rates
  auto rate = [&](auto kern, const char* name, double flops_per_mma) { std::vector<double> s; int iters = 20000, blocks = sms * 4, threads = 128; kern<<<blocks, threads>>>(out, iters, cyc); cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("{\"quantity\": \"tensor_rate[%s]\", \"unit\": \"flop/s\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"%s\"}}\n", name, cudaGetErrorString(e)); cudaGetLastError(); return; }
    for (int r = 0; r < reps; r++) { kern<<<blocks, threads>>>(out, iters, cyc); CK(cudaDeviceSynchronize()); std::vector<long long> h(blocks); CK(cudaMemcpy(h.data(), cyc, blocks * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); double cycles = (double)h[blocks / 2]; s.push_back(4.0 * flops_per_mma * iters * (threads / 32) * 4 /*blocks per SM*/ / (cycles / hz)); }
    char det[2048]; snprintf(det, sizeof det, "{\"shape\": \"%s\", \"chains\": 4, \"blocks_per_sm\": 4, \"warps_per_block\": 4}", name); char q[64]; snprintf(q, sizeof q, "tensor_rate[%s]", name); emit_clocked(q, "flop/s", s, det, mhz); };
  rate(mma_rate<0>, "fp16_m16n8k16", 2.0 * 16 * 8 * 16); rate(mma_rate<1>, "bf16_m16n8k16", 2.0 * 16 * 8 * 16); rate(mma_rate<2>, "tf32_m16n8k8", 2.0 * 16 * 8 * 8); rate(mma_rate<3>, "int8_m16n8k32", 2.0 * 16 * 8 * 32); rate(mma_rate<4>, "fp8_e4m3_m16n8k32", 2.0 * 16 * 8 * 32);
  // 3. latency
  { std::vector<double> s; for (int r = 0; r < reps; r++) { mma_latency<<<1, 32>>>(out, 10000, cyc); CK(cudaDeviceSynchronize()); long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); s.push_back((double)c); } emit_clocked("tensor_op_latency_cycles[fp16_m16n8k16]", "cycles", s, "{\"method\": \"dependent mma.sync chain, accumulator feeds the next\"}", mhz); }
  // 4. ldmatrix feed
  { auto run = [&](auto kern, const char* name) { std::vector<double> s; int iters = 20000, blocks = sms * 4, threads = 128; for (int r = 0; r < reps; r++) { kern<<<blocks, threads>>>(out, iters, cyc); CK(cudaDeviceSynchronize()); std::vector<long long> h(blocks); CK(cudaMemcpy(h.data(), cyc, blocks * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back(512.0 * iters * (threads / 32) * 4 / ((double)h[blocks / 2] / hz)); }
      char q[64]; snprintf(q, sizeof q, "ldmatrix_bytes_per_s_per_sm[%s]", name); emit_clocked(q, "byte/s", s, "{\"method\": \"ldmatrix.x4 (512 B per warp instruction) from a 16 KB tile, 4 warps x 4 blocks per SM\"}", mhz); };
    run(ldmatrix_rate<0>, "unswizzled"); run(ldmatrix_rate<1>, "swizzled"); }
  // 1. FP8 via cublasLt
  { cublasLtHandle_t lt; if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) { printf("{\"quantity\": \"tensor_flops_per_s_chip[fp8_e4m3]\", \"unit\": \"flop/s\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"cublasLtCreate failed\"}}\n"); return 0; }
    int n = 8192; void *A, *B; float* C; CK(cudaMalloc(&A, (size_t)n * n)); CK(cudaMalloc(&B, (size_t)n * n)); CK(cudaMalloc(&C, (size_t)n * n * 4)); CK(cudaMemset(A, 0x38, (size_t)n * n)); CK(cudaMemset(B, 0x38, (size_t)n * n));
    for (auto fmt : {std::make_pair(CUDA_R_8F_E4M3, "fp8_e4m3"), std::make_pair(CUDA_R_8F_E5M2, "fp8_e5m2")}) {
      cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA); cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB);
      cublasLtMatrixLayoutCreate(&la, fmt.first, n, n, n); cublasLtMatrixLayoutCreate(&lb, fmt.first == CUDA_R_8F_E5M2 ? CUDA_R_8F_E4M3 : fmt.first, n, n, n); cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, n, n, n);
      float alpha = 1, beta = 0; void* ws; size_t wss = 64 << 20; CK(cudaMalloc(&ws, wss)); cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref); cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof wss);
      cublasLtMatmulHeuristicResult_t heur; int nres = 0; cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &nres);
      if (st != CUBLAS_STATUS_SUCCESS || nres == 0) { printf("{\"quantity\": \"tensor_flops_per_s_chip[%s]\", \"unit\": \"flop/s\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"no cublasLt algorithm for this FP8 layout (status %d)\"}}\n", fmt.second, (int)st); continue; }
      auto gemm = [&]() { return cublasLtMatmul(lt, op, &alpha, A, la, B, lb, &beta, C, lc, C, lc, &heur.algo, ws, wss, 0); };
      if (gemm() != CUBLAS_STATUS_SUCCESS) { printf("{\"quantity\": \"tensor_flops_per_s_chip[%s]\", \"unit\": \"flop/s\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"cublasLtMatmul failed\"}}\n", fmt.second); continue; }
      CK(cudaDeviceSynchronize()); Timer t; std::vector<double> s; for (int r = 0; r < reps; r++) { t.start(); gemm(); double ms = t.stop_ms(); s.push_back(2.0 * n * n * (double)n / (ms * 1e-3)); }
      char det[2048]; snprintf(det, sizeof det, "{\"n\": 8192, \"api\": \"cublasLtMatmul\", \"inputs\": \"%s (B e4m3 when A is e5m2)\", \"accumulate\": \"fp32\"}", fmt.second); char q[64]; snprintf(q, sizeof q, "tensor_flops_per_s_chip[%s]", fmt.second); emit_clocked(q, "flop/s", s, det, mhz); cudaFree(ws); }
  }
  return 0;
}
