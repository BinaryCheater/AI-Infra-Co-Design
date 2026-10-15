// B31: mma.sync issue rate and latency per shape (round 2, group 1). One warp per sub-partition (block of 128 threads, one block
// per SM via smem padding). (a) dependent chain per shape; (b) a independent accumulators round-robin, a in {1,2,4,8,16};
// (c) the same with 2 and 4 warps per sub-partition. Cycles per mma per warp from a block-wide clock64 window.
#include "common.cuh"
#include <cuda_fp16.h>
#define MMA_F16 "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#define MMA_BF16 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#define MMA_TF32 "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#define MMA_FP8 "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
template <int SHAPE, int A> __device__ __forceinline__ void mma_step(float (&c)[16][4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#pragma unroll
  for (int j = 0; j < A; j++) {
    if (SHAPE == 0) asm volatile(MMA_F16 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    else if (SHAPE == 1) asm volatile(MMA_BF16 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    else if (SHAPE == 2) asm volatile(MMA_TF32 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    else asm volatile(MMA_FP8 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
}
__global__ void __launch_bounds__(512) mma_int8(float* out, int iters, long long* cyc, int A) {   // INT8 m16n8k32 has s32 accumulators: separate kernel, a <= 4
  extern __shared__ char pad[]; int c[4][4] = {}; unsigned a0 = 0x01010101u, b0 = 0x02020202u; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) if (j < A) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};" : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3]) : "r"(a0), "r"(b0)); }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  int s = 0; for (int j = 0; j < 4; j++) s += c[j][0]; out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin; if (iters < 0) pad[0] = 1;
}
template <int SHAPE, int A> __global__ void __launch_bounds__(512) mma_indep(float* out, int iters, long long* cyc) {
  extern __shared__ char pad[]; float c[16][4] = {}; unsigned a0 = 0x3c003c00u, b0 = 0x38003800u; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int i = 0; i < iters; i++) mma_step<SHAPE, A>(c, a0, a0, a0, a0, b0, b0);
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float s = 0; for (int j = 0; j < A; j++) s += c[j][0]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin; if (iters < 0) pad[0] = 1;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); float* out; CK(cudaMalloc(&out, (size_t)sms * 512 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  const char* names[] = {"fp16_m16n8k16", "bf16_m16n8k16", "tf32_m16n8k8", "fp8_e4m3_m16n8k32"}; const double flops[] = {2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 8, 2.0 * 16 * 8 * 32};
  auto run = [&](auto kern, int shape, int A, int w, bool emit) { int threads = 128 * w; int iters = 2000; CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    kern<<<sms, threads, 90 * 1024>>>(out, iters, cyc); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); if (probe) { kern<<<sms, threads, 90 * 1024>>>(out, iters, cyc); CK(cudaDeviceSynchronize()); return 0.0; }
    std::vector<double> s; for (int r = 0; r < reps; r++) { kern<<<sms, threads, 90 * 1024>>>(out, iters, cyc); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[sms / 2] / (iters * (double)A)); }
    std::sort(s.begin(), s.end()); double cpm = s[s.size() / 2];
    if (emit) { char q[96], d[256]; snprintf(q, sizeof q, "mma_cycles_per_op[shape=%s,a=%d,w=%d]", names[shape], A, w); snprintf(d, sizeof d, "{\"shape\": \"%s\", \"a\": %d, \"warps_per_smsp\": %d, \"flops_per_mma\": %.0f, \"warp_mma_per_cycle_per_smsp\": %.4f}", names[shape], A, w, flops[shape], w / cpm); emit_clocked(q, "cycles", s, d, mhz); }
    return cpm; };
  if (probe) {   // --probe <shape> : the saturated configuration (a = 16, w = 4) twice, for ncu
    int sh = atoi(probe); if (sh == 0) run(mma_indep<0, 16>, 0, 16, 4, false); else if (sh == 1) run(mma_indep<1, 16>, 1, 16, 4, false); else if (sh == 2) run(mma_indep<2, 16>, 2, 16, 4, false); else run(mma_indep<3, 16>, 3, 16, 4, false); return 0; }
#define GRID(S) run(mma_indep<S, 1>, S, 1, 1, true); run(mma_indep<S, 2>, S, 2, 1, true); run(mma_indep<S, 4>, S, 4, 1, true); run(mma_indep<S, 8>, S, 8, 1, true); run(mma_indep<S, 16>, S, 16, 1, true); \
  run(mma_indep<S, 1>, S, 1, 2, true); run(mma_indep<S, 4>, S, 4, 2, true); run(mma_indep<S, 16>, S, 16, 2, true); run(mma_indep<S, 1>, S, 1, 4, true); run(mma_indep<S, 4>, S, 4, 4, true); run(mma_indep<S, 16>, S, 16, 4, true);
  GRID(0) GRID(1) GRID(2) GRID(3)
  // INT8 (s32 accumulators), a in {1, 4}, w in {1, 4}
  for (int w : {1, 4}) for (int A : {1, 4}) { int threads = 128 * w, iters = 2000; CK(cudaFuncSetAttribute(mma_int8, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); mma_int8<<<sms, threads, 90 * 1024>>>(out, iters, cyc, A); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); std::vector<double> s;
    for (int r = 0; r < reps; r++) { mma_int8<<<sms, threads, 90 * 1024>>>(out, iters, cyc, A); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[sms / 2] / (iters * (double)A)); }
    char q[96], d[200]; snprintf(q, sizeof q, "mma_cycles_per_op[shape=int8_m16n8k32,a=%d,w=%d]", A, w); snprintf(d, sizeof d, "{\"shape\": \"int8_m16n8k32\", \"a\": %d, \"warps_per_smsp\": %d, \"flops_per_mma\": %.0f}", A, w, 2.0 * 16 * 8 * 32); emit_clocked(q, "cycles", s, d, mhz); }
  return 0;
}
