// B34: FFMA and mma.sync from the same schedulers; accumulator pressure; the epilogue (round 2, group 1).
// (a) one warp per sub-partition alternating mma and independent FFMAs in ratios 1:0, 1:1, 1:4, 1:16; (b) a tensor inner loop with 64, 128,
// 192 accumulators per thread, built once per --maxrreg cap by the driver (B19's variant mechanism); (c) the loop plus its epilogue -- no __launch_bounds__ on tensor_loop: nvcc ignores -maxrregcount for kernels that carry one
// (FP32 accumulators to BF16, stored to global).
#include "common.cuh"
#include <cuda_bf16.h>
#define MMA_BF16 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
template <int FF> __global__ void __launch_bounds__(128) mma_ffma(float* out, int iters, long long* cyc) {
  extern __shared__ char pad[]; float c[4][4] = {}; float x[16]; for (int i = 0; i < 16; i++) x[i] = threadIdx.x + i; unsigned a0 = 0x3f803f80u, b0 = 0x3f803f80u; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) { asm volatile(MMA_BF16 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a0), "r"(a0), "r"(a0), "r"(b0), "r"(b0));
#pragma unroll
      for (int f = 0; f < FF; f++) x[f & 15] = fmaf(x[f & 15], 0.999f, 0.001f); } }
  asm volatile("" : "+f"(x[0]), "+f"(x[1]), "+f"(x[2]), "+f"(x[3]), "+f"(x[4]), "+f"(x[5]), "+f"(x[6]), "+f"(x[7]), "+f"(x[8]), "+f"(x[9]), "+f"(x[10]), "+f"(x[11]), "+f"(x[12]), "+f"(x[13]), "+f"(x[14]), "+f"(x[15]));
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float s = 0; for (int j = 0; j < 4; j++) s += c[j][0]; for (int i = 0; i < 16; i++) s += x[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin; if (iters < 0) pad[0] = 1;
}
// (b)/(c): a warp computes TILES m16n8 output tiles (4 accumulators each) over 16 k-steps per trip from pre-filled shared fragments

// (a2) two warps per sub-partition: warps 0-3 run independent mma (4 accumulators), warps 4-7 run independent FFMA chains (8); does the FFMA
// warp issue while the tensor pipe is busy with the other warp's mma, or do the two serialize on the scheduler as they do within one warp?
__global__ void __launch_bounds__(256) mma_ffma_split(float* out, int iters, long long* cyc, int ffma_only_iters) {
  int warp = threadIdx.x >> 5; float c[4][4] = {}; unsigned a0 = 0x3f803f80u, b0 = 0x3f803f80u; float f[8]; for (int i = 0; i < 8; i++) f[i] = 1.0f + i; float m = 1.0001f;
  asm volatile("" : "+r"(iters), "+r"(ffma_only_iters)); long long t0 = clock64();
  if (warp < 4) { for (int i = 0; i < iters; i++) {
#pragma unroll
      for (int j = 0; j < 4; j++) asm volatile(MMA_BF16 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a0), "r"(a0), "r"(a0), "r"(b0), "r"(b0)); } }
  else { for (int i = 0; i < ffma_only_iters; i++) {
#pragma unroll
      for (int j = 0; j < 8; j++) f[j] = fmaf(f[j], m, 0.5f); } }
  long long t1 = clock64(); float s = 0; for (int j = 0; j < 4; j++) s += c[j][0]; for (int j = 0; j < 8; j++) s += f[j]; out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if ((threadIdx.x & 31) == 0) cyc[blockIdx.x * 8 + warp] = t1 - t0;
}
template <int TILES, int EPI> __global__ void tensor_loop(float* out, __nv_bfloat16* gout, int trips, long long* cyc) {
  extern __shared__ __align__(16) char sm[]; for (int i = threadIdx.x; i < 4096; i += blockDim.x) ((unsigned short*)sm)[i] = 0x3f80 + (i & 7); __syncthreads();
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; unsigned base = (unsigned)__cvta_generic_to_shared(sm); float c[TILES][4]; for (int j = 0; j < TILES; j++) for (int q = 0; q < 4; q++) c[j][q] = 0;
  asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) {
#pragma unroll 4
    for (int k = 0; k < 16; k++) { unsigned fa0, fa1, fa2, fa3; unsigned arow = 16 * warp + (lane & 15), acol = 2 * (k & 1) + (lane >> 4); asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(fa0), "=r"(fa1), "=r"(fa2), "=r"(fa3) : "r"(base + arow * 64 + ((acol ^ (arow & 7)) * 16)));
#pragma unroll
      for (int j = 0; j < TILES; j++) { unsigned fb0 = 0x3f803f80u + j, fb1 = 0x3f803f80u + k; asm volatile(MMA_BF16 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(fa0), "r"(fa1), "r"(fa2), "r"(fa3), "r"(fb0), "r"(fb1)); } }
    if (EPI) {   // epilogue per trip: FP32 accumulators -> BF16 pairs -> global (the tile store), then the accumulators restart
#pragma unroll
      for (int j = 0; j < TILES; j++) { __nv_bfloat162 p0 = __floats2bfloat162_rn(c[j][0], c[j][1]), p1 = __floats2bfloat162_rn(c[j][2], c[j][3]); ((__nv_bfloat162*)gout)[(blockIdx.x * 128 + threadIdx.x) * TILES * 2 + j * 2] = p0; ((__nv_bfloat162*)gout)[(blockIdx.x * 128 + threadIdx.x) * TILES * 2 + j * 2 + 1] = p1; c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0; } } }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float s = 0; for (int j = 0; j < TILES; j++) s += c[j][0]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); int cap = argi(argc, argv, "--maxrreg", 255); int spill = argi(argc, argv, "--spill-bytes", -1);
  float* out; CK(cudaMalloc(&out, (size_t)sms * 512 * 4)); __nv_bfloat16* gout; CK(cudaMalloc(&gout, (size_t)sms * 128 * 48 * 2 * sizeof(__nv_bfloat162) + 4096)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  auto timed = [&](auto go, int per_trip, int trips) { go(); CK(cudaDeviceSynchronize()); if (probe) { go(); CK(cudaDeviceSynchronize()); return std::vector<double>{}; } std::vector<double> s; for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[sms / 2] / (trips * (double)per_trip)); } return s; };
  if (cap == 255) {   // (a) only once
    CK(cudaFuncSetAttribute(mma_ffma<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mma_ffma<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mma_ffma<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mma_ffma<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    if (probe && !strcmp(probe, "ratio4")) { timed([&]() { mma_ffma<4><<<sms, 128, 90 * 1024>>>(out, 2000, cyc); CK(cudaGetLastError()); }, 4, 2000); return 0; }
    auto emit_ratio = [&](std::vector<double> s, int ff) { char q[64], d[160]; snprintf(q, sizeof q, "mma_ffma_cycles_per_mma[ffma_per_mma=%d]", ff); snprintf(d, sizeof d, "{\"ffma_per_mma\": %d, \"warps_per_smsp\": 1, \"method\": \"one warp per sub-partition alternating 4 independent mma with %d FFMAs each\"}", ff, ff); emit_clocked(q, "cycles", s, d, mhz); };
    if (!probe) { emit_ratio(timed([&]() { mma_ffma<0><<<sms, 128, 90 * 1024>>>(out, 2000, cyc); }, 4, 2000), 0); emit_ratio(timed([&]() { mma_ffma<1><<<sms, 128, 90 * 1024>>>(out, 2000, cyc); }, 4, 2000), 1); emit_ratio(timed([&]() { mma_ffma<4><<<sms, 128, 90 * 1024>>>(out, 2000, cyc); }, 4, 2000), 4); emit_ratio(timed([&]() { mma_ffma<16><<<sms, 128, 90 * 1024>>>(out, 2000, cyc); }, 4, 2000), 16); } }
  if (cap == 255 && probe && !strcmp(probe, "split")) { CK(cudaFuncSetAttribute(mma_ffma_split, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); for (int r = 0; r < 2; r++) { mma_ffma_split<<<sms, 256, 90 * 1024>>>(out, 2000, cyc, 8000); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); } return 0; }
  if (cap == 255 && !probe) {   // (a2) split warps: the mma warps' cycles per mma and the FFMA warps' cycles per FFMA, each alone and together
    CK(cudaFuncSetAttribute(mma_ffma_split, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); int it = 2000;
    auto split = [&](int mma_it, int ffma_it, const char* label) { std::vector<double> sm_, sf; for (int r = 0; r < reps + 1; r++) { mma_ffma_split<<<sms, 256, 90 * 1024>>>(out, mma_it, cyc, ffma_it); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); if (r == 0) continue;
        std::vector<long long> h(sms * 8); CK(cudaMemcpy(h.data(), cyc, sms * 8 * 8, cudaMemcpyDeviceToHost)); std::vector<long long> a, b; for (int i = 0; i < sms; i++) for (int w = 0; w < 8; w++) (w < 4 ? a : b).push_back(h[i * 8 + w]); std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        if (mma_it) sm_.push_back((double)a[a.size() / 2] / (4.0 * mma_it)); if (ffma_it) sf.push_back((double)b[b.size() / 2] / (8.0 * ffma_it)); }
      char q[96], d[200]; if (mma_it) { snprintf(q, sizeof q, "mma_split_cycles_per_mma[%s]", label); snprintf(d, sizeof d, "{\"mma_warps_per_smsp\": 1, \"ffma_warps_per_smsp\": %d, \"method\": \"warps 0-3: 4 independent mma per iteration; warps 4-7: 8 independent FFMA chains\"}", ffma_it ? 1 : 0); emit_clocked(q, "cycles", sm_, d, mhz); }
      if (ffma_it) { snprintf(q, sizeof q, "ffma_split_cycles_per_ffma[%s]", label); snprintf(d, sizeof d, "{\"mma_warps_per_smsp\": %d, \"ffma_warps_per_smsp\": 1, \"method\": \"warps 4-7: 8 independent FFMA chains per iteration; warps 0-3: 4 independent mma\"}", mma_it ? 1 : 0); emit_clocked(q, "cycles", sf, d, mhz); } };
    split(it, 0, "mma_alone"); split(0, it * 4, "ffma_alone"); split(it, it * 4, "together"); }
  // (b) and (c): accumulators 64 (16 tiles), 128 (32), 192 (48) per thread, at this build's register cap
  cudaFuncAttributes fa;
  auto loop = [&](auto kern, int tiles, int epi) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncGetAttributes(&fa, kern)); int trips = 100; std::vector<double> s = timed([&]() { kern<<<sms, 128, 90 * 1024>>>(out, gout, trips, cyc); CK(cudaGetLastError()); }, 16 * tiles, trips);
    if (probe) return; char q[96], d[256]; snprintf(q, sizeof q, "tensor_loop_cycles_per_mma[acc=%d,epilogue=%d,maxrreg=%d]", 4 * tiles, epi, cap); snprintf(d, sizeof d, "{\"accumulators_per_thread\": %d, \"tiles_per_warp\": %d, \"epilogue\": %d, \"maxrregcount\": %d, \"regs_used\": %d, \"local_bytes\": %zu, \"spill_bytes_ptxas\": %d}", 4 * tiles, tiles, epi, cap, fa.numRegs, fa.localSizeBytes, spill); emit_clocked(q, "cycles", s, d, mhz); };
  if (probe && !strcmp(probe, "loop128")) { loop(tensor_loop<32, 0>, 32, 0); return 0; }
  loop(tensor_loop<16, 0>, 16, 0); loop(tensor_loop<32, 0>, 32, 0); loop(tensor_loop<48, 0>, 48, 0); if (cap == 255) { loop(tensor_loop<16, 1>, 16, 1); loop(tensor_loop<32, 1>, 32, 1); loop(tensor_loop<48, 1>, 48, 1); }
  return 0;
}
