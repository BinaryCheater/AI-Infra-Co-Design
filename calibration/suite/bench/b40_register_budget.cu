// B40: the register file as a budget (round 2, group 3). (a) an attention-shaped inner loop (Q.K^T tile, online softmax, P.V tile)
// with the live tile scaled by TQ, built once per --maxrreg cap by the driver; spill bytes from ptxas -v through the driver and the
// local bytes per thread from the function attributes; (b) local-memory streaming with 4, 8, 16 warps per SM spilling at once;
// (c) residency vs registers: __launch_bounds__(256, MINB) kernels, the occupancy API against a measured count of co-resident blocks.
#include "common.cuh"
#include <cuda_bf16.h>
#define EX2(x) asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x))
template <int TQ> __global__ void attention_loop(const float* __restrict__ K, float* out, int steps, long long* cyc) {
  // per thread: q[TQ] held, s[TQ] scores per key step, o[TQ] running P.V; per step: TQ FFMA (score) + max/ex2/sum + TQ FFMA (P.V)
  float q[TQ], o[TQ], m = -1e30f, l = 0; for (int i = 0; i < TQ; i++) { q[i] = 1.0f + i * 1e-3f + threadIdx.x * 1e-4f; o[i] = 0; }
  int lane = threadIdx.x & 31; asm volatile("" : "+r"(steps)); long long t0 = clock64();
  for (int st = 0; st < steps; st++) { float kk = K[(st * 32 + lane) & 4095]; float s[TQ]; float mx = m;
#pragma unroll
    for (int i = 0; i < TQ; i++) { s[i] = fmaf(q[i], kk, o[i] * 1e-9f); mx = fmaxf(mx, s[i]); }
    float scale = mx - m; EX2(scale); m = mx; l *= scale; float sum = 0;
#pragma unroll
    for (int i = 0; i < TQ; i++) { float p = s[i] - mx; EX2(p); sum += p; o[i] = fmaf(o[i], scale, p * kk); }
    l += sum; }
  long long t1 = clock64(); float acc = l; for (int i = 0; i < TQ; i++) acc += o[i] + q[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = acc; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
__global__ void local_stream(float* out, int iters, long long* cyc) {   // B19's local stream, launched with 4, 8, 16 warps per block, one block per SM
  float arr[64]; for (int i = 0; i < 64; i++) arr[i] = threadIdx.x + i; float acc = 0; int j = threadIdx.x & 63; long long t0 = clock64();
  for (int k = 0; k < iters; k++) {
#pragma unroll
    for (int u = 0; u < 8; u++) acc += arr[(j + u * 8) & 63];
    j = (j + 1) & 63; arr[j] = acc; }
  asm volatile("" : "+f"(acc)); long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = acc; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
template <int R> __global__ void __launch_bounds__(256) reg_resident(float* out, long long* stamp, int spin) {   // R live floats set the register count
  float a[R]; for (int i = 0; i < R; i++) a[i] = threadIdx.x + i; long long t0 = clock64(); unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  while (clock64() - t0 < spin) {
#pragma unroll
    for (int i = 0; i < R; i++) a[i] = fmaf(a[i], 0.999f, 0.001f); }
  float s = 0; for (int i = 0; i < R; i++) s += a[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) { stamp[blockIdx.x * 3] = t0; stamp[blockIdx.x * 3 + 1] = clock64(); stamp[blockIdx.x * 3 + 2] = id; }
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; int cap = argi(argc, argv, "--maxrreg", 255); int spill = argi(argc, argv, "--spill-bytes", -1);
  float* out; CK(cudaMalloc(&out, (size_t)sms * 8 * 512 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 8192)); float* K; CK(cudaMalloc(&K, 4096 * 4)); CK(cudaMemset(K, 0, 4096 * 4));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  auto timed = [&](auto go, int blocks) { go(); CK(cudaDeviceSynchronize()); std::vector<double> s; if (probe) { go(); CK(cudaDeviceSynchronize()); return s; } for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(blocks); CK(cudaMemcpy(h.data(), cyc, blocks * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[blocks / 2]); } return s; };
  cudaFuncAttributes fa;
  auto att = [&](auto kern, int tq) { CK(cudaFuncGetAttributes(&fa, kern)); int steps = 2000, blocks = sms; std::vector<double> c = timed([&]() { kern<<<blocks, 128>>>(K, out, steps, cyc); CK(cudaGetLastError()); }, blocks); if (probe) return;
    std::vector<double> s; for (double v : c) s.push_back(v / steps); char q[96], d[320]; snprintf(q, sizeof q, "attention_loop_cycles_per_step[tq=%d,maxrreg=%d]", tq, cap);
    snprintf(d, sizeof d, "{\"tq\": %d, \"maxrregcount\": %d, \"regs_used\": %d, \"local_bytes_per_thread\": %zu, \"spill_bytes_ptxas_tq40\": %d, \"ffma_per_step\": %d, \"ex2_per_step\": %d, \"method\": \"one warp per sub-partition; per step TQ scores, online max/ex2/sum, TQ P.V FFMAs; cycles per step\"}", tq, cap, fa.numRegs, fa.localSizeBytes, spill, 2 * tq, tq + 1); emit_clocked(q, "cycles", s, d, mhz); };
  if (probe) { if (!strcmp(probe, "att40")) att(attention_loop<40>, 40); else { int blocks = sms; timed([&]() { local_stream<<<blocks, 512>>>(out, 4000, cyc); CK(cudaGetLastError()); }, blocks); } return 0; }
  att(attention_loop<8>, 8); att(attention_loop<16>, 16); att(attention_loop<24>, 24); att(attention_loop<32>, 32); att(attention_loop<40>, 40); att(attention_loop<56>, 56);
  if (cap != 255) return 0;   // (b) and (c) once
  for (int warps : {4, 8, 16}) { int iters = 4000, blocks = sms; std::vector<double> c = timed([&]() { local_stream<<<blocks, warps * 32>>>(out, iters, cyc); CK(cudaGetLastError()); }, blocks); std::vector<double> s; for (double v : c) s.push_back(8.0 * 4 * iters * warps * 32 / (v / hz));
    char q[80], d[200]; snprintf(q, sizeof q, "local_mem_bytes_per_s_per_sm_by_warps[warps=%d]", warps); snprintf(d, sizeof d, "{\"warps_per_sm\": %d, \"method\": \"B19's local stream (8 independent 4 B local loads per step per thread), one block of %d warps per SM\"}", warps, warps); emit_clocked(q, "byte/s", s, d, mhz); }
  { long long* stamp; CK(cudaMalloc(&stamp, 8 * 3 * 8192)); auto res = [&](auto kern, int minb) { CK(cudaFuncGetAttributes(&fa, kern)); int api = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, kern, 256, 0)); int grid = sms * 12, spin = 200000;
      kern<<<grid, 256>>>(out, stamp, spin); CK(cudaDeviceSynchronize()); std::vector<long long> h((size_t)grid * 3); CK(cudaMemcpy(h.data(), stamp, h.size() * 8, cudaMemcpyDeviceToHost));
      // measured residency: per SM, the most blocks whose [start, end) overlap one block's start
      int best = 0; for (int i = 0; i < sms; i++) { std::vector<std::pair<long long, long long>> iv; for (int b = 0; b < grid; b++) if (h[b * 3 + 2] == i) iv.push_back({h[b * 3], h[b * 3 + 1]}); for (auto& x : iv) { int n = 0; for (auto& y : iv) if (y.first <= x.first + 1000 && y.second > x.first + 1000) n++; best = std::max(best, n); } }
      char q[64], d[400]; snprintf(q, sizeof q, "max_blocks_per_sm_by_regs[regs=%d]", fa.numRegs); snprintf(d, sizeof d, "{\"regs\": %d, \"live_floats\": %d, \"threads\": 256, \"occupancy_api\": %d, \"measured\": %d, \"method\": \"256-thread kernel with R live FMA accumulators; API vs the most blocks co-resident on one SM by clock64 overlap (%%smid), grid 12 x SMs\"}", fa.numRegs, minb, api, best); emit_clocked(q, "count", std::vector<double>{(double)best}, d, mhz); };
    res(reg_resident<16>, 16); res(reg_resident<32>, 32); res(reg_resident<56>, 56); res(reg_resident<80>, 80); res(reg_resident<112>, 112); res(reg_resident<144>, 144); res(reg_resident<176>, 176); res(reg_resident<224>, 224); }
  return 0;
}
