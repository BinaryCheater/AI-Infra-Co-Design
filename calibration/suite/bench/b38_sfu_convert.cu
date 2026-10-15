// B38: the SFU and conversion pipes (round 2, group 3). B2's pattern (ILP 8, 8 blocks x 256 threads per SM) for ex2, rcp, rsqrt, lg2,
// tanh (.approx), F2F fp32->bf16, F2FP packed bf16x2, I2F; dependent chains (one warp per sub-partition) for latency; ex2 mixed with
// FFMA at 1:1, 1:4, 1:16 from the same warp. --probe ex2 / ffma for the pipe counters.
#include "common.cuh"
#include <cuda_bf16.h>
#define EX2(x) asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x))
#define RCP(x) do { asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x)); x += 1.0f; } while (0)   // ptxas folds rcp(rcp(x)); the FADD (4 cycles, B27) keeps the chain and is noted in the fitter
#define RSQ(x) asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x))
#define LG2(x) asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(x))
#define TANH(x) asm volatile("tanh.approx.f32 %0, %1;" : "=f"(x) : "f"(x))
#define F2BF(x) do { unsigned short h; asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x)); asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(x) : "h"(h)); } while (0)
#define F2BF2(x, y) do { unsigned p; asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(x), "f"(y)); asm volatile("mov.b32 %0, %1;" : "=r"(p) : "r"(p)); x = __uint_as_float(p & 0xffff0000u); y = __uint_as_float(p << 16); } while (0)
#define I2F(x) do { int i = __float_as_int(x); asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(x) : "r"(i)); } while (0)
#define OP8(OP) OP(x0); OP(x1); OP(x2); OP(x3); OP(x4); OP(x5); OP(x6); OP(x7);
template <int OP> __global__ void sfu_tp(float* out, int iters, long long* cyc) {   // throughput: ILP 8
  float x0 = 1.0f + threadIdx.x * 1e-3f, x1 = 1.1f, x2 = 1.2f, x3 = 1.3f, x4 = 1.4f, x5 = 1.5f, x6 = 1.6f, x7 = 1.7f; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) {
    if (OP == 0) { OP8(EX2) } else if (OP == 1) { OP8(RCP) } else if (OP == 2) { OP8(RSQ) } else if (OP == 3) { OP8(LG2) } else if (OP == 4) { OP8(TANH) } else if (OP == 5) { OP8(F2BF) } else if (OP == 6) { F2BF2(x0, x1); F2BF2(x2, x3); F2BF2(x4, x5); F2BF2(x6, x7); } else { OP8(I2F) } }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
template <int OP> __global__ void sfu_lat(float* out, int iters, long long* cyc) {   // latency: one dependent chain
  float x0 = 1.0f + threadIdx.x * 1e-3f, x1 = 0; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) { if (OP == 0) EX2(x0); else if (OP == 1) RCP(x0); else if (OP == 2) RSQ(x0); else if (OP == 3) LG2(x0); else if (OP == 4) TANH(x0); else if (OP == 5) F2BF(x0); else if (OP == 6) F2BF2(x0, x1); else I2F(x0); }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
template <int FF> __global__ void ex2_ffma(float* out, int iters, long long* cyc) {   // 8 independent ex2 then FF x 8 independent FFMAs per iteration, one warp per sub-partition
  float x0 = 1.0f + threadIdx.x * 1e-3f, x1 = 1.1f, x2 = 1.2f, x3 = 1.3f, x4 = 1.4f, x5 = 1.5f, x6 = 1.6f, x7 = 1.7f; float f[8]; for (int i = 0; i < 8; i++) f[i] = 1.0f + i; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) { OP8(EX2)
#pragma unroll
    for (int r = 0; r < FF; r++) for (int i = 0; i < 8; i++) f[i] = fmaf(f[i], 0.999f, 0.001f); }
  long long t1 = clock64(); float s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; for (int i = 0; i < 8; i++) s += f[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; float* out; CK(cudaMalloc(&out, (size_t)sms * 8 * 256 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  const char* name[] = {"ex2", "rcp", "rsqrt", "lg2", "tanh", "f2f_bf16", "f2fp_bf16x2", "i2f"};
  auto timed = [&](auto go, int blocks) { go(); CK(cudaDeviceSynchronize()); std::vector<double> s; if (probe) { go(); CK(cudaDeviceSynchronize()); return s; } for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(blocks); CK(cudaMemcpy(h.data(), cyc, blocks * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[blocks / 2]); } return s; };
  auto tp = [&](auto kern, int op) { int iters = 20000, blocks = sms * 8; std::vector<double> c = timed([&]() { kern<<<blocks, 256>>>(out, iters, cyc); CK(cudaGetLastError()); }, blocks); if (probe) return;
    std::vector<double> s; for (double v : c) s.push_back(8.0 * iters * 256 * 8 / (v / hz)); char q[64], d[200]; snprintf(q, sizeof q, "sfu_throughput_per_sm[%s]", name[op]); snprintf(d, sizeof d, "{\"op\": \"%s\", \"ops_per_iter_per_thread\": 8, \"method\": \"B2 pattern: ILP 8, 8 blocks x 256 threads per SM, median block's clock64\"}", name[op]); emit_clocked(q, "ops/s", s, d, mhz); };
  auto lat = [&](auto kern, int op) { int iters = 4000, blocks = sms; std::vector<double> c = timed([&]() { kern<<<blocks, 128>>>(out, iters, cyc); CK(cudaGetLastError()); }, blocks); if (probe) return;
    std::vector<double> s; for (double v : c) s.push_back(v / iters); char q[64], d[200]; snprintf(q, sizeof q, "sfu_latency_cycles[%s]", name[op]); snprintf(d, sizeof d, "{\"op\": \"%s\", \"method\": \"one dependent chain per thread, one warp per sub-partition, cycles per link (the loop adds about one cycle)\"}", name[op]); emit_clocked(q, "cycles", s, d, mhz); };
  if (probe) { if (!strcmp(probe, "ex2")) tp(sfu_tp<0>, 0); else if (!strcmp(probe, "mix4")) { int blocks = sms; timed([&]() { ex2_ffma<4><<<blocks, 128>>>(out, 4000, cyc); CK(cudaGetLastError()); }, blocks); } else tp(sfu_tp<5>, 5); return 0; }
  tp(sfu_tp<0>, 0); tp(sfu_tp<1>, 1); tp(sfu_tp<2>, 2); tp(sfu_tp<3>, 3); tp(sfu_tp<4>, 4); tp(sfu_tp<5>, 5); tp(sfu_tp<6>, 6); tp(sfu_tp<7>, 7);
  lat(sfu_lat<0>, 0); lat(sfu_lat<1>, 1); lat(sfu_lat<2>, 2); lat(sfu_lat<3>, 3); lat(sfu_lat<4>, 4); lat(sfu_lat<5>, 5); lat(sfu_lat<6>, 6); lat(sfu_lat<7>, 7);
  auto mix = [&](auto kern, int ff) { int iters = 4000, blocks = sms; std::vector<double> c = timed([&]() { kern<<<blocks, 128>>>(out, iters, cyc); CK(cudaGetLastError()); }, blocks); std::vector<double> s; for (double v : c) s.push_back(v / iters);
    char q[64], d[220]; snprintf(q, sizeof q, "sfu_ffma_mixed_cycles_per_iter[ffma_per_ex2=%d]", ff); snprintf(d, sizeof d, "{\"ffma_per_ex2\": %d, \"ex2_per_iter\": 8, \"ffma_per_iter\": %d, \"method\": \"one warp per sub-partition: 8 independent ex2 then %d x 8 independent FFMAs per iteration\"}", ff, 8 * ff, ff); emit_clocked(q, "cycles", s, d, mhz); };
  mix(ex2_ffma<0>, 0); mix(ex2_ffma<1>, 1); mix(ex2_ffma<4>, 4); mix(ex2_ffma<16>, 16);
  return 0;
}
