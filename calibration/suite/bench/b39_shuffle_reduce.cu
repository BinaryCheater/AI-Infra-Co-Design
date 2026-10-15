// B39: warp shuffles and reductions (round 2, group 3). (a) shfl.sync bfly / up / idx latency (dependent chain, one warp per
// sub-partition) and throughput (ILP 8, 8 blocks per SM); (b) a 32-lane butterfly reduction (5 shfl + 5 FADD); (c) shuffles beside
// LDS.128 loads from another warp on the same sub-partition; (d) redux.sync; (e) a 256-thread block reduction through shared memory.
#include "common.cuh"
#define SHFL_BFLY(v, m) asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(v) : "r"(v), "r"(m))
#define SHFL_UP(v, d) asm volatile("shfl.sync.up.b32 %0, %1, %2, 0, 0xffffffff;" : "=r"(v) : "r"(v), "r"(d))
#define SHFL_IDX(v, s) asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(v) : "r"(v), "r"(s))
template <int MODE> __global__ void shfl_lat(unsigned* out, int iters, long long* cyc) {
  unsigned v = threadIdx.x; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) { if (MODE == 0) SHFL_BFLY(v, 1); else if (MODE == 1) SHFL_UP(v, 1); else SHFL_IDX(v, (v & 31)); }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = v; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
template <int MODE> __global__ void shfl_tp(unsigned* out, int iters, long long* cyc) {
  unsigned v0 = threadIdx.x, v1 = 1, v2 = 2, v3 = 3, v4 = 4, v5 = 5, v6 = 6, v7 = 7; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) { if (MODE == 0) { SHFL_BFLY(v0, 1); SHFL_BFLY(v1, 2); SHFL_BFLY(v2, 4); SHFL_BFLY(v3, 8); SHFL_BFLY(v4, 16); SHFL_BFLY(v5, 1); SHFL_BFLY(v6, 2); SHFL_BFLY(v7, 4); }
    else if (MODE == 1) { SHFL_UP(v0, 1); SHFL_UP(v1, 2); SHFL_UP(v2, 4); SHFL_UP(v3, 8); SHFL_UP(v4, 16); SHFL_UP(v5, 1); SHFL_UP(v6, 2); SHFL_UP(v7, 4); }
    else { SHFL_IDX(v0, 1); SHFL_IDX(v1, 2); SHFL_IDX(v2, 4); SHFL_IDX(v3, 8); SHFL_IDX(v4, 16); SHFL_IDX(v5, 3); SHFL_IDX(v6, 5); SHFL_IDX(v7, 7); } }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
__global__ void butterfly(float* out, int iters, long long* cyc) {   // (b): 5 shfl.bfly + 5 FADD, dependent, per reduction
  float v = threadIdx.x * 0.5f; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) { unsigned u = __float_as_uint(v), w; asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(w) : "r"(u), "r"(m)); v += __uint_as_float(w); } v *= 0.03125f; }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = v; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
__global__ void redux(int* out, int iters, long long* cyc) {   // (d)
  int v = threadIdx.x; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) { int r; asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v)); v = (r & 7) + threadIdx.x; }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = v; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
__global__ void __launch_bounds__(256) shfl_beside_lds(unsigned* out, float* fout, int iters, long long* cyc, int lds_on) {   // (c): warps 0-3 shuffle (ILP 8), warps 4-7 LDS.128 loop
  extern __shared__ __align__(128) char sm[]; int warp = threadIdx.x >> 5, lane = threadIdx.x & 31; unsigned base = (unsigned)__cvta_generic_to_shared(sm); asm volatile("" : "+r"(iters)); long long t0 = clock64();
  if (warp < 4) { unsigned v0 = threadIdx.x, v1 = 1, v2 = 2, v3 = 3, v4 = 4, v5 = 5, v6 = 6, v7 = 7; for (int k = 0; k < iters; k++) { SHFL_BFLY(v0, 1); SHFL_BFLY(v1, 2); SHFL_BFLY(v2, 4); SHFL_BFLY(v3, 8); SHFL_BFLY(v4, 16); SHFL_BFLY(v5, 1); SHFL_BFLY(v6, 2); SHFL_BFLY(v7, 4); } out[blockIdx.x * 256 + threadIdx.x] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
  else if (lds_on) { float acc = 0; for (int k = 0; k < iters; k++) {
#pragma unroll
      for (int u = 0; u < 8; u++) { float4 v; unsigned a = base + ((lane * 16 + u * 512 + k * 16) & 8191); asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); acc += v.x; } asm volatile("" : "+f"(acc)); } fout[blockIdx.x * 256 + threadIdx.x] = acc; }
  long long t1 = clock64(); if (lane == 0) cyc[blockIdx.x * 8 + warp] = t1 - t0;
}
__global__ void block_reduce(float* out, int iters, long long* cyc) {   // (e): 256 threads -> 1 value through shared memory (warp shuffles then 8 partials)
  __shared__ float part[8]; float v = threadIdx.x * 0.5f; int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int k = 0; k < iters; k++) { float s = v;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) s += __shfl_xor_sync(0xffffffff, s, m);
    if (lane == 0) part[warp] = s; __syncthreads(); float t = part[lane & 7]; __syncthreads();
#pragma unroll
    for (int m = 4; m >= 1; m >>= 1) t += __shfl_xor_sync(0xffffffff, t, m); v = t * 1e-4f + threadIdx.x * 0.5f; }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = v; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; unsigned* out; CK(cudaMalloc(&out, (size_t)sms * 8 * 256 * 4)); float* fout; CK(cudaMalloc(&fout, (size_t)sms * 8 * 256 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 8192));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  const char* mode[] = {"bfly", "up", "idx"};
  auto timed = [&](auto go, int blocks, int stride = 1) { go(); CK(cudaDeviceSynchronize()); std::vector<double> s; if (probe) { go(); CK(cudaDeviceSynchronize()); return s; } for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h((size_t)blocks * stride); CK(cudaMemcpy(h.data(), cyc, h.size() * 8, cudaMemcpyDeviceToHost)); std::vector<long long> v; for (int i = 0; i < blocks; i++) v.push_back(h[(size_t)i * stride]); std::sort(v.begin(), v.end()); s.push_back((double)v[blocks / 2]); } return s; };
  auto lat = [&](auto kern, int m) { int iters = 4000; std::vector<double> c = timed([&]() { kern<<<sms, 128>>>(out, iters, cyc); CK(cudaGetLastError()); }, sms); std::vector<double> s; for (double v : c) s.push_back(v / iters); char q[64], d[160]; snprintf(q, sizeof q, "shfl_latency_cycles[%s]", mode[m]); snprintf(d, sizeof d, "{\"mode\": \"%s\", \"method\": \"dependent chain, one warp per sub-partition, cycles per shuffle (the loop adds about one cycle)\"}", mode[m]); emit_clocked(q, "cycles", s, d, mhz); };
  auto tp = [&](auto kern, int m) { int iters = 20000, blocks = sms * 8; std::vector<double> c = timed([&]() { kern<<<blocks, 256>>>(out, iters, cyc); CK(cudaGetLastError()); }, blocks); if (probe) return; std::vector<double> s; for (double v : c) s.push_back(8.0 * iters * 8 * 8 / (v / hz)); char q[64], d[160]; snprintf(q, sizeof q, "shfl_throughput_per_sm[%s]", mode[m]); snprintf(d, sizeof d, "{\"mode\": \"%s\", \"method\": \"ILP 8, 8 blocks x 256 threads per SM, warp-instructions per second per SM\"}", mode[m]); emit_clocked(q, "instr/s", s, d, mhz); };
  if (probe) { if (!strcmp(probe, "shfl")) tp(shfl_tp<0>, 0); else { int blocks = sms; CK(cudaFuncSetAttribute(shfl_beside_lds, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192)); timed([&]() { shfl_beside_lds<<<blocks, 256, 8192>>>(out, fout, 4000, cyc, 1); CK(cudaGetLastError()); }, blocks); } return 0; }
  lat(shfl_lat<0>, 0); lat(shfl_lat<1>, 1); lat(shfl_lat<2>, 2); tp(shfl_tp<0>, 0); tp(shfl_tp<1>, 1); tp(shfl_tp<2>, 2);
  { int iters = 4000; std::vector<double> c = timed([&]() { butterfly<<<sms, 128>>>(fout, iters, cyc); CK(cudaGetLastError()); }, sms); std::vector<double> s; for (double v : c) s.push_back(v / iters); emit_clocked("warp_reduce_cycles", "cycles", s, "{\"method\": \"32-lane butterfly: 5 shfl.bfly + 5 FADD + 1 FMUL, dependent, one warp per sub-partition, cycles per reduction\"}", mhz); }
  { int iters = 4000; std::vector<double> c = timed([&]() { redux<<<sms, 128>>>((int*)out, iters, cyc); CK(cudaGetLastError()); }, sms); std::vector<double> s; for (double v : c) s.push_back(v / iters); emit_clocked("redux_cycles", "cycles", s, "{\"supported\": true, \"method\": \"redux.sync.add.s32 dependent chain, one warp per sub-partition, cycles per redux (plus an IADD and a LOP)\"}", mhz); }
  { CK(cudaFuncSetAttribute(shfl_beside_lds, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192)); int iters = 4000; for (int on : {0, 1}) { std::vector<double> c = timed([&]() { shfl_beside_lds<<<sms, 256, 8192>>>(out, fout, iters, cyc, on); CK(cudaGetLastError()); }, sms, 8); std::vector<double> s; for (double v : c) s.push_back(v / (iters * 8.0));
      char q[64], d[220]; snprintf(q, sizeof q, "shfl_cycles_beside_lds[lds=%d]", on); snprintf(d, sizeof d, "{\"lds_on\": %d, \"method\": \"warps 0-3: 8 independent shfl.bfly per iteration; warps 4-7: 8 LDS.128 per iteration when lds=1, idle when 0; the shuffle warps' cycles per shuffle\"}", on); emit_clocked(q, "cycles", s, d, mhz); } }
  { int iters = 2000; std::vector<double> c = timed([&]() { block_reduce<<<sms, 256>>>(fout, iters, cyc); CK(cudaGetLastError()); }, sms); std::vector<double> s; for (double v : c) s.push_back(v / iters); emit_clocked("block_reduce_cycles", "cycles", s, "{\"threads\": 256, \"method\": \"warp butterfly, 8 partials through shared memory, two __syncthreads, a 3-step butterfly over the partials; cycles per block reduction\"}", mhz); }
  return 0;
}
