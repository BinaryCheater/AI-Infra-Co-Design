// B27: dependency chains vs ILP vs TLP (B27_plus.md §1). One block per SM with 4*w warps (w per scheduler).
// Variant dep: per iteration A independent ld.shared each feeding one FFMA (acc[i] = fma(s[i], b, acc[i])), L iterations
// unrolled, loads then FFMAs in program order (asm volatile fixes the order the warp model assumes).
// Variant fma: FFMA-only chains (no loads), the latency/throughput reference. Variant lds: A interleaved shared-memory
// pointer chases, the LDS latency reference. Records cycles per iteration per warp (block-wide clock64 window).
#include "common.cuh"
#include <type_traits>
template <int A, int L> __global__ void __launch_bounds__(1024) dep_kernel(float* out, int trips, long long* cyc, unsigned one, unsigned zero) {
  // A independent chains of (ld.shared -> dependent IMAD -> next ld.shared address): ptxas cannot hoist a load above the IMAD it
  // depends on, so the per-chain latency is LDS + IMAD (4 cycles, the FFMA's latency class) and A chains overlap on the scoreboard
  extern __shared__ __align__(16) unsigned su[]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) su[i] = ((i + 32) & 8191) * 4; __syncthreads();
  unsigned p[A]; for (int i = 0; i < A; i++) p[i] = ((threadIdx.x & 31) + i * 512) * 4; unsigned base = (unsigned)__cvta_generic_to_shared(su);
  asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) {
#pragma unroll
    for (int k = 0; k < L; k++) {
#pragma unroll
      for (int i = 0; i < A; i++) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p[i]) : "r"(base + p[i]));
#pragma unroll
      for (int i = 0; i < A; i++) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(p[i]) : "r"(one), "r"(zero)); } }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  unsigned r = 0; for (int i = 0; i < A; i++) r ^= p[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = (float)r; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
template <int A, int L> __global__ void __launch_bounds__(1024) stream_kernel(float* out, int trips, long long* cyc) {
  // A independent streaming ld.shared per iteration with counter-derived addresses (the template's load pattern: no address dependency on load data)
  extern __shared__ __align__(16) float sm[]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) sm[i] = 1.0f + i * 1e-6f; __syncthreads();
  float acc[A]; for (int i = 0; i < A; i++) acc[i] = 0; unsigned base = (unsigned)__cvta_generic_to_shared(sm) + (threadIdx.x & 31) * 4; unsigned off = 0;
  asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) {
#pragma unroll
    for (int k = 0; k < L; k++) { float s[A];
#pragma unroll
      for (int i = 0; i < A; i++) asm volatile("ld.shared.f32 %0, [%1];" : "=f"(s[i]) : "r"(base + ((off + k * A + i) & 63) * 128));
#pragma unroll
      for (int i = 0; i < A; i++) acc[i] += s[i]; }
    asm volatile("" : "+r"(off)); off += L * A; }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float r = 0; for (int i = 0; i < A; i++) r += acc[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = r; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
template <int A, int L> __global__ void __launch_bounds__(1024) fma_kernel(float* out, int trips, long long* cyc) {
  extern __shared__ __align__(16) float sm[]; float acc[A]; for (int i = 0; i < A; i++) acc[i] = threadIdx.x * 1e-3f + i; float b = 0.999f, c = 0.001f; if (trips < 0) sm[0] = 1;
  asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) {
#pragma unroll
    for (int k = 0; k < L; k++) {
#pragma unroll
      for (int i = 0; i < A; i++) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(acc[i]) : "f"(b), "f"(c)); } }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float r = 0; for (int i = 0; i < A; i++) r += acc[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = r; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
template <int A, int L> __global__ void __launch_bounds__(1024) lds_kernel(float* out, int trips, long long* cyc) {
  extern __shared__ __align__(16) unsigned su[]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) su[i] = ((i + 32) & 8191) * 4; __syncthreads();   // chase: next = this + 32 words
  unsigned p[A]; for (int i = 0; i < A; i++) p[i] = ((threadIdx.x & 31) + i * 1024) * 4; unsigned base = (unsigned)__cvta_generic_to_shared(su);
  asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) {
#pragma unroll
    for (int k = 0; k < L; k++) {
#pragma unroll
      for (int i = 0; i < A; i++) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p[i]) : "r"(base + p[i])); } }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  unsigned r = 0; for (int i = 0; i < A; i++) r ^= p[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = (float)r; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
static void launch(void (*k)(float*, int, long long*, unsigned, unsigned), int g, int t, int sm, float* out, int trips, long long* cyc) { k<<<g, t, sm>>>(out, trips, cyc, 1u, 0u); }
static void launch(void (*k)(float*, int, long long*), int g, int t, int sm, float* out, int trips, long long* cyc) { k<<<g, t, sm>>>(out, trips, cyc); }
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); float* out; CK(cudaMalloc(&out, (size_t)sms * 1024 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 1024));
  const int SMEM = 90 * 1024;   // one block per SM
  std::string grid = "["; bool gf = true; double lat_fma = 0, lat_lds = 0; double best_frac = 0;
  auto run = [&](auto kern, const char* variant, int A, int L, int w) { int threads = 4 * w * 32; int trips = std::max(2, 4096 / L); CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    auto go = [&]() { launch(kern, sms, threads, SMEM, out, trips, cyc); };
    go(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); std::vector<double> s;
    for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[sms / 2] / (trips * (double)L)); }
    std::sort(s.begin(), s.end()); double cyc_it = s[s.size() / 2];
    int instr = (!strcmp(variant, "dep") || !strcmp(variant, "stream")) ? 2 * A : A;   // warp-instructions per iteration per warp (stream: LDS + FADD)
    double frac = (double)instr * w / cyc_it;                                  // issued per scheduler cycle (bound 1)
    char q[96]; snprintf(q, sizeof q, "dependency_cycles_per_iter[variant=%s,a=%d,w=%d,L=%d]", variant, A, w, L);
    char det[256]; snprintf(det, sizeof det, "{\"variant\": \"%s\", \"a\": %d, \"w\": %d, \"L\": %d, \"threads\": %d, \"instr_per_iter_per_warp\": %d, \"issue_bound_fraction\": %.4f}", variant, A, w, L, threads, instr, frac);
    emit_clocked(q, "cycles", s, det, mhz); return std::make_pair(cyc_it, frac); };
#define DEP(A, L) for (int w : {1, 2, 4, 8}) { auto r = run(dep_kernel<A, L>, "dep", A, L, w); char b[96]; snprintf(b, sizeof b, "%s[%d, %d, %d, %.1f, %.4f]", gf ? "" : ", ", A, w, L, r.first, r.second); grid += b; gf = false; best_frac = std::max(best_frac, r.second); }
#define FMA(A) for (int w : {1, 2, 4, 8}) { auto r = run(fma_kernel<A, 16>, "fma", A, 16, w); if (A == 1 && w == 1) lat_fma = r.first; }
#define LDS(A) for (int w : {1, 2, 4, 8}) { auto r = run(lds_kernel<A, 16>, "lds", A, 16, w); if (A == 1 && w == 1) lat_lds = r.first; run(stream_kernel<A, 16>, "stream", A, 16, w); }
  FMA(1) FMA(2) FMA(4) FMA(8) FMA(16) LDS(1) LDS(2) LDS(4) LDS(8) LDS(16)
  DEP(1, 4) DEP(1, 8) DEP(1, 16) DEP(1, 32) DEP(1, 64) DEP(2, 4) DEP(2, 8) DEP(2, 16) DEP(2, 32) DEP(2, 64) DEP(4, 4) DEP(4, 8) DEP(4, 16) DEP(4, 32) DEP(4, 64)
  DEP(8, 4) DEP(8, 8) DEP(8, 16) DEP(8, 32) DEP(8, 64) DEP(16, 4) DEP(16, 8) DEP(16, 16) DEP(16, 32) DEP(16, 64)
  grid += "]";
  // headline: the smallest a*w whose dep variant reaches 90 % of the issue bound at L = 16
  emit_clocked("scoreboard_latency_cycles[fma]", "cycles", std::vector<double>{lat_fma}, "{\"method\": \"fma variant, a = 1, w = 1: one dependent FFMA chain, cycles per FFMA\"}", mhz);
  emit_clocked("scoreboard_latency_cycles[lds]", "cycles", std::vector<double>{lat_lds}, "{\"method\": \"lds variant, a = 1, w = 1: one shared-memory pointer chase, cycles per load\"}", mhz);
  std::string det = "{\"grid_a_w_L_cyclesPerIter_fraction\": " + grid + ", \"method\": \"dep variant over the (a, w, L) grid: per warp, a independent chains of ld.shared -> dependent IMAD (4-cycle class, standing in for the FFMA that consumes a load), L iterations unrolled, 4w warps per SM; cycles per iteration per warp; fraction = warp-instructions issued per scheduler cycle (bound 1). The grid is the first check of the warp model (B27_plus.md §1)\"}";
  emit_clocked("issue_bound_fraction_grid", "fraction", std::vector<double>{best_frac}, det.c_str(), mhz);
  return 0;
}
