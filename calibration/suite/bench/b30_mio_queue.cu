// B30: the queue between the warp scheduler and the LSU (B30_plus.md). 1. depth per sub-partition: one warp issues N
// back-to-back independent ld.shared.32, cycles per issued load vs N, alone and with a second warp on the same sub-partition.
// 2. drain rate: 16 warps per SM (4 per sub-partition) in bursts of 32 loads by width and stride, instructions per cycle.
// 3. cost per load: the same kernels under ncu (wavefronts per instruction, ncu_crosscheck.py). 4. sharing: cp.async and
// ldmatrix issue time after a burst of 32 loads vs alone.
#include "common.cuh"
template <int N> __global__ void __launch_bounds__(256) burst_issue(float* out, long long* cyc, int trips) {
  extern __shared__ __align__(16) float sm[]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) sm[i] = i * 1e-6f; __syncthreads();
  unsigned base = (unsigned)__cvta_generic_to_shared(sm) + (threadIdx.x & 31) * 4; float acc = 0; long long tot = 0;
  for (int t = 0; t < trips; t++) { float s[N]; unsigned off = (t * N & 63) * 128;
    __syncwarp(); long long t0 = clock64();
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("ld.shared.f32 %0, [%1];" : "=f"(s[i]) : "r"(base + ((off + i * 128) & 32767)));
    long long t1 = clock64();                                    // after the last issue, before any use
    tot += t1 - t0;
#pragma unroll
    for (int i = 0; i < N; i++) acc += s[i];
    asm volatile("" : "+f"(acc)); }
  if (acc == 123456.789f) out[0] = acc; if ((threadIdx.x & 31) == 0) cyc[blockIdx.x * 8 + (threadIdx.x >> 5)] = tot / trips;
}
template <int WIDTH, int STRIDE, int MASK> __global__ void __launch_bounds__(512) drain_rate(float* out, long long* cyc, int trips) {
  extern __shared__ __align__(16) char sb[]; for (int i = threadIdx.x; i < 16384; i += blockDim.x) ((float*)sb)[i] = i * 1e-6f; __syncthreads();
  unsigned base = (unsigned)__cvta_generic_to_shared(sb) + (threadIdx.x & 31) * STRIDE * WIDTH; float acc[32]; for (int i = 0; i < 32; i++) acc[i] = 0;   // independent consumers: no FADD chain
  asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) { unsigned off = ((t & 7) * 4096) & MASK;
    if (WIDTH == 4) { float s[32];
#pragma unroll
      for (int i = 0; i < 32; i++) asm volatile("ld.shared.f32 %0, [%1];" : "=f"(s[i]) : "r"(base + ((off + i * 128 * STRIDE) & MASK)));
#pragma unroll
      for (int i = 0; i < 32; i++) acc[i] += s[i]; }
    else if (WIDTH == 8) { float2 s[32];
#pragma unroll
      for (int i = 0; i < 32; i++) asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(s[i].x), "=f"(s[i].y) : "r"(base + ((off + i * 256 * STRIDE) & MASK)));
#pragma unroll
      for (int i = 0; i < 32; i++) acc[i] += s[i].x; }
    else { float4 s[32];
#pragma unroll
      for (int i = 0; i < 32; i++) asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(s[i].x), "=f"(s[i].y), "=f"(s[i].z), "=f"(s[i].w) : "r"(base + ((off + i * 512 * STRIDE) & MASK)));
#pragma unroll
      for (int i = 0; i < 32; i++) acc[i] += s[i].x; }
    asm volatile("" : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31])); }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float r = 0; for (int i = 0; i < 32; i++) r += acc[i]; if (r == 123456.789f) out[0] = r; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
template <int MODE> __global__ void __launch_bounds__(32) share_probe(const char* __restrict__ g, float* out, long long* cyc, int trips) {
  // MODE 0: 8 cp.async alone; 1: 32 ld.shared then 8 cp.async; 2: 8 ldmatrix.x4 alone; 3: 32 ld.shared then 8 ldmatrix.x4
  extern __shared__ __align__(16) char sb[]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) ((float*)sb)[i] = i * 1e-6f; __syncthreads();
  unsigned base = (unsigned)__cvta_generic_to_shared(sb); unsigned lbase = base + (threadIdx.x & 31) * 4; float acc = 0; long long tot = 0; unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  for (int t = 0; t < trips; t++) {
    float s[32]; unsigned off = (t & 63) * 128;
    if (MODE & 1) {
#pragma unroll
      for (int i = 0; i < 32; i++) asm volatile("ld.shared.f32 %0, [%1];" : "=f"(s[i]) : "r"(lbase + ((off + i * 128) & 32767))); }
    long long t0 = clock64();
    if (MODE < 2) {
#pragma unroll
      for (int i = 0; i < 8; i++) { unsigned dst = base + 16384 + (i * 512 + threadIdx.x * 16); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(g + (t & 255) * 4096 + i * 512 + threadIdx.x * 16)); } }
    else {
#pragma unroll
      for (int i = 0; i < 8; i++) { unsigned a = base + ((i * 1024 + (threadIdx.x & 15) * 64 + (threadIdx.x >> 4) * 16) & 16383); unsigned x0, x1, x2, x3; asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(a)); r0 ^= x0; r1 ^= x1; r2 ^= x2; r3 ^= x3; } }
    long long t1 = clock64(); tot += t1 - t0;
    if (MODE < 2) asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::: "memory");
    if (MODE & 1) {
#pragma unroll
      for (int i = 0; i < 32; i++) acc += s[i]; }
    asm volatile("" : "+f"(acc), "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3)); }
  if (acc == 123456.789f || (r0 ^ r1 ^ r2 ^ r3) == 0x12345678u) out[0] = acc; if (threadIdx.x == 0) cyc[blockIdx.x] = tot / trips;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); float* out; CK(cudaMalloc(&out, 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096)); char* g; CK(cudaMalloc(&g, 2 << 20)); CK(cudaMemset(g, 1, 2 << 20));
  const int SMEM = 90 * 1024;
  // 1. depth: cycles per issued load vs N, one warp per SM (32 threads) and two warps per sub-partition (256 threads: warps w, w+4 share)
  std::string tab1 = "["; std::string tab2 = "["; bool f1 = true;
  auto depth = [&](auto kern, int N, int threads, std::vector<double>& s) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); kern<<<sms, threads, SMEM>>>(out, cyc, 200); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; r++) { kern<<<sms, threads, SMEM>>>(out, cyc, 200); CK(cudaDeviceSynchronize()); std::vector<long long> h((size_t)sms * 8); CK(cudaMemcpy(h.data(), cyc, sms * 8 * 8, cudaMemcpyDeviceToHost)); std::vector<long long> v; for (int b = 0; b < sms; b++) for (int w = 0; w < threads / 32; w++) v.push_back(h[b * 8 + w]); std::sort(v.begin(), v.end()); s.push_back((double)v[v.size() / 2] / N); } std::sort(s.begin(), s.end()); };
#define DEPTH(N) { std::vector<double> s1, s2; depth(burst_issue<N>, N, 32, s1); depth(burst_issue<N>, N, 256, s2); char q[64], d[160]; \
    snprintf(q, sizeof q, "mio_issue_cycles_per_load[n=%d,warps_per_smsp=1]", N); snprintf(d, sizeof d, "{\"n\": %d, \"warps_per_smsp\": 1}", N); emit_clocked(q, "cycles", s1, d, mhz); \
    snprintf(q, sizeof q, "mio_issue_cycles_per_load[n=%d,warps_per_smsp=2]", N); snprintf(d, sizeof d, "{\"n\": %d, \"warps_per_smsp\": 2}", N); emit_clocked(q, "cycles", s2, d, mhz); \
    char b[96]; snprintf(b, sizeof b, "%s[%d, %.2f, %.2f]", f1 ? "" : ", ", N, s1[s1.size() / 2], s2[s2.size() / 2]); tab1 += b; f1 = false; }
  DEPTH(1) DEPTH(2) DEPTH(4) DEPTH(6) DEPTH(8) DEPTH(12) DEPTH(16) DEPTH(20) DEPTH(24) DEPTH(32) DEPTH(48) DEPTH(64)
  tab1 += "]";
  emit_clocked("mio_queue_depth_table", "cycles", std::vector<double>{0.0}, ("{\"table_n_cyclesPerLoad_1warp_2warps\": " + tab1 + ", \"method\": \"one warp (or two on the same sub-partition) issues n back-to-back independent ld.shared.32; clock64 before the first and after the last issue, cycles per load; the n where it jumps is the queue depth\"}").c_str(), mhz);
  // 2. drain rate: 16 warps per SM, bursts of 32 loads, instructions per SM cycle by width and stride
  auto drain = [&](auto kern, int width, int stride, int span = 65536) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); int trips = 400; kern<<<sms, 512, SMEM>>>(out, cyc, trips); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); std::vector<double> s;
    for (int r = 0; r < reps; r++) { kern<<<sms, 512, SMEM>>>(out, cyc, trips); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back(16.0 * 32 * trips / (double)h[sms / 2]); }
    char q[96], d[160]; if (span == 65536) snprintf(q, sizeof q, "lds_instr_per_cycle_per_sm[width=%d,stride=%d]", width, stride); else snprintf(q, sizeof q, "lds_instr_per_cycle_per_sm[width=%d,stride=%d,span=%d]", width, stride, span); snprintf(d, sizeof d, "{\"width_bytes\": %d, \"stride_words\": %d, \"warps_per_sm\": 16, \"burst\": 32, \"address_span_bytes\": %d}", width, stride, span); emit_clocked(q, "instr/cycle", s, d, mhz); };
  drain(drain_rate<4, 1, 65535>, 4, 1); drain(drain_rate<4, 2, 65535>, 4, 2); drain(drain_rate<4, 4, 65535>, 4, 4); drain(drain_rate<4, 8, 65535>, 4, 8);
  drain(drain_rate<8, 1, 65535>, 8, 1); drain(drain_rate<8, 2, 65535>, 8, 2); drain(drain_rate<8, 4, 65535>, 8, 4);
  drain(drain_rate<16, 1, 65535>, 16, 1); drain(drain_rate<16, 2, 65535>, 16, 2); drain(drain_rate<16, 4, 65535>, 16, 4);
  drain(drain_rate<4, 1, 8191>, 4, 1, 8192); drain(drain_rate<4, 1, 32767>, 4, 1, 32768); drain(drain_rate<8, 1, 8191>, 8, 1, 8192); drain(drain_rate<16, 1, 8191>, 16, 1, 8192);
  // 4. sharing: cp.async / ldmatrix issue time alone and after a burst of 32 ld.shared, one warp per SM
  auto share = [&](auto kern, const char* name) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); kern<<<sms, 32, SMEM>>>(g, out, cyc, 200); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); std::vector<double> s;
    for (int r = 0; r < reps; r++) { kern<<<sms, 32, SMEM>>>(g, out, cyc, 200); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[sms / 2] / 8); }
    char q[80]; snprintf(q, sizeof q, "mio_share_issue_cycles[%s]", name); emit_clocked(q, "cycles", s, "{\"instructions\": 8, \"method\": \"clock64 around 8 instructions issued by one warp, alone or right after 32 ld.shared\"}", mhz); };
  share(share_probe<0>, "cp_async_alone"); share(share_probe<1>, "cp_async_after_lds_burst"); share(share_probe<2>, "ldmatrix_alone"); share(share_probe<3>, "ldmatrix_after_lds_burst");
  return 0;
}
