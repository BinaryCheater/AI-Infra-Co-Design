// B14: L1 capacity and latency per shared-memory carve-out (pointer chase vs footprint), line size,
// whether cp.async.cg / .ca / cp.async.bulk allocate in L1, and the L1 + smem pool.
#include "common.cuh"
__global__ void chase_l1(const unsigned* __restrict__ next, int steps, unsigned* out, long long* cyc) {
  extern __shared__ char pad[]; unsigned p = 0; for (int i = 0; i < 64; i++) p = next[p];           // warm
  long long t0 = clock64(); for (int i = 0; i < steps; i++) p = next[p]; long long t1 = clock64(); out[0] = p; cyc[0] = t1 - t0; if (p == 0xFFFFFFFF) pad[0] = 1;
}
template <int MODE> __global__ void copy_then_chase(const unsigned* __restrict__ src, const unsigned* __restrict__ next, int steps, unsigned* out, long long* cyc) {
  extern __shared__ __align__(128) char smem[]; __shared__ unsigned long long bar;
  // MODE 0: cp.async.cg; 1: cp.async.ca; 2: cp.async.bulk (TMA). Copy 32 KB of `src` (same addresses as the chase) into smem, wait, then chase.
  if (MODE < 2) { for (int c = threadIdx.x; c < 2048; c += blockDim.x) { unsigned s = (unsigned)__cvta_generic_to_shared(smem + c * 16); if (MODE == 0) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src + c * 4)); else asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src + c * 4)); } asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n"); }
  else { unsigned b = (unsigned)__cvta_generic_to_shared(&bar), dst = (unsigned)__cvta_generic_to_shared(smem); if (threadIdx.x == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b)); } __syncthreads();
    if (threadIdx.x == 0) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(32768) : "memory"); asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(32768), "r"(b) : "memory"); }
    unsigned done = 0; while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b) : "memory"); } }
  __syncthreads();
  if (threadIdx.x == 0) { unsigned p = 0; long long t0 = clock64(); for (int i = 0; i < steps; i++) { unsigned v; asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(next + p)); p = v; } long long t1 = clock64(); out[0] = p; cyc[0] = t1 - t0; }
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5); double mhz = probe_clock_mhz(), hz = mhz * 1e6; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  unsigned* d; CK(cudaMalloc(&d, 1 << 20)); unsigned* out; CK(cudaMalloc(&out, 4)); long long* cyc; CK(cudaMalloc(&cyc, 8)); std::vector<unsigned> h(1 << 18);
  auto build = [&](size_t bytes, size_t stride) { size_t n = bytes / stride, sw = stride / 4; for (size_t i = 0; i < n; i++) h[i * sw] = (unsigned)(((i + 1) % n) * sw); CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice)); };
  auto chase = [&](int smem, int steps) { chase_l1<<<1, 32, smem>>>(d, steps, out, cyc); CK(cudaDeviceSynchronize()); long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); return (double)c / steps; };
  CK(cudaFuncSetAttribute(chase_l1, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  // 1. capacity per carve-out (the carve-out is a hint; the dynamic smem request is what actually sets the split)
  std::string table = "{"; bool f = true; std::vector<double> caps; std::vector<int> smems = {0, 16 * 1024, 32 * 1024, 48 * 1024, 64 * 1024, 96 * 1024}; double l1_lat = 0;
  for (int smem : smems) { int carve = smem == 0 ? 0 : (int)(100.0 * smem / prop.sharedMemPerMultiprocessor); CK(cudaFuncSetAttribute(chase_l1, cudaFuncAttributePreferredSharedMemoryCarveout, std::min(100, std::max(0, carve))));
    std::string row = "["; double cap = 0; double base = 0; bool ff = true;
    for (size_t kb : {4, 8, 16, 32, 48, 64, 96, 128, 192, 256}) { build(kb << 10, 128); std::vector<double> s; for (int r = 0; r < reps; r++) s.push_back(chase(smem, 4000)); std::sort(s.begin(), s.end()); double c = s[s.size() / 2]; if (kb == 4) { base = c; if (smem == 0) l1_lat = c; } if (c < 1.5 * base) cap = kb << 10; char b[2048]; snprintf(b, sizeof b, "%s[%zu, %.1f]", ff ? "" : ", ", kb, c); row += b; ff = false; }
    row += "]"; caps.push_back(cap); char b[2048]; snprintf(b, sizeof b, "%s\"smem_%d\": {\"carveout_pct\": %d, \"cycles_by_footprint_kb\": %s, \"l1_capacity_bytes\": %.0f}", f ? "" : ", ", smem, carve, row.c_str(), cap); table += b; f = false;
    char q[48]; snprintf(q, sizeof q, "l1_bytes_by_carveout[smem=%d]", smem); char det[2048]; snprintf(det, sizeof det, "{\"dynamic_smem_bytes\": %d, \"carveout_pct\": %d}", smem, carve); emit_clocked(q, "bytes", std::vector<double>{cap}, det, mhz); }
  table += "}";
  // 2. line size at 16 KiB footprint
  std::string lines = "["; { CK(cudaFuncSetAttribute(chase_l1, cudaFuncAttributePreferredSharedMemoryCarveout, 0)); bool ff = true; for (size_t st : {32, 64, 128, 256}) { build(16 << 10, st); std::vector<double> s; for (int r = 0; r < reps; r++) s.push_back(chase(0, 4000)); std::sort(s.begin(), s.end()); char b[2048]; snprintf(b, sizeof b, "%s[%zu, %.1f]", ff ? "" : ", ", st, s[s.size() / 2]); lines += b; ff = false; } lines += "]"; }
  std::string det = "{\"table\": " + table + ", \"line_cycles_by_stride_16KiB\": " + lines + ", \"l1_hit_cycles\": " + std::to_string(l1_lat) + ", \"smem_per_sm_declared\": " + std::to_string(prop.sharedMemPerMultiprocessor) + ", \"method\": \"single-thread chase at 128 B stride; capacity = largest footprint under 1.5x the 4 KiB latency, per dynamic-smem request (the carve-out attribute is set to match)\"}";
  emit_clocked("l1_latency_ns", "ns", std::vector<double>{l1_lat / hz * 1e9}, det.c_str(), mhz);
  // 3. does cp.async allocate in L1? copy 32 KB then chase the same addresses with ld.global.ca
  { build(32 << 10, 128); CK(cudaFuncSetAttribute(copy_then_chase<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 40 * 1024)); CK(cudaFuncSetAttribute(copy_then_chase<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 40 * 1024)); CK(cudaFuncSetAttribute(copy_then_chase<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 40 * 1024));
    auto run = [&](auto kern, const char* name) { std::vector<double> s; for (int r = 0; r < reps; r++) { CK(cudaCtxResetPersistingL2Cache()); kern<<<1, 256, 33 * 1024>>>(d, d, 256, out, cyc); CK(cudaDeviceSynchronize()); long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); s.push_back((double)c / 256); }
      std::sort(s.begin(), s.end()); double c = s[s.size() / 2]; bool alloc = c < 1.5 * l1_lat; char det[2048]; snprintf(det, sizeof det, "{\"chase_cycles_after_copy\": %.1f, \"l1_hit_cycles\": %.1f, \"allocates_l1\": %s, \"method\": \"32 KB copied global->smem, then the same 256 lines chased with ld.global.ca: L1-hit latency means the copy allocated\"}", c, l1_lat, alloc ? "true" : "false");
      char q[48]; snprintf(q, sizeof q, "%s_allocates_l1", name); emit_clocked(q, "bool", std::vector<double>{alloc ? 1.0 : 0.0}, det, mhz); };
    run(copy_then_chase<0>, "cp_async_cg"); run(copy_then_chase<1>, "cp_async_ca"); run(copy_then_chase<2>, "tma"); }
  // 4. pool: L1 capacity + configured smem per setting
  { std::string pool = "["; bool ff = true; for (size_t i = 0; i < smems.size(); i++) { char b[2048]; snprintf(b, sizeof b, "%s[%d, %.0f]", ff ? "" : ", ", smems[i], caps[i] + smems[i]); pool += b; ff = false; } pool += "]";
    double mx = 0; for (size_t i = 0; i < smems.size(); i++) mx = std::max(mx, caps[i] + smems[i]); std::string det2 = "{\"l1_plus_smem_by_setting\": " + pool + ", \"method\": \"measured L1 capacity + dynamic smem request per setting; constant within one line if the pool is unified\"}"; emit_clocked("l1_smem_pool_bytes", "bytes", std::vector<double>{mx}, det2.c_str(), mhz); }
  return 0;
}
