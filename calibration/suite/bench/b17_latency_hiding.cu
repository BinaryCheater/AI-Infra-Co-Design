// B17: latency hiding. 1. bandwidth vs resident warps per SM (DRAM and L2 footprints); 2. per-warp MLP
// (independent 128-bit loads before use); 3. per-SM outstanding requests at the knee (Little's law);
// 4. issue width (FMA IPC vs warps) and dependent-chain latencies of FMA, IMAD, ld.shared, ld.global L1-hit.
#include "common.cuh"
__global__ void gs_read(const float4* __restrict__ a, size_t n, float* out) {
  extern __shared__ char pad[]; float4 acc = make_float4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = a[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) { out[0] = 1; pad[0] = 1; }
}
template <int U> __global__ void mlp_kernel(const float4* __restrict__ a, size_t n, int iters, float* out, long long* cyc, unsigned salt) {
  // Each iteration issues U independent 16 B loads (8 lanes per 128 B line, 64 KB apart, an odd multiplier walks the whole 1 GiB so the
  // lines miss L2); the next iteration's addresses depend on this iteration's data, so iterations serialize and U is the only overlap.
  extern __shared__ char pad[]; float4 acc = make_float4(0, 0, 0, 0); unsigned mask = (unsigned)n - 1; unsigned dep = 0; unsigned lane8 = (threadIdx.x & 7), warp_off = (threadIdx.x >> 3) * 8 + blockIdx.x * 64 + salt * 1000003u;   // salt moves every run to fresh lines (no L2 reuse from the previous run)
  asm volatile("" : "+r"(iters)); long long t0 = clock64();
  for (int it = 0; it < iters; it++) { float4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) { unsigned idx = (((unsigned)(it * U + u) * 4097u + warp_off + dep) * 8u + lane8) & mask; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v[u].x), "=f"(v[u].y), "=f"(v[u].z), "=f"(v[u].w) : "l"(a + idx)); }
#pragma unroll
    for (int u = 0; u < U; u++) { acc.x += v[u].x; acc.y += v[u].y; acc.z += v[u].z; acc.w += v[u].w; }
    dep = __float_as_uint(acc.x);      // the buffer is zero-filled, so dep is 0 at run time, but the next addresses depend on this iteration's data (ptxas folds an and-with-0, so no masking here)
  }
  asm volatile("" : "+f"(acc.x), "+f"(acc.y), "+f"(acc.z), "+f"(acc.w)); long long t1 = clock64();
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) { out[0] = 1; pad[0] = 1; } if (threadIdx.x == 0) cyc[blockIdx.x] = (t1 - t0) / iters;
}
__global__ void fma_ipc(float* out, int iters, long long* cyc) {
  float x0 = threadIdx.x * 1e-3f, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7; long long t0 = clock64(); asm volatile("" : "+r"(iters));   // the loop cannot be hoisted above t0
  for (int i = 0; i < iters; i++) { x0 = fmaf(x0, 0.999f, 0.001f); x1 = fmaf(x1, 0.999f, 0.001f); x2 = fmaf(x2, 0.999f, 0.001f); x3 = fmaf(x3, 0.999f, 0.001f); x4 = fmaf(x4, 0.999f, 0.001f); x5 = fmaf(x5, 0.999f, 0.001f); x6 = fmaf(x6, 0.999f, 0.001f); x7 = fmaf(x7, 0.999f, 0.001f); }
  asm volatile("" : "+f"(x0), "+f"(x1), "+f"(x2), "+f"(x3), "+f"(x4), "+f"(x5), "+f"(x6), "+f"(x7)); long long t1 = clock64();
  // block-wide window: the earliest start and the latest end over all warps (one warp's own window undercounts when warps do not overlap)
  __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
__global__ void dep_chains(long long* cyc, float* out, int* gchain, int iters) {
  // each chain is one dependent instruction per step; the empty asm barriers pin the work between the clock reads
  __shared__ int sm[1024]; for (int i = threadIdx.x; i < 1024; i += blockDim.x) sm[i] = (i + 1) & 1023; __syncthreads();
  float x = threadIdx.x; long long t0 = clock64(); asm volatile("" : "+r"(iters), "+f"(x)); for (int i = 0; i < iters; i++) x = fmaf(x, 0.999f, 0.001f); asm volatile("" : "+f"(x)); long long t1 = clock64();
  int y = threadIdx.x; asm volatile("" : "+r"(iters), "+r"(y)); for (int i = 0; i < iters; i++) { asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(y) : "r"(3), "r"(1)); } long long t2 = clock64();
  int idx = threadIdx.x; asm volatile("" : "+r"(iters), "+r"(idx)); for (int i = 0; i < iters; i++) idx = sm[idx]; asm volatile("" : "+r"(idx)); long long t3 = clock64();
  int gi = threadIdx.x; asm volatile("" : "+r"(iters), "+r"(gi)); for (int i = 0; i < iters; i++) gi = gchain[gi]; asm volatile("" : "+r"(gi)); long long t4 = clock64();   // gchain (1 KB) is L1-resident after the first touch
  if (idx == 123456 || gi == 123456) out[0] = x;
  if (threadIdx.x == 0) { cyc[0] = (t1 - t0) / iters; cyc[1] = (t2 - t1) / iters; cyc[2] = (t3 - t2) / iters; cyc[3] = (t4 - t3) / iters; }
  if (x + y + idx + gi == 123456.789f) out[1] = 1;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6;
  size_t big = 1ull << 30, small = 8ull << 20; float4* a; CK(cudaMalloc(&a, big)); CK(cudaMemset(a, 0, big)); float* out; CK(cudaMalloc(&out, 4096)); CK(cudaMemset(out, 0, 4096)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  CK(cudaFuncSetAttribute(gs_read, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  Timer t;
  // 1. bandwidth vs resident warps per SM: blocks of 32 threads (1 warp) limited by dynamic smem; >= 24 warps use 2-warp blocks
  for (const char* lvl : {"dram", "l2"}) { size_t bytes = strcmp(lvl, "dram") == 0 ? big : small; size_t n = bytes / 16; std::string table = "["; bool f = true; double knee_warps = -1, plateau = 0; std::vector<double> bw_at;
    for (int warps : {1, 2, 4, 8, 16, 24, 32, 48}) { int tpb = warps <= 16 ? 32 : 64; int bps = warps / (tpb / 32); int smem = std::max(0, (int)(96 * 1024 / bps) - 1024); if (bps > 24) { bps = 24; smem = 0; }
      int grid = sms * bps; int passes = (int)std::max<size_t>(1, (512ull << 20) / bytes); std::vector<double> s;
      gs_read<<<grid, tpb, smem>>>(a, n, out); CK(cudaDeviceSynchronize());
      for (int r = 0; r < reps; r++) { t.start(); for (int p = 0; p < passes; p++) gs_read<<<grid, tpb, smem>>>(a, n, out); double ms = t.stop_ms(); s.push_back((double)bytes * passes / (ms * 1e-3)); }
      std::sort(s.begin(), s.end()); double bw = s[s.size() / 2]; bw_at.push_back(bw); char b[64]; snprintf(b, sizeof b, "%s[%d, %.4g]", f ? "" : ", ", warps, bw); table += b; f = false;
      char q[64]; snprintf(q, sizeof q, "bandwidth_vs_warps[%s,warps=%d]", lvl, warps); char det[2048]; snprintf(det, sizeof det, "{\"level\": \"%s\", \"warps_per_sm\": %d, \"threads_per_block\": %d, \"blocks_per_sm\": %d}", lvl, warps, tpb, bps); emit_clocked(q, "byte/s", s, det, mhz); }
    table += "]"; plateau = *std::max_element(bw_at.begin(), bw_at.end()); int ws[] = {1, 2, 4, 8, 16, 24, 32, 48}; for (size_t i = 0; i < bw_at.size(); i++) if (bw_at[i] >= 0.9 * plateau) { knee_warps = ws[i]; break; }
    std::string det = "{\"level\": \"" + std::string(lvl) + "\", \"table_warps_bandwidth\": " + table + ", \"plateau_bytes_per_s\": " + std::to_string(plateau) + ", \"per_warp_bytes_per_s\": " + std::to_string(bw_at[0] / sms) + ", \"method\": \"grid-stride float4 reads with resident warps per SM set by dynamic smem; knee = first warp count within 90 % of the plateau\"}";
    char q[48]; snprintf(q, sizeof q, "warps_to_hide_latency[%s]", lvl); emit_clocked(q, "warps", std::vector<double>{knee_warps}, det.c_str(), mhz); }
  // 2. MLP per warp: one warp per SM, U independent loads before use
  { std::string table = "["; std::vector<double> per_u; int limit = -1; double base = 0;
    auto run = [&](auto kern, int U) { std::vector<double> s; for (int r = 0; r < reps; r++) { kern<<<sms, 32, 90 * 1024>>>(a, big / 16, 400, out, cyc, (unsigned)(r * 7 + U)); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[sms / 2]); } std::sort(s.begin(), s.end()); per_u.push_back(s[s.size() / 2]); char b[48]; snprintf(b, sizeof b, "%s[%d, %.1f]", U > 1 ? ", " : "", U, per_u.back()); table += b; };
    CK(cudaFuncSetAttribute(mlp_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mlp_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mlp_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mlp_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mlp_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(mlp_kernel<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(fma_ipc, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    run(mlp_kernel<1>, 1); run(mlp_kernel<2>, 2); run(mlp_kernel<4>, 4); run(mlp_kernel<8>, 8); run(mlp_kernel<16>, 16); run(mlp_kernel<32>, 32); table += "]";
    int us[] = {1, 2, 4, 8, 16, 32}; limit = 1; for (size_t i = 1; i < per_u.size(); i++) if (per_u[i] <= 1.3 * per_u[0]) limit = us[i]; else break;   // flat = fully overlapped
    std::string det = "{\"table_u_cyclesPerIteration\": " + table + ", \"inferred_per_warp_outstanding_loads\": " + std::to_string(limit) + ", \"method\": \"one warp per SM; each iteration issues U independent 128-bit L2-missing loads and the next iteration depends on their data; cycles per iteration stay at the latency while all U overlap and grow once U exceeds the per-warp outstanding limit (largest U within 1.3x the U=1 cycles)\"}";
    emit_clocked("per_warp_outstanding_loads", "loads", std::vector<double>{(double)limit}, det.c_str(), mhz); }
  // 3. per-SM outstanding requests at the DRAM knee (Little's law): bytes in flight per SM = per-SM bandwidth x latency
  { double lat_dram = 299.8e-9, lat_l2 = 135.5e-9;   // B5 (the contract's values; recomputed by fit.py from the raw records)
    emit_clocked("per_sm_outstanding_requests", "requests", std::vector<double>{0}, "{\"note\": \"derived by fit.py: plateau bandwidth per SM x latency / 32 B sectors, for dram and l2, from bandwidth_vs_warps and B5\"}", mhz); (void)lat_dram; (void)lat_l2; }
  // 4. issue: FMA IPC vs warps per SM (1 block per SM)
  { std::string table = "["; bool f = true; double sat = 0; int sat_w = 0;
    for (int warps : {1, 2, 4, 8, 16}) { std::vector<double> s; for (int r = 0; r < reps; r++) { fma_ipc<<<sms, warps * 32, 90 * 1024>>>(out, 20000, cyc); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back(8.0 * 20000 * warps / h[sms / 2]); }   // warp-instructions per cycle per SM
      std::sort(s.begin(), s.end()); double ipc = s[s.size() / 2]; char b[48]; snprintf(b, sizeof b, "%s[%d, %.3f]", f ? "" : ", ", warps, ipc); table += b; f = false; if (ipc > sat * 1.1) { sat = ipc; sat_w = warps; } }
    table += "]";
    std::string det = "{\"table_warps_warpInstrPerCycle\": " + table + ", \"saturation_ipc\": " + std::to_string(sat) + ", \"saturation_warps\": " + std::to_string(sat_w) + ", \"method\": \"independent FMA chains (ILP 8) from w warps in one block per SM; warp-instructions per cycle per SM; the saturating IPC is schedulers x issue width\"}";
    emit_clocked("issue_width_per_sm", "warp-instr/cycle", std::vector<double>{sat}, det.c_str(), mhz); }
  CK(cudaFuncSetAttribute(fma_ipc, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  { int* gchain; CK(cudaMalloc(&gchain, 1024 * 4)); { std::vector<int> hc(1024); for (int i = 0; i < 1024; i++) hc[i] = (i + 1) & 255; CK(cudaMemcpy(gchain, hc.data(), 4096, cudaMemcpyHostToDevice)); } dep_chains<<<1, 32>>>(cyc, out, gchain, 100); CK(cudaDeviceSynchronize()); std::vector<double> s[4]; for (int r = 0; r < reps; r++) { dep_chains<<<1, 32>>>(cyc, out, gchain, 10000); CK(cudaDeviceSynchronize()); long long h[4]; CK(cudaMemcpy(h, cyc, 32, cudaMemcpyDeviceToHost)); for (int i = 0; i < 4; i++) s[i].push_back((double)h[i]); }
    const char* names[] = {"fma", "imad", "ld_shared", "ld_l1hit"}; for (int i = 0; i < 4; i++) { char q[48]; snprintf(q, sizeof q, "op_latency_cycles[%s]", names[i]); emit_clocked(q, "cycles", s[i], "{\"method\": \"single-warp dependent chain, clock64 per op\"}", mhz); } }
  return 0;
}
