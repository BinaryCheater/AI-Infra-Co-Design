// B35: TMA (cp.async.bulk + mbarrier, 1-D) completion by tile size and residency (round 2, group 2). Tiles 4-48 KB at 1, 2, 4, 6
// resident blocks per SM (residency set by a shared-memory pad), 10 000 trials per block, issue-to-completion cycles into a
// distribution; chip and per-SM bytes/s by residency from the trial loop's wall time.
#include "tma_common.cuh"
__global__ void tma_trial(const char* __restrict__ src, long long* out, int tile, int trials) {
  extern __shared__ __align__(128) char sm[]; __shared__ unsigned long long bar; unsigned b = saddr(&bar), dst = saddr(sm);
  if (threadIdx.x == 0) mbar_init(b, 1); __syncthreads();
  for (int i = 0; i < trials; i++) { const char* base = src + (size_t)((i * 7 + blockIdx.x) & 63) * tile; __syncthreads(); long long t0 = clock64();
    if (threadIdx.x == 0) { mbar_expect_tx(b, tile); tma_load_1d(dst, base, tile, b); }
    mbar_wait(b, i & 1); long long t1 = clock64(); if (threadIdx.x == 0) out[(size_t)blockIdx.x * trials + i] = t1 - t0; }
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 3), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; const int trials = 10000;
  char* src; CK(cudaMalloc(&src, 64 * 65536)); CK(cudaMemset(src, 7, 64 * 65536)); long long* out; CK(cudaMalloc(&out, (size_t)sms * 6 * trials * 8));
  CK(cudaFuncSetAttribute(tma_trial, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  for (int resident : {1, 2, 4, 6}) for (int tile : {4096, 8192, 16384, 32768, 49152}) {
    if (resident * tile > 96 * 1024) continue; int pad = std::max(tile, (96 * 1024) / resident); int grid = sms * resident;
    if (probe && (atoi(probe) != resident || tile != 16384)) continue;
    tma_trial<<<grid, 128, pad>>>(src, out, tile, 200); CK(cudaDeviceSynchronize()); if (probe) { tma_trial<<<grid, 128, pad>>>(src, out, tile, 2000); CK(cudaDeviceSynchronize()); continue; }
    std::vector<double> p50s, bws; std::vector<long long> v; Timer t;
    for (int r = 0; r < reps; r++) { t.start(); tma_trial<<<grid, 128, pad>>>(src, out, tile, trials); double ms = t.stop_ms(); std::vector<long long> h((size_t)grid * trials); CK(cudaMemcpy(h.data(), out, h.size() * 8, cudaMemcpyDeviceToHost)); v = h; std::sort(v.begin(), v.end()); p50s.push_back((double)v[v.size() / 2]); bws.push_back((double)tile * trials * grid / (ms * 1e-3)); }
    size_t n = v.size(); double p1 = v[n / 100], p50 = v[n / 2], p99 = v[n * 99 / 100], mx = v[n - 1]; std::sort(bws.begin(), bws.end()); double bw = bws[bws.size() / 2];
    char q[80], d[600]; snprintf(q, sizeof q, "tma_latency_dist[tile=%d,resident=%d]", tile, resident);
    snprintf(d, sizeof d, "{\"tile_bytes\": %d, \"resident_blocks_per_sm\": %d, \"trials\": %zu, \"p1_cycles\": %.0f, \"p50_cycles\": %.0f, \"p99_cycles\": %.0f, \"max_cycles\": %.0f, \"p1_s\": %.3e, \"p50_s\": %.3e, \"p99_s\": %.3e, \"max_s\": %.3e, \"chip_bytes_per_s\": %.4e, \"bytes_per_s_per_sm\": %.4e, \"method\": \"one 1-D cp.async.bulk per trial with an mbarrier, issue to try_wait return, every block on every SM at once\"}", tile, resident, n, p1, p50, p99, mx, p1 / hz, p50 / hz, p99 / hz, mx / hz, bw, bw / sms);
    emit_clocked(q, "cycles", p50s, d, mhz);
    snprintf(q, sizeof q, "tma_bytes_per_s_per_sm_by_residency[tile=%d,resident=%d]", tile, resident); std::vector<double> per(bws.size()); for (size_t i = 0; i < bws.size(); i++) per[i] = bws[i] / sms; emit_clocked(q, "byte/s", per, d, mhz); }
  return 0;
}
