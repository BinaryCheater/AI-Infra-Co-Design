// B16: the block scheduler. Each block records %smid and %globaltimer at entry (and exit), then spins,
// so a whole wave is resident together. Sub-tests: dispatch order across grid sizes, dispatch rate,
// dynamic refill, and the wave staircase. Structural: samples carry counts; the tables are in detail.
#include "common.cuh"
__global__ void probe(unsigned* sm, unsigned long long* tin, unsigned long long* tout, int spin_ns) {
  extern __shared__ char pad[];
  if (threadIdx.x == 0) { sm[blockIdx.x] = smid(); tin[blockIdx.x] = gtimer(); }
  unsigned long long t0 = gtimer(); while (gtimer() - t0 < (unsigned long long)spin_ns) { }
  if (threadIdx.x == 0) { tout[blockIdx.x] = gtimer(); if (spin_ns < 0) pad[0] = 1; }
}
__global__ void probe_alt(unsigned* sm, unsigned long long* tin, unsigned long long* tout, int short_ns, int long_ns) {
  extern __shared__ char pad[]; int spin = (blockIdx.x & 1) ? long_ns : short_ns; if (short_ns < 0) pad[0] = 1;
  if (threadIdx.x == 0) { sm[blockIdx.x] = smid(); tin[blockIdx.x] = gtimer(); }
  unsigned long long t0 = gtimer(); while (gtimer() - t0 < (unsigned long long)spin) { }
  if (threadIdx.x == 0) tout[blockIdx.x] = gtimer();
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 3), sms = sm_count(); double mhz = probe_clock_mhz();
  const int MAXB = 8192; unsigned* sm; unsigned long long *tin, *tout; CK(cudaMalloc(&sm, MAXB * 4)); CK(cudaMalloc(&tin, MAXB * 8)); CK(cudaMalloc(&tout, MAXB * 8));
  std::vector<unsigned> hs(MAXB); std::vector<unsigned long long> hin(MAXB), hout(MAXB);
  CK(cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(probe_alt, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  // 1. order
  for (int grid : {1, 8, 24, 32, 47, 48, 49, 96, 144, 192}) for (int threads : {128, 256}) for (int smem : {0, 60 * 1024}) {
    std::vector<std::string> orders; std::vector<double> distinct, two;
    for (int r = 0; r < 10; r++) {
      probe<<<grid, threads, smem>>>(sm, tin, tout, 20000); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hs.data(), sm, grid * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hin.data(), tin, grid * 8, cudaMemcpyDeviceToHost));
      std::string o; std::vector<int> cnt(1024, 0); int d = 0, t2 = 0;
      for (int b = 0; b < grid; b++) { o += std::to_string(hs[b]) + ","; if (cnt[hs[b]]++ == 0) d++; }
      for (int i = 0; i < 1024; i++) if (cnt[i] >= 2) t2++;
      orders.push_back(o); distinct.push_back(d); two.push_back(t2);
      if (r == 0) { // the table of the first repetition: sm by block index and entry time relative to the first
        unsigned long long tmin = ~0ull; for (int b = 0; b < grid; b++) tmin = std::min(tmin, hin[b]);
        std::string sms_j = "[", t_j = "["; for (int b = 0; b < grid; b++) { sms_j += (b ? "," : "") + std::to_string(hs[b]); t_j += (b ? "," : "") + std::to_string(hin[b] - tmin); }
        sms_j += "]"; t_j += "]";
        std::string det = "{\"grid\": " + std::to_string(grid) + ", \"threads\": " + std::to_string(threads) + ", \"smem_bytes\": " + std::to_string(smem) + ", \"sm_by_block\": " + sms_j + ", \"entry_ns_by_block\": " + t_j + "}";
        char q[96]; snprintf(q, sizeof q, "block_order_table[grid=%d,threads=%d,smem=%d]", grid, threads, smem); emit_clocked(q, "sms", std::vector<double>{(double)d}, det.c_str(), mhz);
      }
    }
    int same = 0; for (auto& o : orders) if (o == orders[0]) same++;
    char det[2048]; snprintf(det, sizeof det, "{\"grid\": %d, \"threads\": %d, \"smem_bytes\": %d, \"identical_orderings_fraction\": %.2f, \"sms_with_two_blocks_median\": %.0f}", grid, threads, smem, same / 10.0, two[two.size() / 2]);
    char q[96]; snprintf(q, sizeof q, "block_distinct_sms[grid=%d,threads=%d,smem=%d]", grid, threads, smem); emit_clocked(q, "sms", distinct, det, mhz);
  }
  // 2. dispatch rate: wall time of 4096 vs 48 trivial blocks (cudaEvent), blocks per second beyond the first wave; plus the globaltimer resolution
  { std::vector<double> rate; Timer t; double res = 0;
    { int grid = 4096; probe<<<grid, 128, 0>>>(sm, tin, tout, 0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hin.data(), tin, grid * 8, cudaMemcpyDeviceToHost)); std::vector<unsigned long long> tt(hin.begin(), hin.begin() + grid); std::sort(tt.begin(), tt.end());
      unsigned long long mn = ~0ull; for (int i = 1; i < grid; i++) if (tt[i] != tt[i - 1]) mn = std::min(mn, tt[i] - tt[i - 1]); res = mn == ~0ull ? 0 : (double)mn; }
    for (int r = 0; r < reps; r++) { double ms_small, ms_big; t.start(); for (int k = 0; k < 20; k++) probe<<<sms, 128, 0>>>(sm, tin, tout, 0); ms_small = t.stop_ms() / 20; t.start(); for (int k = 0; k < 20; k++) probe<<<4096, 128, 0>>>(sm, tin, tout, 0); ms_big = t.stop_ms() / 20;
      rate.push_back((4096.0 - sms) / ((ms_big - ms_small) * 1e-3)); }
    char det[2048]; snprintf(det, sizeof det, "{\"blocks\": 4096, \"threads\": 128, \"globaltimer_resolution_ns\": %.0f, \"method\": \"(time of 4096 empty blocks - time of 48) / (4096 - 48), cudaEvent over 20 launches; the entry-timestamp method is limited by the globaltimer resolution given here\"}", res);
    emit_clocked("block_dispatch_rate_per_s", "blocks/s", rate, det, mhz); }
  // 3. refill latency: short/long alternating, grid 4 x SMs, one block per SM (60 KB smem)
  { std::vector<double> gaps;
    for (int r = 0; r < reps; r++) { int grid = 4 * sms; probe_alt<<<grid, 128, 60 * 1024>>>(sm, tin, tout, 10000, 100000); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hs.data(), sm, grid * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hin.data(), tin, grid * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hout.data(), tout, grid * 8, cudaMemcpyDeviceToHost));
      std::vector<std::vector<int>> by_sm(1024); for (int b = 0; b < grid; b++) by_sm[hs[b]].push_back(b);
      std::vector<double> g; for (auto& v : by_sm) { std::sort(v.begin(), v.end(), [&](int a, int b) { return hin[a] < hin[b]; });
        for (size_t i = 1; i < v.size(); i++) { long long gap = (long long)hin[v[i]] - (long long)hout[v[i - 1]]; if (gap >= 0 && gap < 1000000) g.push_back(gap * 1e-9); } }
      if (!g.empty()) { std::sort(g.begin(), g.end()); gaps.push_back(g[g.size() / 2]); } }
    emit_clocked("block_refill_latency_s", "s", gaps, "{\"grid\": \"4 x SMs\", \"pattern\": \"alternating 10 us / 100 us blocks\", \"statistic\": \"median gap exit -> next entry on the same SM\"}", mhz); }
  // 4. wave staircase: grid = 2*SMs + r, 50 us blocks, one block per SM (60 KB smem), total time by events
  { Timer t; for (int r : {0, 1, 8, 16, 24, 32, 40, 47}) { std::vector<double> s; int grid = 2 * sms + r;
      for (int k = 0; k < reps; k++) { t.start(); probe<<<grid, 128, 60 * 1024>>>(sm, tin, tout, 50000); double ms = t.stop_ms(); s.push_back(ms * 1e-3); }
      char det[2048]; snprintf(det, sizeof det, "{\"grid\": %d, \"r\": %d, \"block_us\": 50, \"smem_bytes\": 61440, \"expected_waves\": %d}", grid, r, 2 + (r > 0)); char q[64]; snprintf(q, sizeof q, "wave_staircase[r=%d]", r); emit_clocked(q, "s", s, det, mhz); } }
  return 0;
}
