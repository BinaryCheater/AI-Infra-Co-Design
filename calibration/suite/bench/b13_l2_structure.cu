// B13: L2 structure by pointer chase. 1. fill granularity (bytes moved per request vs stride);
// 2. set conflicts: n addresses spaced by a power-of-two stride s (ways and set bits); 3. slice bits:
// latency of a two-address chase for each flipped address bit, and aggregate bandwidth of 48 blocks
// at region offsets of 2^bit; 4. replacement: hit rate vs footprint for cyclic vs random order.
// The inferred structure AND the tables it was inferred from go into detail, so the inference can be checked.
#include "common.cuh"
#include <random>
__global__ void chase(const unsigned* __restrict__ next, unsigned start, int steps, unsigned* out, long long* cyc) {
  unsigned p = start; long long t0 = clock64();
  for (int i = 0; i < steps; i++) { unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(next + p) : "memory"); p = v; }   // .cg: L2 only, so the L1 never serves the chase
  long long t1 = clock64(); out[0] = p; if (cyc) cyc[0] = t1 - t0;
}
__global__ void strided_touch(const float* __restrict__ a, size_t n_words, size_t stride_words, float* out) {
  float acc = 0; size_t nreq = n_words / stride_words;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nreq; i += (size_t)gridDim.x * blockDim.x) acc += a[i * stride_words];
  if (acc == 123456.789f) out[0] = acc;
}
__global__ void region_stream(const float4* __restrict__ a, size_t words4_per_block, size_t block_stride4, float* out) {
  const float4* base = a + blockIdx.x * block_stride4; float4 acc = make_float4(0, 0, 0, 0);
  for (size_t i = threadIdx.x; i < words4_per_block; i += blockDim.x) { float4 v = base[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
static double chase_ns(unsigned* d, unsigned* out, long long* cyc, int steps, double hz, unsigned start = 0, unsigned* end = nullptr) {
  chase<<<1, 1>>>(d, start, steps, out, cyc); CK(cudaDeviceSynchronize()); long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); if (end) CK(cudaMemcpy(end, out, 4, cudaMemcpyDeviceToHost)); return (double)c / steps / hz * 1e9;
}
static void build_cycle(std::vector<unsigned>& h, const std::vector<size_t>& addrs_words) {
  // h[addr] = next addr; addrs visited in the given order, cyclic
  size_t n = addrs_words.size(); for (size_t i = 0; i < n; i++) h[addrs_words[i]] = (unsigned)addrs_words[(i + 1) % n];
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 3); int sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6;
  const size_t maxb = 1ull << 30; unsigned* d; unsigned* out; long long* cyc; CK(cudaMalloc(&d, maxb)); CK(cudaMalloc(&out, 4)); CK(cudaMalloc(&cyc, 8));
  std::vector<unsigned> h(maxb / 4, 0); std::mt19937_64 rng(7);
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); int l2_decl = p.l2CacheSize;
  // ---- reference latencies: 4 MiB random cycle (L2) and 1 GiB (DRAM), 64 B lines
  auto random_cycle_ns = [&](size_t bytes, size_t line, int steps) { size_t lines = bytes / line, stride = line / 4; std::vector<size_t> addr(lines); for (size_t i = 0; i < lines; i++) addr[i] = i * stride; std::shuffle(addr.begin(), addr.end(), rng); build_cycle(h, addr); CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    // warm-up covers the whole cycle when it fits in L2 (so the timed chase hits), else a prefix; the timed chase continues from where the warm-up stopped, so a cycle larger than L2 keeps missing
    unsigned endp = 0; chase_ns(d, out, cyc, (int)std::min<size_t>(lines, 1 << 18), hz, 0, &endp); return chase_ns(d, out, cyc, steps, hz, endp); };
  double lat_l2 = random_cycle_ns(4 << 20, 64, 20000), lat_dram = random_cycle_ns(1 << 30, 64, 20000);
  // ---- 1. fill granularity: all SMs touch one word every `stride` bytes over 1 GiB; bytes-moved per request from B4's DRAM rate
  { float* out2; CK(cudaMalloc(&out2, 4)); Timer t; std::string table = "["; double plateau = 0; int first_plateau = 0;
    std::vector<double> per_req;
    for (int stride : {4, 32, 64, 128, 256, 512}) { size_t n_words = maxb / 4, sw = stride / 4; size_t nreq = n_words / sw; std::vector<double> s;
      strided_touch<<<sms * 8, 256>>>((const float*)d, n_words, sw, out2); CK(cudaDeviceSynchronize());
      for (int r = 0; r < reps; r++) { t.start(); strided_touch<<<sms * 8, 256>>>((const float*)d, n_words, sw, out2); double ms = t.stop_ms(); s.push_back(ms * 1e-3); }
      std::sort(s.begin(), s.end()); double sec = s[s.size() / 2]; double bytes_per_req = 2.3536e11 * sec / nreq;   // at the measured DRAM read rate (B4)
      per_req.push_back(bytes_per_req); char b[96]; snprintf(b, sizeof b, "%s[%d, %.4g, %.1f]", stride > 4 ? ", " : "", stride, sec, bytes_per_req); table += b; }
    table += "]";
    // the fill granularity: the stride where bytes-per-request first exceeds 0.8 x the next stride's value (stops growing proportionally)
    int strides[] = {4, 32, 64, 128, 256, 512}; int sector = 32; for (size_t i = 1; i + 1 < per_req.size(); i++) if (per_req[i + 1] < 1.5 * per_req[i]) { sector = strides[i]; break; }
    std::string det = "{\"table_stride_seconds_bytesPerRequest\": " + table + ", \"inferred_sector_bytes\": " + std::to_string(sector) + ", \"method\": \"time at stride s over 1 GiB times the B4 DRAM rate, divided by requests; the stride where bytes per request stops growing is the fill granularity\"}";
    emit_clocked("l2_sector_bytes", "bytes", std::vector<double>{(double)sector}, det.c_str(), mhz); cudaFree(out2); }
  // ---- 2. set conflicts: n addresses spaced by stride s, latency; ways = smallest n with a jump at the largest s; set stride = smallest s that jumps
  { std::string grid = "["; int ways = -1; size_t set_stride = 0; double mid = (lat_l2 + lat_dram) / 2; bool first = true;
    std::vector<size_t> strides = {4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608};
    for (size_t s : strides) { grid += first ? "" : ", "; first = false; grid += "{\"stride\": " + std::to_string(s) + ", \"latency_ns_by_n\": [";
      int jump_n = -1;
      for (int n = 2; n <= 64; n *= 2) { if ((size_t)n * s > maxb) { grid += ", null"; continue; } std::vector<size_t> addr; for (int i = 0; i < n; i++) addr.push_back((size_t)i * s / 4); build_cycle(h, addr); CK(cudaMemcpy(d, h.data(), (size_t)n * s, cudaMemcpyHostToDevice));
        chase_ns(d, out, cyc, 20000, hz); double ns = chase_ns(d, out, cyc, 50000, hz); char b[32]; snprintf(b, sizeof b, "%s%.1f", n > 2 ? ", " : "", ns); grid += b;
        if (jump_n < 0 && ns > mid) jump_n = n; }
      grid += "], \"first_n_at_dram_latency\": " + std::to_string(jump_n) + "}";
      if (jump_n > 0 && set_stride == 0) set_stride = s;
      if (s == strides.back() && jump_n > 0) ways = jump_n / 2; }
    grid += "]";
    long long sets = set_stride ? (long long)(set_stride / 128) : -1;      // set stride / line: the number of sets per slice
    std::string det = "{\"grid\": " + grid + ", \"lat_l2_ns\": " + std::to_string(lat_l2) + ", \"lat_dram_ns\": " + std::to_string(lat_dram) + ", \"inferred_ways\": " + std::to_string(ways) + ", \"inferred_set_stride_bytes\": " + std::to_string(set_stride) + ", \"inferred_sets_if_128B_lines\": " + std::to_string(sets) + ", \"inferred_capacity_bytes\": " + std::to_string(ways > 0 && sets > 0 ? ways * sets * 128 : -1) + ", \"declared_l2_bytes\": " + std::to_string(l2_decl) + ", \"method\": \"n addresses at stride s chased cyclically; at the largest stride the first n whose latency crosses the L2/DRAM midpoint is the associativity (reported as n/2..n); the smallest stride that conflicts gives the set stride\"}";
    emit_clocked("l2_set_conflicts", "ways", std::vector<double>{(double)ways}, det.c_str(), mhz); }
  // ---- 3. slice bits: two-address chase, base vs base ^ (1 << bit); and 48-block aggregate bandwidth at region offsets 2^bit
  { std::string lat = "["; std::vector<double> lats; double base_ns = 0;
    for (int bit = 7; bit <= 29; bit++) { std::vector<size_t> addr = {0, (size_t)(1ull << bit) / 4}; build_cycle(h, addr); CK(cudaMemcpy(d, h.data(), (1ull << bit) + 64, cudaMemcpyHostToDevice)); chase_ns(d, out, cyc, 20000, hz); double ns = chase_ns(d, out, cyc, 100000, hz); lats.push_back(ns); char b[48]; snprintf(b, sizeof b, "%s[%d, %.2f]", bit > 7 ? ", " : "", bit, ns); lat += b; if (bit == 7) base_ns = ns; }
    lat += "]";
    double mn = *std::min_element(lats.begin(), lats.end()), mx = *std::max_element(lats.begin(), lats.end()); std::string bits = "["; bool f = true;
    for (size_t i = 0; i < lats.size(); i++) if (lats[i] - mn > 0.5 * (mx - mn) && mx - mn > 5) { bits += (f ? "" : ", ") + std::to_string(7 + i); f = false; } bits += "]";
    float* out2; CK(cudaMalloc(&out2, 4)); Timer t; std::string bw = "["; std::vector<double> bws;
    for (int bit = 12; bit <= 26; bit++) { size_t per_block4 = (64 << 10) / 16, stride4 = (1ull << bit) / 16; if (stride4 * sms * 16 > maxb) break; std::vector<double> s; region_stream<<<sms, 256>>>((const float4*)d, per_block4, stride4, out2); CK(cudaDeviceSynchronize());
      for (int r = 0; r < reps; r++) { t.start(); for (int k = 0; k < 20; k++) region_stream<<<sms, 256>>>((const float4*)d, per_block4, stride4, out2); double ms = t.stop_ms(); s.push_back(20.0 * sms * per_block4 * 16 / (ms * 1e-3)); }
      std::sort(s.begin(), s.end()); bws.push_back(s[s.size() / 2]); char b[48]; snprintf(b, sizeof b, "%s[%d, %.4g]", bit > 12 ? ", " : "", bit, bws.back()); bw += b; }
    bw += "]";
    std::string det = "{\"latency_ns_by_flipped_bit\": " + lat + ", \"near_far_ns\": [" + std::to_string(mn) + ", " + std::to_string(mx) + "], \"inferred_slice_bits\": " + bits + ", \"aggregate_bw_by_region_bit\": " + bw + ", \"method\": \"two-address chase between base and base XOR 2^bit; bits whose flip raises latency by more than half the spread are slice-select bits when the spread exceeds 5 ns. Aggregate: 48 blocks each streaming 64 KB at a 2^bit offset; a collapse at some bit means those blocks share a slice\"}";
    emit_clocked("l2_slice_hash", "ns", std::vector<double>{mx - mn}, det.c_str(), mhz); cudaFree(out2); }
  // ---- 4. replacement: hit rate vs footprint, sequential-cyclic vs random-cyclic, from mean latency
  { std::string curve = "["; bool f = true; double seq_drop = 0, rnd_drop = 0;
    for (size_t mb : {8, 12, 16, 20, 24, 32, 48}) { size_t bytes = mb << 20, lines = bytes / 128; double hr[2];
      for (int order = 0; order < 2; order++) { std::vector<size_t> addr(lines); for (size_t i = 0; i < lines; i++) addr[i] = i * 32; if (order) std::shuffle(addr.begin(), addr.end(), rng); build_cycle(h, addr); CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
        chase_ns(d, out, cyc, (int)lines, hz); double ns = chase_ns(d, out, cyc, 100000, hz); hr[order] = std::max(0.0, std::min(1.0, (lat_dram - ns) / (lat_dram - lat_l2))); }
      char b[80]; snprintf(b, sizeof b, "%s[%zu, %.3f, %.3f]", f ? "" : ", ", mb, hr[0], hr[1]); curve += b; f = false; if (mb == 24) { seq_drop = hr[0]; rnd_drop = hr[1]; } }
    curve += "]";
    const char* cls = (seq_drop < 0.1 && rnd_drop < 0.1) ? "lru-like (both orders miss past capacity)" : (seq_drop < 0.1 && rnd_drop > 0.2) ? "protected/random-like (random order keeps hits past capacity)" : (seq_drop > 0.2) ? "other (sequential cyclic keeps hits past capacity)" : "unclassified";
    std::string det = "{\"hit_rate_by_footprint_mb_seq_rand\": " + curve + ", \"classification\": \"" + cls + "\", \"method\": \"hit rate = (lat_dram - mean latency) / (lat_dram - lat_l2) for a cyclic chase over the footprint, sequential vs random order of the same lines\"}";
    emit_clocked("l2_replacement", "hit_rate_at_24MiB_seq", std::vector<double>{seq_drop}, det.c_str(), mhz); }
  return 0;
}
