// B18: LPDDR5X behaviour. 1. bandwidth vs power-of-two stride (interleave bits, channel count);
// 2. sequential vs random 128 B vs random 4 KB chunks (page locality); 3. read/write mix (turnaround);
// 4. GPU read bandwidth with and without a 10-thread host memcpy load; 5. 60 s drift.
#include "common.cuh"
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
__global__ void strided_warps(const float4* __restrict__ a, size_t n4, size_t stride4, size_t nreq, float* out) {
  // each warp reads 128 B (8 float4 by 8 lanes... use 32 lanes x 16 B = 512 B? no: 8 lanes x 16 B = 128 B) at request i * stride
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32; int nw = gridDim.x * blockDim.x / 32; float4 acc = make_float4(0, 0, 0, 0);
  for (size_t i = warp; i < nreq; i += nw) { if (lane < 8) { float4 v = a[(i * stride4 + lane) % n4]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; } }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
__global__ void masked_random(const float4* __restrict__ a, size_t region, unsigned long long mask, size_t nreq, float* out) {
  // request i reads 128 B at a hashed address with the masked bits cleared; 8 lanes per line (16 B each), 4 lines per warp instruction
  float4 acc = make_float4(0, 0, 0, 0); size_t lines = region >> 7;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nreq * 8; i += (size_t)gridDim.x * blockDim.x) {
    unsigned long long r = (i >> 3) * 0x9E3779B97F4A7C15ull; r ^= r >> 29; unsigned long long addr = ((r % lines) << 7) & ~mask; addr += (i & 7) * 16;
    float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"((const char*)a + addr)); acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
__global__ void gather_chunks(const float4* __restrict__ a, const unsigned* __restrict__ order, size_t nchunks, size_t chunk4, float* out) {
  float4 acc = make_float4(0, 0, 0, 0); int warps = gridDim.x * blockDim.x / 32, warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  for (size_t c = warp; c < nchunks; c += warps) { const float4* base = a + (size_t)order[c] * chunk4; for (size_t i = lane; i < chunk4; i += 32) { float4 v = base[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; } }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
__global__ void mixed_rw(const float4* __restrict__ a, float4* __restrict__ b, size_t n4, int read_per_8, float* out) {
  float4 acc = make_float4(0, 0, 0, 0); for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) { if ((int)((i / 256) % 8) < read_per_8) { float4 v = a[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; } else b[i] = acc; }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
__global__ void read_kernel(const float4* __restrict__ a, size_t n, float* out) { float4 acc = make_float4(0, 0, 0, 0); for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = a[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; } if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1; }
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 3), sms = sm_count(); double mhz = probe_clock_mhz();
  size_t bytes = 1ull << 30, n4 = bytes / 16; float4 *a, *b; CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes)); CK(cudaMemset(a, 0, bytes)); float* out; CK(cudaMalloc(&out, 4)); Timer t; int grid = sms * 8, threads = 256;
  // 1. interleave: random 128 B requests over 1 GiB with one address bit forced to zero; an interleave (channel-select) bit halves the
  //    bandwidth when forced; forcing all found bits together gives the single-channel floor, and top/floor cross-checks the count
  { size_t nreq = 1u << 22; auto bw_masked = [&](unsigned long long mask) { std::vector<double> v; masked_random<<<grid, threads>>>(a, bytes, mask, nreq, out); CK(cudaDeviceSynchronize());
      for (int r = 0; r < reps; r++) { t.start(); masked_random<<<grid, threads>>>(a, bytes, mask, nreq, out); double ms = t.stop_ms(); v.push_back(128.0 * nreq / (ms * 1e-3)); } std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    double top = bw_masked(0); std::string table = "["; std::vector<int> bits; bool f = true; unsigned long long all = 0;
    for (int bit = 7; bit <= 29; bit++) { double bw = bw_masked(1ull << bit); char bb[64]; snprintf(bb, sizeof bb, "%s[%d, %.4g]", f ? "" : ", ", bit, bw); table += bb; f = false; if (bw < 0.7 * top) { bits.push_back(bit); all |= 1ull << bit; } }
    table += "]"; double floor_bw = bits.empty() ? top : bw_masked(all); std::string bits_j = "["; for (size_t k = 0; k < bits.size(); k++) bits_j += (k ? ", " : "") + std::to_string(bits[k]); bits_j += "]";
    int channels_from_bits = 1 << bits.size(); double ratio = top / floor_bw;
    std::string det = "{\"top_bytes_per_s\": " + std::to_string(top) + ", \"table_forcedBit_bytesPerS\": " + table + ", \"inferred_interleave_bits\": " + bits_j + ", \"floor_bytes_per_s_all_bits_forced\": " + std::to_string(floor_bw) + ", \"top_over_floor\": " + std::to_string(ratio) + ", \"inferred_channels\": " + std::to_string(channels_from_bits) + ", \"method\": \"random 128 B requests (ld.cg, 4M per pass) over 1 GiB with one address bit forced to 0: a bit whose forcing drops the bandwidth below 70 % of the unmasked value selects a channel (or a bank group); channels = 2^(bits found), cross-checked by the unmasked / all-forced ratio. XOR-hashed channel functions can hide a bit from single-bit forcing; the table shows every bit\"}";
    emit_clocked("dram_interleave", "channels", std::vector<double>{(double)channels_from_bits}, det.c_str(), mhz); }
  // 2. page locality: sequential 128 B, random 128 B, random 4 KB chunks over 1 GiB
  { unsigned* order; size_t nc128 = bytes / 128, nc4k = bytes / 4096; CK(cudaMalloc(&order, nc128 * 4)); std::vector<unsigned> h(nc128); std::mt19937 rng(3);
    auto run = [&](const char* name, size_t chunk, bool random) { size_t nchunks = bytes / chunk; for (size_t i = 0; i < nchunks; i++) h[i] = i; if (random) std::shuffle(h.begin(), h.begin() + nchunks, rng); CK(cudaMemcpy(order, h.data(), nchunks * 4, cudaMemcpyHostToDevice));
      std::vector<double> v; gather_chunks<<<grid, threads>>>(a, order, nchunks, chunk / 16, out); CK(cudaDeviceSynchronize()); for (int r = 0; r < reps; r++) { t.start(); gather_chunks<<<grid, threads>>>(a, order, nchunks, chunk / 16, out); double ms = t.stop_ms(); v.push_back((double)bytes / (ms * 1e-3)); }
      char det[2048]; snprintf(det, sizeof det, "{\"chunk_bytes\": %zu, \"order\": \"%s\"}", chunk, random ? "random" : "sequential"); char q[64]; snprintf(q, sizeof q, "dram_locality[%s]", name); emit_clocked(q, "byte/s", v, det, mhz); std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    double seq = run("seq_128B", 128, false), r128 = run("rand_128B", 128, true), r512 = run("rand_512B", 512, true), r4k = run("rand_4KB", 4096, true), r16k = run("rand_16KB", 16384, true);
    size_t page = r128 >= 0.9 * seq ? 128 : r512 >= 0.9 * seq ? 512 : r4k >= 0.9 * seq ? 4096 : r16k >= 0.9 * seq ? 16384 : 65536;
    char det[2048]; snprintf(det, sizeof det, "{\"seq_random_ratio_128B\": %.3f, \"ratios_512B_4KB_16KB\": [%.3f, %.3f, %.3f], \"inferred_effective_page_bytes\": %zu, \"method\": \"random chunk size at which bandwidth reaches 90 %% of sequential\"}", seq / r128, r512 / seq, r4k / seq, r16k / seq, page);
    emit_clocked("dram_page_bytes_effective", "bytes", std::vector<double>{(double)page}, det, mhz); (void)nc4k; }
  // 3. read/write mix
  { std::string table = "["; bool f = true; std::vector<double> mix; double pure_r = 0, pure_w = 0, worst_dip = 0;
    for (int rp : {8, 6, 4, 2, 0}) { std::vector<double> v; mixed_rw<<<grid, threads>>>(a, b, n4, rp, out); CK(cudaDeviceSynchronize()); for (int r = 0; r < reps; r++) { t.start(); mixed_rw<<<grid, threads>>>(a, b, n4, rp, out); double ms = t.stop_ms(); v.push_back((double)bytes / (ms * 1e-3)); } std::sort(v.begin(), v.end()); mix.push_back(v[v.size() / 2]); if (rp == 8) pure_r = mix.back(); if (rp == 0) pure_w = mix.back();
      char bb[48]; snprintf(bb, sizeof bb, "%s[%.2f, %.4g]", f ? "" : ", ", rp / 8.0, mix.back()); table += bb; f = false; }
    table += "]"; int rps[] = {8, 6, 4, 2, 0}; for (int i = 1; i < 4; i++) { double frac = rps[i] / 8.0; double lin = frac * pure_r + (1 - frac) * pure_w; worst_dip = std::max(worst_dip, 1 - mix[i] / lin); }
    std::string det = "{\"table_readFraction_bytesPerS\": " + table + ", \"worst_dip_below_linear\": " + std::to_string(worst_dip) + ", \"method\": \"4 KB runs of reads and writes interleaved in the stated fraction over 1 GiB; dip below the linear interpolation of the pure cases is the turnaround cost\"}";
    emit_clocked("dram_rw_turnaround_penalty", "fraction", std::vector<double>{worst_dip}, det.c_str(), mhz); }
  // 4. CPU contention: host memcpy loop over 4 GiB on 10 threads during the GPU stream
  { std::vector<double> alone, loaded; for (int r = 0; r < reps; r++) { t.start(); for (int p = 0; p < 4; p++) read_kernel<<<grid, threads>>>(a, n4, out); double ms = t.stop_ms(); alone.push_back(4.0 * bytes / (ms * 1e-3)); }
    size_t hb = 4ull << 30; std::vector<char*> src(10), dst(10); for (int i = 0; i < 10; i++) { src[i] = (char*)malloc(hb / 10); dst[i] = (char*)malloc(hb / 10); memset(src[i], 1, hb / 10); memset(dst[i], 2, hb / 10); }
    std::atomic<bool> stop(false); std::vector<std::thread> th; for (int i = 0; i < 10; i++) th.emplace_back([&, i]() { while (!stop) memcpy(dst[i], src[i], hb / 10); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); for (int r = 0; r < reps; r++) { t.start(); for (int p = 0; p < 4; p++) read_kernel<<<grid, threads>>>(a, n4, out); double ms = t.stop_ms(); loaded.push_back(4.0 * bytes / (ms * 1e-3)); }
    stop = true; for (auto& x : th) x.join(); for (int i = 0; i < 10; i++) { free(src[i]); free(dst[i]); }
    std::sort(alone.begin(), alone.end()); char det[2048]; snprintf(det, sizeof det, "{\"alone_bytes_per_s\": %.4g, \"host_load\": \"10 threads memcpy over 4 GiB\", \"method\": \"GPU 1 GiB streaming read with and without the host load\"}", alone[alone.size() / 2]);
    emit_clocked("dram_bytes_per_s_under_cpu_load", "byte/s", loaded, det, mhz); }
  // 5. drift over 60 s
  { std::vector<double> series; auto t0 = std::chrono::steady_clock::now(); while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 60) { t.start(); for (int p = 0; p < 4; p++) read_kernel<<<grid, threads>>>(a, n4, out); double ms = t.stop_ms(); series.push_back(4.0 * bytes / (ms * 1e-3)); }
    std::vector<double> s = series; std::sort(s.begin(), s.end()); std::string det = "{\"samples\": " + std::to_string(series.size()) + ", \"min_bytes_per_s\": " + std::to_string(s.front()) + ", \"max_bytes_per_s\": " + std::to_string(s.back()) + ", \"drift\": " + std::to_string((s.back() - s.front()) / s.back()) + ", \"series_subsampled\": " + vec_json(std::vector<double>(series.begin(), series.begin() + std::min<size_t>(series.size(), 60)), "%.4g") + ", \"method\": \"1 GiB x 4 read passes repeated for 60 s; drift = (max - min) / max\"}";
    emit_clocked("dram_drift_60s", "fraction", std::vector<double>{(s.back() - s.front()) / s.back()}, det.c_str(), mhz); }
  return 0;
}
