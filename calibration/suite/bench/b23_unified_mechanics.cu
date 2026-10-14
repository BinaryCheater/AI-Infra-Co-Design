// B23: unified-memory mechanics. 1. first-touch cost per page size; 2. prefetch; 3. host-resident access
// (preferred location CPU); 4. managed vs device with the CPU idle; 5. allocation limits.
#include "common.cuh"
__global__ void touch_pages(const char* p, size_t pages, size_t page, unsigned long long* cyc, float* out) { unsigned long long t0 = gtimer(); float acc = 0; for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < pages; i += (size_t)gridDim.x * blockDim.x) acc += p[i * page]; unsigned long long t1 = gtimer(); if (acc == 123.f) out[0] = 1; if (threadIdx.x == 0 && blockIdx.x == 0) cyc[0] = t1 - t0; }
__global__ void read_kernel(const float4* __restrict__ a, size_t n, float* out) { float4 acc = make_float4(0, 0, 0, 0); for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = a[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; } if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1; }
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 3), sms = sm_count(); double mhz = probe_clock_mhz(); size_t bytes = 1ull << 30, n4 = bytes / 16; float* out; CK(cudaMalloc(&out, 4)); unsigned long long* cyc; CK(cudaMalloc(&cyc, 8)); Timer t; int grid = sms * 8, threads = 256;
  auto bw = [&](const float4* p, int passes) { t.start(); for (int i = 0; i < passes; i++) read_kernel<<<grid, threads>>>(p, n4, out); double ms = t.stop_ms(); return (double)bytes * passes / (ms * 1e-3); };
  // 1. first touch by page size (CPU touched first), time per page from a one-byte-per-page kernel, vs steady state
  for (size_t page : {4096ull, 65536ull, 2097152ull}) { std::vector<double> first, steady; for (int r = 0; r < reps; r++) { char* m; CK(cudaMallocManaged(&m, bytes)); for (size_t i = 0; i < bytes; i += page) m[i] = 1; size_t pages = bytes / page;
      touch_pages<<<grid, threads>>>(m, pages, page, cyc, out); CK(cudaDeviceSynchronize()); unsigned long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); first.push_back(c * 1e-9 / pages * (grid * threads));   // per page, per thread-parallel touch
      touch_pages<<<grid, threads>>>(m, pages, page, cyc, out); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); steady.push_back(c * 1e-9 / pages * (grid * threads)); cudaFree(m); }
    std::sort(steady.begin(), steady.end()); char det[2048]; snprintf(det, sizeof det, "{\"page_bytes\": %zu, \"steady_touch_s_per_page_median\": %.3e, \"method\": \"managed 1 GiB touched by the CPU one byte per page; GPU kernel touches one byte per page (wall time x threads / pages = per-touch cost); second pass = steady\"}", page, steady[steady.size() / 2]);
    char q[48]; snprintf(q, sizeof q, "um_first_touch_s[page=%zu]", page); emit_clocked(q, "s", first, det, mhz); }
  // 2. prefetch
  { std::vector<double> s; for (int r = 0; r < reps; r++) { float4* m; CK(cudaMallocManaged(&m, bytes)); for (size_t i = 0; i < n4; i += 256) m[i] = make_float4(1, 1, 1, 1); { cudaMemLocation loc; loc.type = cudaMemLocationTypeDevice; loc.id = 0; CK(cudaMemPrefetchAsync(m, bytes, loc, 0, 0)); } CK(cudaDeviceSynchronize()); bw(m, 1); s.push_back(bw(m, 2)); cudaFree(m); }
    emit_clocked("um_prefetched_bytes_per_s", "byte/s", s, "{\"method\": \"managed buffer touched by the CPU, cudaMemPrefetchAsync to the GPU, then streamed\"}", mhz); }
  // 3. host-resident: preferred location CPU, accessed by GPU
  { std::vector<double> s; for (int r = 0; r < reps; r++) { float4* m; CK(cudaMallocManaged(&m, bytes)); for (size_t i = 0; i < n4; i += 256) m[i] = make_float4(1, 1, 1, 1);
      cudaMemLocation host; host.type = cudaMemLocationTypeHost; host.id = 0; cudaMemLocation dev; dev.type = cudaMemLocationTypeDevice; dev.id = 0; cudaError_t e1 = cudaMemAdvise(m, bytes, cudaMemAdviseSetPreferredLocation, host), e2 = cudaMemAdvise(m, bytes, cudaMemAdviseSetAccessedBy, dev); (void)e1; (void)e2; bw(m, 1); s.push_back(bw(m, 2)); cudaFree(m); }
    emit_clocked("um_host_resident_bytes_per_s", "byte/s", s, "{\"method\": \"managed buffer with preferred location CPU and accessed-by GPU, streamed from the GPU\"}", mhz); }
  // 4. managed vs device, CPU idle
  { std::vector<double> dev, man; float4* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes)); bw(d, 1); for (int r = 0; r < reps; r++) dev.push_back(bw(d, 2)); cudaFree(d);
    float4* m; CK(cudaMallocManaged(&m, bytes)); read_kernel<<<grid, threads>>>(m, n4, out); CK(cudaDeviceSynchronize()); bw(m, 1); for (int r = 0; r < reps; r++) man.push_back(bw(m, 2)); cudaFree(m);
    std::sort(dev.begin(), dev.end()); std::sort(man.begin(), man.end()); char det[2048]; snprintf(det, sizeof det, "{\"device_bytes_per_s\": %.4g, \"managed_bytes_per_s\": %.4g, \"method\": \"device vs GPU-first managed streaming with the CPU idle; the ratio is the managed-page overhead itself\"}", dev[dev.size() / 2], man[man.size() / 2]);
    emit_clocked("um_translation_overhead", "ratio", std::vector<double>{dev[dev.size() / 2] / man[man.size() / 2]}, det, mhz); }
  // 5. limits: one allocation of (free - 8 GiB) each for cudaMalloc and cudaMallocManaged, freed untouched. A binary search up to
  //    1.5x the total OOM-killed host services on the Spark (unified memory: cudaMalloc commits host RAM), so oversubscription is not probed.
  { size_t total = 0, freeb = 0; CK(cudaMemGetInfo(&freeb, &total)); size_t want = freeb > (16ull << 30) ? freeb - (8ull << 30) : freeb / 2; void* p = nullptr; bool dev_ok = cudaMalloc(&p, want) == cudaSuccess; if (dev_ok) cudaFree(p); else cudaGetLastError();
    void* m = nullptr; bool man_ok = cudaMallocManaged(&m, want) == cudaSuccess; if (man_ok) cudaFree(m); else cudaGetLastError();
    char det[512]; snprintf(det, sizeof det, "{\"total_bytes\": %zu, \"free_bytes\": %zu, \"tried_bytes\": %zu, \"cudaMalloc_ok\": %s, \"cudaMallocManaged_ok\": %s, \"oversubscription_probed\": false, \"method\": \"one allocation of free - 8 GiB per API, freed untouched; larger probes are unsafe on unified memory (a binary search OOM-killed host services)\"}", total, freeb, want, dev_ok ? "true" : "false", man_ok ? "true" : "false");
    emit_clocked("um_max_alloc_bytes", "bytes", std::vector<double>{(double)(dev_ok ? want : 0)}, det, mhz); }
  return 0;
}
