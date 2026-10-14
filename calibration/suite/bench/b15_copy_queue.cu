// B15: the asynchronous copy queue. Group issue time vs pending groups; completion-time distribution;
// cp.async vs TMA contention; TMA 2-D (cuTensorMapEncodeTiled) and mbarrier cost; per-instruction issue cost.
#include "common.cuh"
#include <cuda.h>
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) { unsigned s = (unsigned)__cvta_generic_to_shared(smem); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem)); }
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void wait0() { asm volatile("cp.async.wait_group 0;\n"); }
// 1. issue time of the k-th group, k groups of 4 KB pending, one block per SM
__global__ void issue_k(const char* src, long long* out, int k, int tile) {
  extern __shared__ char smem[]; int per = tile / 16 / blockDim.x;
  long long t0 = clock64();
  for (int g = 0; g < k; g++) { const char* base = src + (size_t)g * tile; for (int c = 0; c < per; c++) { int off = (c * blockDim.x + threadIdx.x) * 16; cp_async16(smem + g * tile + off, base + off); } commit(); if (threadIdx.x == 0) out[blockIdx.x * 64 + g] = clock64() - t0; }
  wait0(); __syncthreads();
}
// 2. completion distribution: trials of one tile, issue -> wait_group 0
__global__ void trial(const char* src, long long* out, int tile, int trials) {
  extern __shared__ char smem[]; int per = tile / 16 / blockDim.x;
  for (int i = 0; i < trials; i++) { const char* base = src + (size_t)(i & 63) * tile; __syncthreads(); long long t0 = clock64();
    for (int c = 0; c < per; c++) { int off = (c * blockDim.x + threadIdx.x) * 16; cp_async16(smem + off, base + off); } commit(); wait0(); __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x * trials + i] = clock64() - t0; }
}
// 3. contention: pipelined streams, cp.async or TMA (bulk) by block parity
__device__ __forceinline__ void stream_cp_body(const char* src, int tile, int iters, float* sink) {
  extern __shared__ char smem[]; int per = tile / 16 / blockDim.x;
  for (int i = 0; i < iters; i++) { const char* base = src + (size_t)((i + blockIdx.x) & 63) * tile; char* dst = smem + (i & 1) * tile;
    for (int c = 0; c < per; c++) { int off = (c * blockDim.x + threadIdx.x) * 16; cp_async16(dst + off, base + off); } commit(); asm volatile("cp.async.wait_group 1;\n"); __syncthreads(); }
  wait0(); __syncthreads(); if (smem[threadIdx.x] == 123) sink[0] = 1;
}
__device__ __forceinline__ void stream_tma_body(const char* src, int tile, int iters, float* sink) {
  extern __shared__ __align__(128) char smem[]; __shared__ unsigned long long bar[2];
  unsigned b0 = (unsigned)__cvta_generic_to_shared(&bar[0]), b1 = (unsigned)__cvta_generic_to_shared(&bar[1]);
  if (threadIdx.x == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b0)); asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b1)); }
  __syncthreads();
  for (int i = 0; i < iters; i++) { const char* base = src + (size_t)((i + blockIdx.x) & 63) * tile; unsigned dst = (unsigned)__cvta_generic_to_shared(smem + (i & 1) * tile); unsigned b = (i & 1) ? b1 : b0;
    if (threadIdx.x == 0) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tile) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(base), "r"(tile), "r"(b) : "memory"); }
    unsigned phase = (i >> 1) & 1, done = 0; while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b), "r"(phase) : "memory"); }
    __syncthreads(); }
  if (smem[threadIdx.x] == 123) sink[0] = 1;
}
__global__ void stream_cp(const char* src, int tile, int iters, float* sink) { stream_cp_body(src, tile, iters, sink); }
__global__ void stream_tma(const char* src, int tile, int iters, float* sink) { stream_tma_body(src, tile, iters, sink); }
__global__ void stream_mixed(const char* src, int tile, int iters, float* sink) { if (blockIdx.x & 1) stream_tma_body(src, tile, iters, sink); else stream_cp_body(src, tile, iters, sink); }
// 4b. mbarrier arrive + try_wait with nothing pending, 256 threads
__global__ void mbar_cost(long long* out, int iters) {
  __shared__ unsigned long long bar; unsigned b = (unsigned)__cvta_generic_to_shared(&bar);
  if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(blockDim.x)); __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); unsigned phase = i & 1, done = 0; while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b), "r"(phase) : "memory"); } }
  if (threadIdx.x == 0) out[blockIdx.x] = (clock64() - t0) / iters;
}
// 5. issue cost: 64 cp.async per thread, no wait inside the timed region
__global__ void issue_cost(const char* src, long long* out) {
  extern __shared__ char smem[]; long long t0 = clock64();
  #pragma unroll 1
  for (int c = 0; c < 64; c++) { int off = (c * blockDim.x + threadIdx.x) * 16; cp_async16(smem + (off & 65535), src + off); }
  long long t1 = clock64(); commit(); wait0(); __syncthreads(); if (threadIdx.x == 0) out[blockIdx.x] = (t1 - t0) / 64;
}
// 4a. TMA 2-D: a tensor map over a 1024 x 1024 fp32 matrix, 64 x 64 tile
__global__ void tma2d(const CUtensorMap* map, long long* out, int iters) {
  extern __shared__ __align__(128) char smem[]; __shared__ unsigned long long bar; unsigned b = (unsigned)__cvta_generic_to_shared(&bar); unsigned dst = (unsigned)__cvta_generic_to_shared(smem);
  if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b)); __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) { int x = (i & 7) * 64, y = ((i >> 3) & 7) * 64;
    if (threadIdx.x == 0) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(64 * 64 * 4) : "memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" :: "r"(dst), "l"(map), "r"(x), "r"(y), "r"(b) : "memory"); }
    unsigned phase = i & 1, done = 0; while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b), "r"(phase) : "memory"); } __syncthreads(); }
  if (threadIdx.x == 0) out[blockIdx.x] = (clock64() - t0) / iters;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); double hz = mhz * 1e6;
  char* src; CK(cudaMalloc(&src, 64 * 65536)); CK(cudaMemset(src, 7, 64 * 65536)); long long* out; CK(cudaMalloc(&out, (size_t)sms * 12000 * 8)); float* sink; CK(cudaMalloc(&sink, 4));
  std::vector<long long> h((size_t)sms * 12000);
  CK(cudaFuncSetAttribute(issue_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(trial, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(issue_cost, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  CK(cudaFuncSetAttribute(stream_cp, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(stream_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(stream_mixed, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(tma2d, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
  // 1. issue time vs pending groups (4 KB groups, up to 22 pending in 88 KB)
  { int tile = 4096, K = 22; std::vector<double> med; std::string table = "[";
    for (int k = 1; k <= K; k++) { std::vector<double> s; for (int r = 0; r < reps; r++) { issue_k<<<sms, 256, K * tile>>>(src, out, k, tile); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), out, (size_t)sms * 64 * 8, cudaMemcpyDeviceToHost)); std::vector<long long> v; for (int b = 0; b < sms; b++) v.push_back(h[b * 64 + k - 1]); std::sort(v.begin(), v.end()); s.push_back((double)v[sms / 2]); }
      std::sort(s.begin(), s.end()); med.push_back(s[s.size() / 2]); table += (k > 1 ? ", " : "") + std::to_string((long long)med.back()); }
    table += "]";
    // the step: first k whose per-group increment exceeds 3x the median increment
    std::vector<double> inc; for (size_t i = 1; i < med.size(); i++) inc.push_back(med[i] - med[i - 1]); std::vector<double> si = inc; std::sort(si.begin(), si.end()); double mi = si[si.size() / 2]; int step = -1; for (size_t i = 0; i < inc.size(); i++) if (inc[i] > 3 * mi + 50) { step = (int)i + 2; break; }
    std::string det = "{\"tile_bytes\": 4096, \"groups\": 22, \"issue_cycles_by_k\": " + table + ", \"step_at_k\": " + std::to_string(step) + ", \"note\": \"cycles from clock64 to issue the k-th group with k-1 pending, one block per SM; -1 = no step within 22\"}";
    emit_clocked("cp_async_groups_in_flight", "groups", std::vector<double>{(double)step}, det.c_str(), mhz); }
  // 2. completion distribution by tile size and residency
  for (int tile : {4096, 8192, 16384, 32768, 49152}) for (int resident : {1, 6}) {
    if (resident == 6 && tile > 16384) continue; int trials = 2000; int smem = (resident == 1) ? 96 * 1024 : 16 * 1024;
    trial<<<sms * resident, 256, smem>>>(src, out, tile, trials); CK(cudaDeviceSynchronize()); size_t n = (size_t)sms * resident * trials; CK(cudaMemcpy(h.data(), out, n * 8, cudaMemcpyDeviceToHost));
    std::vector<long long> v(h.begin(), h.begin() + n); std::sort(v.begin(), v.end());
    double p1 = v[n / 100], p50 = v[n / 2], p99 = v[n * 99 / 100], mx = v[n - 1];
    char det[2048]; snprintf(det, sizeof det, "{\"tile_bytes\": %d, \"resident_blocks_per_sm\": %d, \"trials\": %zu, \"p1_cycles\": %.0f, \"p50_cycles\": %.0f, \"p99_cycles\": %.0f, \"max_cycles\": %.0f, \"p1_s\": %.3e, \"p50_s\": %.3e, \"p99_s\": %.3e, \"max_s\": %.3e}", tile, resident, n, p1, p50, p99, mx, p1 / hz, p50 / hz, p99 / hz, mx / hz);
    char q[80]; snprintf(q, sizeof q, "cp_async_latency_dist[tile=%d,resident=%d]", tile, resident); emit_clocked(q, "cycles", std::vector<double>{p50}, det, mhz); }
  // 3. contention: 96 blocks (2 per SM), 16 KB tiles, pipelined
  { int tile = 16384, iters = 400, grid = 2 * sms; Timer t; auto run = [&](auto kern, const char* name) { std::vector<double> s; kern<<<grid, 256, 2 * tile>>>(src, tile, iters, sink); CK(cudaDeviceSynchronize());
      for (int r = 0; r < reps; r++) { t.start(); kern<<<grid, 256, 2 * tile>>>(src, tile, iters, sink); double ms = t.stop_ms(); s.push_back((double)tile * iters * grid / (ms * 1e-3)); }
      char det[2048]; snprintf(det, sizeof det, "{\"tile_bytes\": %d, \"blocks\": %d, \"iters\": %d, \"path\": \"%s\"}", tile, grid, iters, name); char q[64]; snprintf(q, sizeof q, "copy_throughput[%s]", name); emit_clocked(q, "byte/s", s, det, mhz); };
    run(stream_cp, "cp_async"); run(stream_tma, "tma_bulk"); run(stream_mixed, "mixed_half_half"); }
  // 4. mbarrier cost; TMA 2-D
  { std::vector<double> s; for (int r = 0; r < reps; r++) { mbar_cost<<<sms, 256>>>(out, 1000); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), out, sms * 8, cudaMemcpyDeviceToHost)); std::vector<long long> v(h.begin(), h.begin() + sms); std::sort(v.begin(), v.end()); s.push_back((double)v[sms / 2] / hz); }
    emit_clocked("mbarrier_s", "s", s, "{\"threads\": 256, \"pattern\": \"arrive + try_wait.parity, nothing pending, per round trip\"}", mhz); }
  { // tensor map through the driver API
    CUtensorMap map; float* mat; CK(cudaMalloc(&mat, 1024 * 1024 * 4)); cuuint64_t dims[2] = {1024, 1024}; cuuint64_t strides[1] = {1024 * 4}; cuuint32_t box[2] = {64, 64}; cuuint32_t estr[2] = {1, 1};
    CUresult cr = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, mat, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) { printf("{\"quantity\": \"tma_2d_latency_s\", \"unit\": \"s\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"cuTensorMapEncodeTiled failed: %d\"}}\n", (int)cr); }
    else { CUtensorMap* dmap; CK(cudaMalloc(&dmap, sizeof map)); CK(cudaMemcpy(dmap, &map, sizeof map, cudaMemcpyHostToDevice));
      tma2d<<<sms, 128, 64 * 64 * 4>>>(dmap, out, 100); cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) printf("{\"quantity\": \"tma_2d_latency_s\", \"unit\": \"s\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"%s\"}}\n", cudaGetErrorString(e));
      else { std::vector<double> s; for (int r = 0; r < reps; r++) { tma2d<<<sms, 128, 64 * 64 * 4>>>(dmap, out, 100); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), out, sms * 8, cudaMemcpyDeviceToHost)); std::vector<long long> v(h.begin(), h.begin() + sms); std::sort(v.begin(), v.end()); s.push_back((double)v[sms / 2] / hz); }
        emit_clocked("tma_2d_latency_s", "s", s, "{\"tile\": \"64 x 64 fp32 (16 KB)\", \"api\": \"cp.async.bulk.tensor.2d + mbarrier, cuTensorMapEncodeTiled\"}", mhz); } } }
  // 5. issue cost per cp.async instruction
  { std::vector<double> s; for (int r = 0; r < reps; r++) { issue_cost<<<sms, 256, 64 * 1024>>>(src, out); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), out, sms * 8, cudaMemcpyDeviceToHost)); std::vector<long long> v(h.begin(), h.begin() + sms); std::sort(v.begin(), v.end()); s.push_back((double)v[sms / 2]); }
    emit_clocked("cp_async_issue_cycles_per_16B", "cycles", s, "{\"threads\": 256, \"instructions_per_thread\": 64, \"note\": \"clock64 delta over 64 cp.async.cg 16 B issues per thread, divided by 64 (includes address arithmetic)\"}", mhz); }
  return 0;
}
