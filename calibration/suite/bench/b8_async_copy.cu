// B8: asynchronous global->shared copies. cp.async (16 B per thread, Ampere+): issue-to-completion
// latency for one tile (wait_group 0 after each tile) and pipelined throughput (two tiles in flight,
// wait_group 1), tile sizes 4 KB .. 64 KB. cp.async.bulk (TMA, Hopper+/Blackwell): a 1-D bulk copy
// with an mbarrier; recorded as tma_supported = true only if it compiles, runs and copies correctly.
#include "common.cuh"
#include <cuda/barrier>
#include <cooperative_groups.h>
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// one tile at a time: latency = issue -> wait_group 0 complete
__global__ void cpasync_latency(const char* __restrict__ src, int tile_bytes, int iters, long long* out) {
  extern __shared__ char smem[];
  int per_thread = tile_bytes / 16 / blockDim.x;           // 16 B chunks per thread
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    const char* base = src + (size_t)(i & 63) * tile_bytes;
    for (int c = 0; c < per_thread; c++) { int off = (c * blockDim.x + threadIdx.x) * 16; cp_async16(smem + off, base + off); }
    cp_async_commit(); cp_async_wait<0>(); __syncthreads();
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}
// two tiles in flight: throughput
__global__ void cpasync_pipelined(const char* __restrict__ src, int tile_bytes, int iters, float* sink) {
  extern __shared__ char smem[];
  int per_thread = tile_bytes / 16 / blockDim.x;
  for (int i = 0; i < iters; i++) {
    const char* base = src + (size_t)(i & 63) * tile_bytes; char* dst = smem + (i & 1) * tile_bytes;
    for (int c = 0; c < per_thread; c++) { int off = (c * blockDim.x + threadIdx.x) * 16; cp_async16(dst + off, base + off); }
    cp_async_commit(); cp_async_wait<1>(); __syncthreads();
  }
  cp_async_wait<0>(); __syncthreads();
  if (smem[threadIdx.x] == 123) sink[0] = 1;
}
#if __CUDA_ARCH__ >= 900 || !defined(__CUDA_ARCH__)
__global__ void tma_bulk(const char* __restrict__ src, int tile_bytes, int iters, long long* out, int* ok) {
  extern __shared__ __align__(128) char smem[];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  if (threadIdx.x == 0) init(&bar, blockDim.x);
  __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    const char* base = src + (size_t)(i & 63) * tile_bytes;
    if (threadIdx.x == 0) {
      unsigned s = (unsigned)__cvta_generic_to_shared(smem), b = (unsigned)__cvta_generic_to_shared(&bar);
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(s), "l"(base), "r"(tile_bytes), "r"(b) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tile_bytes) : "memory");
    } else { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"((unsigned)__cvta_generic_to_shared(&bar)) : "memory"); }
    // wait on the mbarrier phase
    unsigned b = (unsigned)__cvta_generic_to_shared(&bar); unsigned phase = i & 1; unsigned done = 0;
    while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b), "r"(phase) : "memory"); }
    __syncthreads();
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) { out[blockIdx.x] = t1 - t0; int good = 1; for (int j = 0; j < 64; j++) if (smem[j] != src[(size_t)((iters - 1) & 63) * tile_bytes + j]) good = 0; ok[blockIdx.x] = good; }
}
#endif
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 10), iters = argi(argc, argv, "--iters", 2000);
  int sms = sm_count(), threads = 256; int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  char* src; CK(cudaMalloc(&src, 64 * 65536)); std::vector<char> h(64 * 65536); for (size_t i = 0; i < h.size(); i++) h[i] = (char)(i * 7); CK(cudaMemcpy(src, h.data(), h.size(), cudaMemcpyHostToDevice));
  long long* out; int* ok; float* sink; CK(cudaMalloc(&out, sms * 8)); CK(cudaMalloc(&ok, sms * 4)); CK(cudaMalloc(&sink, 4));
  CK(cudaFuncSetAttribute(cpasync_latency, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
  CK(cudaFuncSetAttribute(cpasync_pipelined, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
  Timer t; std::vector<long long> hc(sms);
  for (int tile : {4096, 8192, 16384, 32768, 49152}) {
    std::vector<double> lat, thr;
    cpasync_latency<<<sms, threads, tile>>>(src, tile, iters, out); CK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; r++) {
      cpasync_latency<<<sms, threads, tile>>>(src, tile, iters, out); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hc.data(), out, sms * 8, cudaMemcpyDeviceToHost)); std::sort(hc.begin(), hc.end());
      lat.push_back((double)hc[sms / 2] / iters / (clk_khz * 1e3));      // s per tile, median SM, at the attribute clock
      t.start(); cpasync_pipelined<<<sms, threads, 2 * tile>>>(src, tile, iters, sink); double ms = t.stop_ms();
      thr.push_back((double)tile * iters * sms / (ms * 1e-3));           // aggregate bytes/s
    }
    char d[128]; snprintf(d, sizeof d, "{\"tile_bytes\": %d, \"threads\": %d, \"iters\": %d, \"clock_khz_assumed\": %d}", tile, threads, iters, clk_khz);
    char q[64]; snprintf(q, sizeof q, "async_copy_latency_s[%d]", tile); emit(q, "s", lat, d);
    snprintf(q, sizeof q, "async_copy_bytes_per_s[%d]", tile); emit(q, "byte/s", thr, d);
  }
#if __CUDA_ARCH__ >= 900 || !defined(__CUDA_ARCH__)
  {
    int tile = 16384; std::vector<double> lat; std::vector<int> hok(sms); bool good = true;
    CK(cudaFuncSetAttribute(tma_bulk, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    tma_bulk<<<sms, threads, tile>>>(src, tile, iters, out, ok); cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) {
      for (int r = 0; r < reps; r++) { tma_bulk<<<sms, threads, tile>>>(src, tile, iters, out, ok); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hc.data(), out, sms * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hok.data(), ok, sms * 4, cudaMemcpyDeviceToHost));
        std::sort(hc.begin(), hc.end()); lat.push_back((double)hc[sms / 2] / iters / (clk_khz * 1e3)); for (int x : hok) good = good && x; }
      char d[160]; snprintf(d, sizeof d, "{\"tile_bytes\": %d, \"correct\": %s, \"api\": \"cp.async.bulk + mbarrier\", \"clock_khz_assumed\": %d}", tile, good ? "true" : "false", clk_khz);
      emit("tma_bulk_latency_s[16384]", "s", lat, d);
      printf("{\"quantity\": \"tma_supported\", \"unit\": \"\", \"samples\": [%d], \"detail\": {\"correct\": %s}}\n", good ? 1 : 0, good ? "true" : "false");
    } else {
      printf("{\"quantity\": \"tma_supported\", \"unit\": \"\", \"samples\": [0], \"detail\": {\"error\": \"%s\"}}\n", cudaGetErrorString(e));
    }
  }
#endif
  return 0;
}
