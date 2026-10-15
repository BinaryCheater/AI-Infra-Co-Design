// B37: mbarrier costs, cluster multicast, DSMEM (round 2, group 2). (a) arrive + try_wait round trip with 1, 32, 256 arriving
// threads, and with a 4 KB expect_tx pending; (b) the phase flip: last arrive to try_wait success with one thread; (c) clusters of
// 2 and 4 with a multicast TMA load into every block's shared memory, L2 sectors per block from ncu (--probe multicast / --probe
// unicast); (d) DSMEM read latency and bandwidth between the two blocks of a cluster. Cluster launches that the chip refuses are
// recorded as unsupported with the error text.
#include "tma_common.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// A __syncthreads after every wait keeps the arriving threads from completing two phases before a slow waiter looks: a parity wait
// whose phase has completed twice never returns. The reference loop below (arrivals = 0) is the same loop without the mbarrier,
// so the mbarrier's cost is the difference (the fitter subtracts it).
__global__ void mbar_cost(long long* out, int iters, int arrivals) {
  __shared__ unsigned long long bar; unsigned b = saddr(&bar); if (threadIdx.x == 0) mbar_init(b, arrivals > 0 ? arrivals : 1); __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) { if (arrivals > 0) { if ((int)threadIdx.x < arrivals) mbar_arrive(b); mbar_wait(b, i & 1); } __syncthreads(); }
  long long t1 = clock64(); if (threadIdx.x == 0) out[blockIdx.x] = (t1 - t0) / iters;
}
__global__ void mbar_tx(const char* src, long long* out, int iters) {   // one thread: expect_tx 4 KB + bulk copy, every thread waits
  extern __shared__ __align__(128) char sm[]; __shared__ unsigned long long bar; unsigned b = saddr(&bar); if (threadIdx.x == 0) mbar_init(b, 1); __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) { if (threadIdx.x == 0) { mbar_expect_tx(b, 4096); tma_load_1d(saddr(sm), src + (size_t)(i & 63) * 4096, 4096, b); } mbar_wait(b, i & 1); __syncthreads(); }
  long long t1 = clock64(); if (threadIdx.x == 0) out[blockIdx.x] = (t1 - t0) / iters;
}
__global__ void phase_flip(long long* out, int iters) {   // thread 0 arrives and times its own try_wait success; the other threads wait too
  __shared__ unsigned long long bar; unsigned b = saddr(&bar); if (threadIdx.x == 0) mbar_init(b, 1); __syncthreads(); long long acc = 0;
  for (int i = 0; i < iters; i++) { __syncthreads(); long long t0 = clock64(); if (threadIdx.x == 0) mbar_arrive(b); mbar_wait(b, i & 1); if (threadIdx.x == 0) acc += clock64() - t0; }
  if (threadIdx.x == 0) out[blockIdx.x] = acc / iters;
}
__device__ __forceinline__ int mbar_wait_bounded(unsigned b, unsigned phase, long long limit) { long long t0 = clock64(); unsigned done = 0; while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b), "r"(phase) : "memory"); if (clock64() - t0 > limit) return 0; } return 1; }
template <int CL> __global__ void __cluster_dims__(CL, 1, 1) multicast(const char* src, long long* out, int iters, int use_multicast) {
  extern __shared__ __align__(128) char sm[]; __shared__ unsigned long long bar; unsigned b = saddr(&bar), dst = saddr(sm); cg::cluster_group cluster = cg::this_cluster(); unsigned rank = cluster.block_rank();
  if (threadIdx.x == 0) mbar_init(b, 1); __syncthreads(); cluster.sync(); long long t0 = clock64(); __shared__ int bad; if (threadIdx.x == 0) bad = 0; __syncthreads();
  for (int i = 0; i < iters; i++) { const char* base = src + (size_t)((i * 3 + blockIdx.x / CL) & 63) * 16384;
    if (!bad) {   // a block whose barrier never completes stops copying but keeps every cluster.sync, so its partners do not hang
      if (threadIdx.x == 0) { mbar_expect_tx(b, 16384);
        if (use_multicast) { if (rank == 0) { unsigned short mask = (1 << CL) - 1; asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(base), "r"(16384), "r"(b), "h"(mask) : "memory"); } }
        else tma_load_1d(dst, base, 16384, b); }
      if (!mbar_wait_bounded(b, i & 1, 2000000)) { if (threadIdx.x == 0) bad = 1; } __syncthreads(); }
    cluster.sync(); }
  long long t1 = clock64(); if (threadIdx.x == 0) out[blockIdx.x] = bad ? -1 : (t1 - t0) / iters;
}
__global__ void __cluster_dims__(2, 1, 1) dsmem(long long* out, float* sink, int iters) {   // block 1 reads block 0's shared memory: chase (latency) and stream (bandwidth)
  extern __shared__ __align__(128) char sm[]; cg::cluster_group cluster = cg::this_cluster(); unsigned rank = cluster.block_rank(); unsigned* local = (unsigned*)sm;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) local[i] = (i * 97 + 1) & 4095; cluster.sync();
  unsigned* remote = cluster.map_shared_rank(local, rank ^ 1); long long lat = 0, bw = 0;
  if (rank == 1) { unsigned p = threadIdx.x & 4095; long long t0 = clock64(); for (int i = 0; i < iters; i++) p = remote[p]; lat = (clock64() - t0) / iters; sink[blockIdx.x * blockDim.x + threadIdx.x] = p;
    float acc = 0; long long t1 = clock64(); for (int i = 0; i < iters; i++) {
#pragma unroll
      for (int u = 0; u < 8; u++) acc += __uint_as_float(remote[(threadIdx.x + u * 256 + i * 2048) & 4095]); } asm volatile("" : "+f"(acc)); bw = clock64() - t1; sink[blockIdx.x * blockDim.x + threadIdx.x] += acc; }
  cluster.sync(); if (rank == 1 && threadIdx.x == 0) { out[blockIdx.x * 2] = lat; out[blockIdx.x * 2 + 1] = bw; }
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6;
  char* src; CK(cudaMalloc(&src, 64 * 65536)); CK(cudaMemset(src, 7, 64 * 65536)); long long* out; CK(cudaMalloc(&out, 8 * 8192)); float* sink; CK(cudaMalloc(&sink, (size_t)sms * 2 * 256 * 4));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  std::vector<long long> h(8192);
  auto med = [&](int n, int stride = 1, int off = 0) { CK(cudaMemcpy(h.data(), out, (size_t)n * stride * 8, cudaMemcpyDeviceToHost)); std::vector<long long> v; for (int i = 0; i < n; i++) v.push_back(h[i * stride + off]); std::sort(v.begin(), v.end()); return (double)v[n / 2]; };
  if (!probe) {
    for (int arr : {0, 1, 32, 256}) { std::vector<double> s; for (int r = 0; r < reps + 1; r++) { mbar_cost<<<sms, 256>>>(out, 2000, arr); CK(cudaDeviceSynchronize()); if (r) s.push_back(med(sms)); }
      char q[64], d[260]; snprintf(q, sizeof q, "mbarrier_cycles[arrivals=%d]", arr); snprintf(d, sizeof d, "{\"arrivals\": %d, \"threads\": 256, \"method\": \"arrive by the first %d threads + try_wait.parity by all + __syncthreads, nothing pending, cycles per round trip; arrivals=0 is the loop with the __syncthreads alone (the reference to subtract)\"}", arr, arr); emit_clocked(q, "cycles", s, d, mhz); }
    { CK(cudaFuncSetAttribute(mbar_tx, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192)); std::vector<double> s; for (int r = 0; r < reps + 1; r++) { mbar_tx<<<sms, 256, 8192>>>(src, out, 2000); CK(cudaDeviceSynchronize()); if (r) s.push_back(med(sms)); }
      emit_clocked("mbarrier_cycles[arrivals=1,expect_tx=4096]", "cycles", s, "{\"arrivals\": 1, \"expect_tx_bytes\": 4096, \"method\": \"expect_tx + a 4 KB bulk copy by one thread, try_wait by all, then __syncthreads: the mbarrier with a transaction pending (= the 4 KB TMA latency plus the reference loop)\"}", mhz); }
    { std::vector<double> s; for (int r = 0; r < reps + 1; r++) { phase_flip<<<sms, 256>>>(out, 2000); CK(cudaDeviceSynchronize()); if (r) s.push_back(med(sms)); }
      emit_clocked("mbarrier_phase_flip_cycles", "cycles", s, "{\"method\": \"one arriving thread: its own arrive to its try_wait success, 256 threads waiting\"}", mhz); } }
  // (c) clusters: multicast vs unicast, 16 KB per block per iteration
  auto cluster_run = [&](int cl, int mc, const char* label, bool emit) {
    int grid = sms - (sms % cl); int iters = 200; cudaError_t e;
    if (cl == 2) { CK(cudaFuncSetAttribute(multicast<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384)); multicast<2><<<grid, 128, 16384>>>(src, out, iters, mc); } else { CK(cudaFuncSetAttribute(multicast<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384)); multicast<4><<<grid, 128, 16384>>>(src, out, iters, mc); }
    e = cudaGetLastError(); if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { if (emit) printf("{\"quantity\": \"tma_multicast_cycles_per_tile[%s]\", \"unit\": \"cycles\", \"samples\": [], \"detail\": {\"measured\": false, \"cluster\": %d, \"multicast\": %d, \"reason\": \"%s\"}}\n", label, cl, mc, cudaGetErrorString(e)); cudaGetLastError(); return false; }
    { CK(cudaMemcpy(h.data(), out, (size_t)grid * 8, cudaMemcpyDeviceToHost)); bool timeout = false; for (int i = 0; i < grid; i++) timeout |= h[i] < 0;
      if (timeout) { if (emit) printf("{\"quantity\": \"tma_multicast_cycles_per_tile[%s]\", \"unit\": \"cycles\", \"samples\": [], \"detail\": {\"measured\": false, \"cluster\": %d, \"multicast\": %d, \"reason\": \"the mbarrier never completed: the copy's bytes did not arrive (timeout after 20 M cycles)\"}}\n", label, cl, mc); return false; } }
    if (!emit) return true;
    std::vector<double> s; for (int r = 0; r < reps; r++) { if (cl == 2) multicast<2><<<grid, 128, 16384>>>(src, out, iters, mc); else multicast<4><<<grid, 128, 16384>>>(src, out, iters, mc); CK(cudaDeviceSynchronize()); s.push_back(med(grid)); }
    char q[80], d[240]; snprintf(q, sizeof q, "tma_multicast_cycles_per_tile[%s]", label); snprintf(d, sizeof d, "{\"cluster\": %d, \"multicast\": %d, \"tile_bytes\": 16384, \"blocks\": %d, \"method\": \"16 KB per block per iteration; multicast: rank 0 issues one copy into every block of the cluster\"}", cl, mc, grid); emit_clocked(q, "cycles", s, d, mhz); return true; };
  if (probe) { int mc = !strcmp(probe, "multicast"); cluster_run(2, mc, "", false); cluster_run(2, mc, "", false); return 0; }   // two launches: ncu skips the first
  bool ok2 = cluster_run(2, 0, "cluster=2,unicast", true); if (ok2) { cluster_run(2, 1, "cluster=2,multicast", true); cluster_run(4, 0, "cluster=4,unicast", true); cluster_run(4, 1, "cluster=4,multicast", true); }
  // (d) DSMEM
  { int grid = sms - (sms % 2); CK(cudaFuncSetAttribute(dsmem, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384)); dsmem<<<grid, 256, 16384>>>(out, sink, 2000); cudaError_t e = cudaGetLastError(); if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("{\"quantity\": \"dsmem_latency_cycles\", \"unit\": \"cycles\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"%s\"}}\n", cudaGetErrorString(e)); cudaGetLastError(); }
    else { std::vector<double> lat, bw; for (int r = 0; r < reps; r++) { dsmem<<<grid, 256, 16384>>>(out, sink, 2000); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), out, (size_t)grid * 2 * 8, cudaMemcpyDeviceToHost)); std::vector<long long> a, b; for (int i = 1; i < grid; i += 2) { a.push_back(h[i * 2]); b.push_back(h[i * 2 + 1]); } std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); lat.push_back((double)a[a.size() / 2]); bw.push_back(2000.0 * 8 * 256 * 4 / ((double)b[b.size() / 2] / hz)); }
      emit_clocked("dsmem_latency_cycles", "cycles", lat, "{\"method\": \"dependent 4 B loads from the other block's shared memory (map_shared_rank), one thread per lane, cycles per load\"}", mhz);
      emit_clocked("dsmem_bytes_per_s_per_sm", "byte/s", bw, "{\"method\": \"256 threads, 8 independent 4 B remote shared loads per step, bytes per second per reading SM\"}", mhz); } }
  return 0;
}
