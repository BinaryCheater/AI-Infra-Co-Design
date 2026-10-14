// B24 (M12): atomic throughput by scope and pattern, red vs atom, contention curve, latency.
#include "common.cuh"
template <int PATTERN, int RED> __global__ void atom_global(float* g, int* gi, int iters, long long* cyc) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32; int idx = PATTERN == 0 ? 0 : PATTERN == 1 ? warp * 32 : (blockIdx.x * blockDim.x + threadIdx.x) * 32; float r = 0; long long t0 = clock64();
  for (int i = 0; i < iters; i++) { if (RED) { atomicAdd(&g[idx & 1048575], 1.0f); } else { r += atomicAdd(&g[idx & 1048575], 1.0f); } }
  long long t1 = clock64(); if (r == 123456.789f) gi[0] = 1; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0; (void)lane;
}
template <int PATTERN> __global__ void atom_shared(int* gi, int iters, long long* cyc) {
  __shared__ float s[8192]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) s[i] = 0; __syncthreads(); int warp = threadIdx.x / 32; int idx = PATTERN == 0 ? 0 : PATTERN == 1 ? warp * 32 : threadIdx.x * 32; float r = 0; long long t0 = clock64();
  for (int i = 0; i < iters; i++) r += atomicAdd(&s[idx & 8191], 1.0f);
  long long t1 = clock64(); if (r == 123456.789f) gi[0] = 1; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
__global__ void atom_latency(float* g, long long* cyc, int iters) { float r = 0; long long t0 = clock64(); for (int i = 0; i < iters; i++) r = atomicAdd(&g[(int)r & 1023], 1.0f) * 0.f; long long t1 = clock64(); cyc[0] = (t1 - t0) / iters; if (r == 1.f) g[1] = 1; }
__global__ void atom_latency_sh(long long* cyc, int iters, int* gi) { __shared__ float s[1024]; s[0] = 0; __syncthreads(); float r = 0; long long t0 = clock64(); for (int i = 0; i < iters; i++) r = atomicAdd(&s[(int)r & 1023], 1.0f) * 0.f; long long t1 = clock64(); cyc[0] = (t1 - t0) / iters; if (r == 1.f) gi[0] = 1; }
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; float* g; CK(cudaMalloc(&g, 1048576 * 4)); CK(cudaMemset(g, 0, 1048576 * 4)); int* gi; CK(cudaMalloc(&gi, 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  auto emit_rate = [&](const char* name, int blocks, int iters, std::vector<double>& s) { char det[2048]; snprintf(det, sizeof det, "{\"blocks\": %d, \"threads\": 256, \"iters\": %d}", blocks, iters); char q[128]; snprintf(q, sizeof q, "atomic_throughput_per_s[%s]", name); emit_clocked(q, "atomics/s", s, det, mhz); };
#define RUN_G(KERN, NAME, BLOCKS, ITERS) { std::vector<double> s; for (int r = 0; r < reps; r++) { KERN<<<BLOCKS, 256>>>(g, gi, ITERS, cyc); CK(cudaDeviceSynchronize()); std::vector<long long> h(BLOCKS); CK(cudaMemcpy(h.data(), cyc, (BLOCKS) * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)(ITERS) * 256 * (BLOCKS) / ((double)h[(BLOCKS) - 1] / hz)); } emit_rate(NAME, BLOCKS, ITERS, s); }
#define RUN_S(KERN, NAME, BLOCKS, ITERS) { std::vector<double> s; for (int r = 0; r < reps; r++) { KERN<<<BLOCKS, 256>>>(gi, ITERS, cyc); CK(cudaDeviceSynchronize()); std::vector<long long> h(BLOCKS); CK(cudaMemcpy(h.data(), cyc, (BLOCKS) * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)(ITERS) * 256 * (BLOCKS) / ((double)h[(BLOCKS) - 1] / hz)); } emit_rate(NAME, BLOCKS, ITERS, s); }
  RUN_G((atom_global<0, 0>), "global_same_address_atom", sms, 2000); RUN_G((atom_global<0, 1>), "global_same_address_red", sms, 2000);
  RUN_G((atom_global<1, 0>), "global_per_warp_atom", sms, 2000); RUN_G((atom_global<1, 1>), "global_per_warp_red", sms, 2000);
  RUN_G((atom_global<2, 0>), "global_per_thread_atom", sms, 2000); RUN_G((atom_global<2, 1>), "global_per_thread_red", sms, 2000);
  RUN_S((atom_shared<0>), "shared_same_address", sms, 2000); RUN_S((atom_shared<1>), "shared_per_warp", sms, 2000); RUN_S((atom_shared<2>), "shared_per_thread", sms, 2000);
  for (int b : {1, 2, 4, 8, 16, 32, 48}) { char n[48]; snprintf(n, sizeof n, "global_same_address_contention_blocks=%d", b); RUN_G((atom_global<0, 0>), n, b, 1000); }
  { std::vector<double> s; for (int r = 0; r < reps; r++) { atom_latency<<<1, 1>>>(g, cyc, 2000); CK(cudaDeviceSynchronize()); long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); s.push_back((double)c / hz * 1e9); } emit_clocked("atomic_latency_ns[global]", "ns", s, "{\"method\": \"single-thread dependent atomicAdd chain (L2)\"}", mhz); }
  { std::vector<double> s; for (int r = 0; r < reps; r++) { atom_latency_sh<<<1, 1>>>(cyc, 2000, gi); CK(cudaDeviceSynchronize()); long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); s.push_back((double)c / hz * 1e9); } emit_clocked("atomic_latency_ns[shared]", "ns", s, "{\"method\": \"single-thread dependent shared atomicAdd chain\"}", mhz); }
  return 0;
}
