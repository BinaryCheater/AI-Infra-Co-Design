// B19: 1. spill cost: the same FMA-dense 64-accumulator kernel built here is measured at the register
// cap the driver passes (--maxrreg N, rebuilt per cap by driver's b19 loop); 2. local memory latency
// and bandwidth; 3. register bank conflicts: not measured (needs SASS-level operand control);
// 4. allocation granularity: extracted by fit.py from B9's raw records.
#include "common.cuh"
__global__ void acc64(float* out, int iters) {
  float a[64]; for (int i = 0; i < 64; i++) a[i] = threadIdx.x + i; float m = 0.999f, c = 0.001f;
  for (int k = 0; k < iters; k++) {
#pragma unroll
    for (int i = 0; i < 64; i++) a[i] = fmaf(a[i], m, c); }
  float s = 0; for (int i = 0; i < 64; i++) s += a[i]; out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void local_chase(int steps, unsigned* out, long long* cyc, int idx0) {
  unsigned arr[256]; for (int i = 0; i < 256; i++) arr[i] = (i * 97 + 1) % 256;   // dynamic indexing forces local memory
  unsigned p = idx0 & 255; long long t0 = clock64(); for (int i = 0; i < steps; i++) p = arr[p]; long long t1 = clock64(); out[0] = p; cyc[0] = t1 - t0;
}
__global__ void local_stream(float* out, int iters, long long* cyc) {
  float arr[64]; for (int i = 0; i < 64; i++) arr[i] = threadIdx.x + i; float acc = 0; int j = threadIdx.x & 63; long long t0 = clock64();
  for (int k = 0; k < iters; k++) {
#pragma unroll
    for (int u = 0; u < 8; u++) acc += arr[(j + u * 8) & 63];      // 8 independent 4 B local loads per step (dynamic index keeps the array in local memory)
    j = (j + 1) & 63; arr[j] = acc; }
  asm volatile("" : "+f"(acc)); long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = acc; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; int cap = argi(argc, argv, "--maxrreg", 255); int spill = argi(argc, argv, "--spill-bytes", -1);
  float* out; CK(cudaMalloc(&out, (size_t)sms * 8 * 256 * 4)); unsigned* uo; CK(cudaMalloc(&uo, 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 1024));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, acc64));
  { std::vector<double> s; int iters = 20000; Timer t; acc64<<<sms * 8, 256>>>(out, iters); CK(cudaDeviceSynchronize()); for (int r = 0; r < reps; r++) { t.start(); acc64<<<sms * 8, 256>>>(out, iters); double ms = t.stop_ms(); s.push_back(2.0 * 64 * iters * sms * 8 * 256 / (ms * 1e-3) / sms); }
    char det[2048]; snprintf(det, sizeof det, "{\"maxrregcount\": %d, \"regs_used\": %d, \"local_bytes\": %zu, \"spill_bytes_ptxas\": %d, \"method\": \"64-accumulator FMA kernel per register cap; throughput per SM; spill bytes from ptxas -v\"}", cap, fa.numRegs, fa.localSizeBytes, spill);
    char q[48]; snprintf(q, sizeof q, "spill_flops_per_s_per_sm[maxrreg=%d]", cap); emit_clocked(q, "flop/s", s, det, mhz); }
  if (cap == 255) {   // local memory and bandwidth only once
    { std::vector<double> s; for (int r = 0; r < reps; r++) { local_chase<<<1, 32>>>(10000, uo, cyc, r); CK(cudaDeviceSynchronize()); long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost)); s.push_back((double)c / 10000 / hz * 1e9); }
      cudaFuncAttributes la; CK(cudaFuncGetAttributes(&la, local_chase)); char det[2048]; snprintf(det, sizeof det, "{\"local_bytes_per_thread\": %zu, \"method\": \"single-thread dependent chase over a 1 KB local array (dynamic index)\"}", la.localSizeBytes); emit_clocked("local_mem_latency_ns", "ns", s, det, mhz); }
    { std::vector<double> s; int iters = 20000; for (int r = 0; r < reps; r++) { local_stream<<<sms, 256>>>(out, iters, cyc); CK(cudaDeviceSynchronize()); std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), cyc, sms * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); double cycles = (double)h[h.size() / 2]; s.push_back((8.0 * 4 + 4) * iters * 256 / (cycles / hz)); }
      emit_clocked("local_mem_bytes_per_s_per_sm", "byte/s", s, "{\"method\": \"one block of 256 threads per SM streaming a 256 B local array: 8 independent 4 B loads and one store per step, bytes per second per SM (clock64 of the median block)\"}", mhz); }
    printf("{\"quantity\": \"reg_bank_conflict_penalty\", \"unit\": \"\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"needs operand-register control at the SASS level (register pinning is not available from CUDA C++ or PTX; would require patching cubins), lowest priority per B13_plus.md\"}}\n");
  }
  return 0;
}
