// B4: global-memory bandwidth (read, write, copy) as a function of working set, 1 MB .. 8 GB.
// Grid-stride float4 kernels; the knee between the small-set and large-set plateaus gives the
// effective L2 capacity. Method follows the STREAM-style kernels of Jia et al. and the Accel-Sim tuner.
#include "common.cuh"
__global__ void read_kernel(const float4* __restrict__ a, float* out, size_t n) {
  float4 acc = make_float4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = a[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;   // never true; defeats dead-code elimination
}
__global__ void write_kernel(float4* __restrict__ a, size_t n) {
  float4 v = make_float4(1, 2, 3, 4);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a[i] = v;
}
__global__ void copy_kernel(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5); double max_gb = atof(args(argc, argv, "--max-gb", "8"));
  int sms = sm_count(), threads = 256, blocks = sms * 8;
  size_t sizes[] = {1u << 20, 4u << 20, 16u << 20, 64u << 20, 256u << 20, 1ull << 30, 2ull << 30, 4ull << 30, 8ull << 30};
  size_t maxb = (size_t)(max_gb * (1ull << 30));
  float4 *a, *b; float* out; CK(cudaMalloc(&a, maxb)); CK(cudaMalloc(&b, maxb)); CK(cudaMalloc(&out, 4));
  CK(cudaMemset(a, 0, maxb)); CK(cudaMemset(b, 0, maxb));
  Timer t;
  for (size_t bytes : sizes) {
    if (bytes > maxb) break;
    size_t n = bytes / sizeof(float4);
    int passes = (int)std::max<size_t>(1, (256u << 20) / bytes);      // >= 256 MB moved per sample
    std::vector<double> sr, sw, sc;
    for (int w = 0; w < 2; w++) { read_kernel<<<blocks, threads>>>(a, out, n); write_kernel<<<blocks, threads>>>(b, n); copy_kernel<<<blocks, threads>>>(a, b, n); } CK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; r++) {
      t.start(); for (int p = 0; p < passes; p++) read_kernel<<<blocks, threads>>>(a, out, n); double ms = t.stop_ms(); sr.push_back((double)bytes * passes / (ms * 1e-3));
      t.start(); for (int p = 0; p < passes; p++) write_kernel<<<blocks, threads>>>(b, n); ms = t.stop_ms(); sw.push_back((double)bytes * passes / (ms * 1e-3));
      t.start(); for (int p = 0; p < passes; p++) copy_kernel<<<blocks, threads>>>(a, b, n); ms = t.stop_ms(); sc.push_back(2.0 * bytes * passes / (ms * 1e-3));
    }
    char d[160]; snprintf(d, sizeof d, "{\"working_set_bytes\": %zu, \"passes\": %d, \"blocks\": %d, \"threads\": %d}", bytes, passes, blocks, threads);
    char q[64];
    snprintf(q, sizeof q, "bandwidth_read[%zu]", bytes); emit(q, "byte/s", sr, d);
    snprintf(q, sizeof q, "bandwidth_write[%zu]", bytes); emit(q, "byte/s", sw, d);
    snprintf(q, sizeof q, "bandwidth_copy[%zu]", bytes); emit(q, "byte/s", sc, d);
  }
  return 0;
}
