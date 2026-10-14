// B12 (Spark): unified-memory access cost from the GPU. Streaming reads over (a) cudaMalloc (device),
// (b) cudaMallocManaged touched by the GPU first, (c) cudaMallocManaged touched by the CPU first (first
// GPU pass pays any migration, second pass is steady state), (d) cudaMallocHost (pinned, coherent
// access). 1 GiB working set.
#include "common.cuh"
__global__ void read_kernel(const float4* __restrict__ a, float* out, size_t n) {
  float4 acc = make_float4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = a[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
__global__ void touch(float4* a, size_t n) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a[i] = make_float4(1, 1, 1, 1); }
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5); size_t bytes = 1ull << 30, n = bytes / 16; int sms = sm_count(), blocks = sms * 8, threads = 256;
  float* out; CK(cudaMalloc(&out, 4)); Timer t;
  auto bw = [&](const float4* p, int passes, std::vector<double>& s) { t.start(); for (int i = 0; i < passes; i++) read_kernel<<<blocks, threads>>>(p, out, n); double ms = t.stop_ms(); s.push_back((double)bytes * passes / (ms * 1e-3)); };
  { float4* d; CK(cudaMalloc(&d, bytes)); touch<<<blocks, threads>>>(d, n); CK(cudaDeviceSynchronize()); std::vector<double> s; bw(d, 1, s); s.clear(); for (int r = 0; r < reps; r++) bw(d, 2, s); emit("unified_read_bytes_per_s[device]", "byte/s", s, "{\"alloc\": \"cudaMalloc\", \"working_set_bytes\": 1073741824}"); cudaFree(d); }
  { float4* m; CK(cudaMallocManaged(&m, bytes)); touch<<<blocks, threads>>>(m, n); CK(cudaDeviceSynchronize()); std::vector<double> s; bw(m, 1, s); s.clear(); for (int r = 0; r < reps; r++) bw(m, 2, s); emit("unified_read_bytes_per_s[managed_gpu_first]", "byte/s", s, "{\"alloc\": \"cudaMallocManaged, GPU touched first\", \"working_set_bytes\": 1073741824}"); cudaFree(m); }
  { std::vector<double> first, steady;
    for (int r = 0; r < reps; r++) { float4* m; CK(cudaMallocManaged(&m, bytes)); for (size_t i = 0; i < n; i += 4096) m[i] = make_float4(2, 2, 2, 2);   // CPU touches every page
      bw(m, 1, first); bw(m, 2, steady); cudaFree(m); }
    emit("unified_read_bytes_per_s[managed_cpu_first_pass1]", "byte/s", first, "{\"alloc\": \"cudaMallocManaged, CPU touched first; first GPU pass\", \"working_set_bytes\": 1073741824}");
    emit("unified_read_bytes_per_s[managed_cpu_first_steady]", "byte/s", steady, "{\"alloc\": \"cudaMallocManaged, CPU touched first; later passes\", \"working_set_bytes\": 1073741824}"); }
  { float4* h; CK(cudaMallocHost(&h, bytes)); for (size_t i = 0; i < n; i += 4096) h[i] = make_float4(3, 3, 3, 3); std::vector<double> s; bw(h, 1, s); s.clear(); for (int r = 0; r < reps; r++) bw(h, 2, s); emit("unified_read_bytes_per_s[pinned_host]", "byte/s", s, "{\"alloc\": \"cudaMallocHost (coherent access)\", \"working_set_bytes\": 1073741824}"); cudaFreeHost(h); }
  return 0;
}
