// B2: FP32 FMA throughput per SM. Eight independent FMA chains per thread (ILP 8), 256 threads per
// block, several blocks per SM so the pipeline is saturated; FLOPs = 2 * 8 * iters * threads.
// Method follows Jia et al. (2018) "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking".
#include "common.cuh"
__global__ void fma_kernel(float* out, int iters, float a, float b) {
  float x0 = threadIdx.x * 1e-3f, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
  for (int i = 0; i < iters; i++) {
    x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b);
    x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 10), iters = argi(argc, argv, "--iters", 200000), bps = argi(argc, argv, "--blocks-per-sm", 8);
  int sms = sm_count(), threads = 256, blocks = sms * bps;
  float* out; CK(cudaMalloc(&out, (size_t)blocks * threads * sizeof(float)));
  Timer t; std::vector<double> s;
  for (int w = 0; w < 3; w++) { fma_kernel<<<blocks, threads>>>(out, iters, 0.999f, 0.001f); } CK(cudaDeviceSynchronize());
  for (int r = 0; r < reps; r++) {
    t.start(); fma_kernel<<<blocks, threads>>>(out, iters, 0.999f, 0.001f); double ms = t.stop_ms();
    double flops = 2.0 * 8 * (double)iters * blocks * threads;
    s.push_back(flops / (ms * 1e-3) / sms);
  }
  char d[256]; snprintf(d, sizeof d, "{\"sms\": %d, \"blocks\": %d, \"threads\": %d, \"iters\": %d, \"ilp\": 8}", sms, blocks, threads, iters);
  emit("fp32_flops_per_s_per_sm", "flop/s", s, d);
  return 0;
}
