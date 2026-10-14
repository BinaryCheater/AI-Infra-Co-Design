// B6: shared-memory bandwidth per SM and the bank-conflict penalty: ld.shared throughput for word
// strides 1, 2, 4, 8, 16, 32 (stride 1 is conflict-free; stride 32 serializes a warp on one bank).
// Method follows Jia et al. (2018) and Luo et al. (2024) "Benchmarking and Dissecting the Nvidia Hopper GPU".
#include "common.cuh"
__global__ void smem_kernel(float* out, int iters, int stride, int words) {
  extern __shared__ float buf[];
  for (int i = threadIdx.x; i < words; i += blockDim.x) buf[i] = i * 0.5f;
  __syncthreads();
  float acc = 0; int idx = (threadIdx.x * stride) & (words - 1);
  for (int i = 0; i < iters; i++) { acc += buf[idx]; idx = (idx + 33 * stride) & (words - 1); }
  if (acc == 123456.789f) out[0] = acc;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 10), iters = argi(argc, argv, "--iters", 100000);
  int sms = sm_count(), threads = 256, blocks = sms * 4, words = 8192;        // 32 KiB per block
  float* out; CK(cudaMalloc(&out, 4));
  Timer t;
  for (int stride : {1, 2, 4, 8, 16, 32}) {
    smem_kernel<<<blocks, threads, words * 4>>>(out, iters, stride, words); CK(cudaDeviceSynchronize());
    std::vector<double> s;
    for (int r = 0; r < reps; r++) {
      t.start(); smem_kernel<<<blocks, threads, words * 4>>>(out, iters, stride, words); double ms = t.stop_ms();
      double bytes = 4.0 * iters * (double)blocks * threads;
      s.push_back(bytes / (ms * 1e-3) / sms);
    }
    char d[128]; snprintf(d, sizeof d, "{\"stride_words\": %d, \"blocks\": %d, \"threads\": %d, \"iters\": %d}", stride, blocks, threads, iters);
    char q[64]; snprintf(q, sizeof q, "smem_bytes_per_s_per_sm[stride=%d]", stride); emit(q, "byte/s", s, d);
  }
  return 0;
}
