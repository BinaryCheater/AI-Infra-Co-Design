// B7: __syncthreads cost per barrier as a function of block size, one block per SM, a loop of
// barriers with a trivial dependent update so the compiler keeps every one.
#include "common.cuh"
__global__ void barrier_kernel(float* out, int iters) {
  __shared__ float s[1024];
  s[threadIdx.x] = threadIdx.x;
  float v = 0;
  for (int i = 0; i < iters; i++) { __syncthreads(); v += s[(threadIdx.x + i) & 1023]; }
  if (v == 123456.789f) out[0] = v;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 10), iters = argi(argc, argv, "--iters", 100000);
  int sms = sm_count(); float* out; CK(cudaMalloc(&out, 4));
  Timer t;
  for (int threads : {128, 256, 512, 1024}) {
    barrier_kernel<<<sms, threads>>>(out, iters); CK(cudaDeviceSynchronize());
    std::vector<double> s;
    for (int r = 0; r < reps; r++) { t.start(); barrier_kernel<<<sms, threads>>>(out, iters); double ms = t.stop_ms(); s.push_back(ms * 1e-3 / iters); }
    char d[96]; snprintf(d, sizeof d, "{\"threads\": %d, \"blocks\": %d, \"iters\": %d}", threads, sms, iters);
    char q[48]; snprintf(q, sizeof q, "barrier_s[%d]", threads); emit(q, "s", s, d);
  }
  return 0;
}
