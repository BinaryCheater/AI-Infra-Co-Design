// B5: memory latency at each rung by pointer chasing over a random cyclic permutation of cache
// lines (64 B granularity), one thread, working sets 4 KB .. 1 GB. Method: Mei & Chu (2017) and
// Jia et al. (2018), fine-grained P-chase with a random pattern so prefetchers do not help.
#include "common.cuh"
#include <random>
__global__ void chase(const unsigned* __restrict__ next, unsigned start, int steps, unsigned* out) {
  unsigned p = start;
  for (int i = 0; i < steps; i++) p = next[p];
  out[0] = p;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), steps = argi(argc, argv, "--steps", 200000);
  const size_t line = 64;
  size_t sizes[] = {4u << 10, 16u << 10, 64u << 10, 128u << 10, 256u << 10, 1u << 20, 4u << 20, 8u << 20, 16u << 20, 32u << 20, 64u << 20, 256u << 20, 1ull << 30};
  size_t maxb = 1ull << 30;
  unsigned* d; unsigned* out; CK(cudaMalloc(&d, maxb)); CK(cudaMalloc(&out, 4));
  std::vector<unsigned> h(maxb / sizeof(unsigned));
  std::mt19937_64 rng(12345);
  Timer t;
  for (size_t bytes : sizes) {
    size_t lines = bytes / line, stride = line / sizeof(unsigned);
    std::vector<unsigned> perm(lines); for (size_t i = 0; i < lines; i++) perm[i] = i;
    for (size_t i = lines - 1; i > 0; i--) { size_t j = rng() % i; std::swap(perm[i], perm[j]); }   // Sattolo: one cycle
    for (size_t i = 0; i < lines; i++) h[perm[i] * stride] = perm[(i + 1) % lines] * stride;
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    chase<<<1, 1>>>(d, 0, steps, out); CK(cudaDeviceSynchronize());
    std::vector<double> s;
    for (int r = 0; r < reps; r++) { t.start(); chase<<<1, 1>>>(d, 0, steps, out); double ms = t.stop_ms(); s.push_back(ms * 1e6 / steps); }
    char dd[128]; snprintf(dd, sizeof dd, "{\"working_set_bytes\": %zu, \"steps\": %d, \"line_bytes\": %zu}", bytes, steps, line);
    char q[64]; snprintf(q, sizeof q, "latency[%zu]", bytes); emit(q, "ns", s, dd);
  }
  return 0;
}
