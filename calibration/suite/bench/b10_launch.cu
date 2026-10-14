// B10: kernel launch overhead: 1000 empty launches, device time per launch (events) and host time per
// asynchronous launch call.
#include "common.cuh"
#include <chrono>
__global__ void empty_kernel() {}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 10), n = 1000;
  for (int w = 0; w < 100; w++) empty_kernel<<<1, 32>>>(); CK(cudaDeviceSynchronize());
  Timer t; std::vector<double> dev, host;
  for (int r = 0; r < reps; r++) {
    auto h0 = std::chrono::steady_clock::now(); t.start();
    for (int i = 0; i < n; i++) empty_kernel<<<1, 32>>>();
    auto h1 = std::chrono::steady_clock::now(); double ms = t.stop_ms();
    dev.push_back(ms * 1e-3 / n); host.push_back(std::chrono::duration<double>(h1 - h0).count() / n);
  }
  emit("launch_s", "s", dev, "{\"launches\": 1000, \"timing\": \"cudaEvent, back-to-back\"}");
  emit("launch_host_s", "s", host, "{\"launches\": 1000, \"timing\": \"host wall time per async launch call\"}");
  return 0;
}
