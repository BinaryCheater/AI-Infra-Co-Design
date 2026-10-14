// B11: clock behaviour under sustained load: the B2 FMA kernel run for --seconds (default 60) in
// one-second chunks while nvidia-smi is sampled once per chunk (clock, power, temperature). Reports
// the per-chunk throughput drift and the clock samples; the fitter derives the sustained clock and
// its variance. The driver's own before/after samples are not enough for a sustained run.
#include "common.cuh"
#include <chrono>
#include <string>
__global__ void fma_kernel(float* out, int iters, float a, float b) {
  float x0 = threadIdx.x * 1e-3f, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
  for (int i = 0; i < iters; i++) { x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b); x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b); }
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
static std::string smi() {
  FILE* f = popen("nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits", "r");
  char buf[256] = {0}; if (f) { if (!fgets(buf, sizeof buf, f)) buf[0] = 0; pclose(f); }
  std::string s(buf); while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s;
}
int main(int argc, char** argv) {
  int seconds = argi(argc, argv, "--seconds", 60), iters = argi(argc, argv, "--iters", 400000);
  int sms = sm_count(), threads = 256, blocks = sms * 8; float* out; CK(cudaMalloc(&out, (size_t)blocks * threads * 4));
  Timer t; std::vector<double> flops_per_s, clk, pwr, temp;
  auto start = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds) {
    t.start(); int launches = 0; double ms = 0;
    auto c0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count() < 1.0) { fma_kernel<<<blocks, threads>>>(out, iters, 0.999f, 0.001f); launches++; CK(cudaDeviceSynchronize()); }
    ms = t.stop_ms();
    flops_per_s.push_back(2.0 * 8 * (double)iters * blocks * threads * launches / (ms * 1e-3) / sms);
    std::string s = smi(); double a = 0, b = 0, c = 0; if (sscanf(s.c_str(), "%lf, %lf, %lf", &a, &b, &c) >= 1) { clk.push_back(a); pwr.push_back(b); temp.push_back(c); }
  }
  char d[96]; snprintf(d, sizeof d, "{\"seconds\": %d, \"chunks\": %zu}", seconds, flops_per_s.size());
  emit("sustained_fp32_flops_per_s_per_sm", "flop/s", flops_per_s, d);
  emit("sustained_clock_mhz", "MHz", clk, d);
  emit("sustained_power_w", "W", pwr, d);
  emit("sustained_temperature_c", "C", temp, d);
  return 0;
}
