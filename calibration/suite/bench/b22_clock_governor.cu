// B22: the clock governor. 1. step response from idle (clock64 vs globaltimer sampled every 1 ms inside
// a 2 s FMA burst); 2. sustained clock under four 30 s loads (FMA, DRAM stream, tensor, the T5-shaped
// GEMM loop); 3. power/clock samples per second from nvidia-smi during each; 4. clock at kernel start after idle gaps.
#include "common.cuh"
#include <mma.h>
#include <cuda_bf16.h>
#include <chrono>
#include <thread>
using namespace nvcuda;
__global__ void fma_burst(float* out, long long* clk_samples, int* nsamples, double seconds) {
  float x0 = threadIdx.x * 1e-3f, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
  unsigned long long t0 = gtimer(), last = t0; long long c_last = clock64(); int k = 0;
  while (gtimer() - t0 < (unsigned long long)(seconds * 1e9)) {
    for (int i = 0; i < 2000; i++) { x0 = fmaf(x0, 0.999f, 0.001f); x1 = fmaf(x1, 0.999f, 0.001f); x2 = fmaf(x2, 0.999f, 0.001f); x3 = fmaf(x3, 0.999f, 0.001f); x4 = fmaf(x4, 0.999f, 0.001f); x5 = fmaf(x5, 0.999f, 0.001f); x6 = fmaf(x6, 0.999f, 0.001f); x7 = fmaf(x7, 0.999f, 0.001f); }
    if (blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long now = gtimer(); if (now - last >= 1000000ull && k < 4000) { long long c = clock64(); clk_samples[k++] = (long long)((double)(c - c_last) / (double)(now - last) * 1e3); last = now; c_last = c; } } }
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; if (blockIdx.x == 0 && threadIdx.x == 0) nsamples[0] = k;
}
__global__ void stream_burst(const float4* a, size_t n, float* out, double seconds) {
  unsigned long long t0 = gtimer(); float4 acc = make_float4(0, 0, 0, 0);
  while (gtimer() - t0 < (unsigned long long)(seconds * 1e9)) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = a[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; } }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
__global__ void tensor_burst(float* out, double seconds) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a; wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b; wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1, c2, c3;
  for (int i = 0; i < a.num_elements; i++) a.x[i] = 1.0f; for (int i = 0; i < b.num_elements; i++) b.x[i] = 0.5f; wmma::fill_fragment(c0, 0.f); wmma::fill_fragment(c1, 0.f); wmma::fill_fragment(c2, 0.f); wmma::fill_fragment(c3, 0.f);
  unsigned long long t0 = gtimer(); while (gtimer() - t0 < (unsigned long long)(seconds * 1e9)) { for (int i = 0; i < 500; i++) { wmma::mma_sync(c0, a, b, c0); wmma::mma_sync(c1, a, b, c1); wmma::mma_sync(c2, a, b, c2); wmma::mma_sync(c3, a, b, c3); } }
  for (int i = 0; i < c0.num_elements; i++) c0.x[i] += c1.x[i] + c2.x[i] + c3.x[i]; __shared__ float buf[4][256]; wmma::store_matrix_sync(buf[(threadIdx.x / 32) & 3], c0, 16, wmma::mem_row_major); if (buf[0][threadIdx.x & 15] == 123456.f) out[0] = 1;
}
// a GEMM-shaped loop: 16 x 16 thread grid, 8 x 8 outputs, smem tile reads + FMAs, like the generated kernel's inner loop
__global__ void gemm_burst(float* out, double seconds) {
  __shared__ float As[128 * 32], Bs[32 * 128]; for (int i = threadIdx.x; i < 128 * 32; i += blockDim.x) { As[i] = i * 1e-4f; Bs[i] = i * 2e-4f; } __syncthreads();
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16; float acc[8][8]; for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0;
  unsigned long long t0 = gtimer();
  while (gtimer() - t0 < (unsigned long long)(seconds * 1e9)) { for (int kk = 0; kk < 32; kk++) { float af[8], bf[8]; for (int i = 0; i < 8; i++) af[i] = As[(ty + i * 16) * 32 + kk]; for (int j = 0; j < 8; j++) bf[j] = Bs[kk * 128 + tx + j * 16]; for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]); } __syncthreads(); }
  float s = 0; for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) s += acc[i][j]; if (s == 123456.789f) out[0] = s;
}
static std::string smi() { FILE* f = popen("nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown --format=csv,noheader,nounits", "r"); char buf[256] = {0}; if (f) { if (!fgets(buf, sizeof buf, f)) buf[0] = 0; pclose(f); } std::string s(buf); while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s; }
int main(int argc, char** argv) {
  int sms = sm_count(); double load_s = argi(argc, argv, "--load-seconds", 30);
  float* out; CK(cudaMalloc(&out, (size_t)sms * 8 * 256 * 4)); long long* clk; CK(cudaMalloc(&clk, 4000 * 8)); int* ns; CK(cudaMalloc(&ns, 4)); float4* a; CK(cudaMalloc(&a, 1 << 30)); CK(cudaMemset(a, 0, 1 << 30));
  // 1. step response from idle
  { double idle = probe_clock_mhz(); std::this_thread::sleep_for(std::chrono::seconds(3)); double idle2 = probe_clock_mhz();
    fma_burst<<<sms * 8, 256>>>(out, clk, ns, 2.0); CK(cudaDeviceSynchronize()); int k; CK(cudaMemcpy(&k, ns, 4, cudaMemcpyDeviceToHost)); std::vector<long long> h(k); CK(cudaMemcpy(h.data(), clk, k * 8, cudaMemcpyDeviceToHost));
    long long sustained = h[k - 1]; int t_sust = k; for (int i = 0; i < k; i++) if (h[i] >= 0.98 * sustained) { t_sust = i; break; }
    std::string series = "["; for (int i = 0; i < k; i += std::max(1, k / 200)) series += (i ? ", " : "") + std::to_string(h[i]); series += "]";
    std::string det = "{\"idle_probe_mhz\": [" + std::to_string(idle) + ", " + std::to_string(idle2) + "], \"start_mhz\": " + std::to_string(h[0]) + ", \"sustained_mhz\": " + std::to_string(sustained) + ", \"samples_ms\": " + std::to_string(k) + ", \"clock_mhz_series_subsampled\": " + series + ", \"method\": \"2 s FMA burst from idle; clock64/globaltimer every 1 ms; warm-up = first ms within 2 % of the final clock\"}";
    emit("clock_warmup_s", "s", std::vector<double>{t_sust * 1e-3}, det.c_str()); emit("clock_idle_hz", "Hz", std::vector<double>{idle * 1e6, idle2 * 1e6}, "{\"method\": \"clock probe at idle, twice\"}"); }
  // 2 + 3. sustained clock by load, with nvidia-smi samples per second
  auto sustained = [&](const char* name, auto launch) {
    std::vector<double> clkv, pwr, temp; std::string cap = "";
    launch(); // async, runs load_s seconds
    auto t0 = std::chrono::steady_clock::now(); while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < load_s - 0.5) { std::this_thread::sleep_for(std::chrono::seconds(1)); std::string s = smi(); double c = 0, p = 0, t = 0; char a1[32] = "", a2[32] = ""; if (sscanf(s.c_str(), "%lf, %lf, %lf, %31[^,], %31s", &c, &p, &t, a1, a2) >= 3) { clkv.push_back(c); pwr.push_back(p); temp.push_back(t); cap = std::string(a1) + "/" + a2; } }
    CK(cudaDeviceSynchronize()); double probe = probe_clock_mhz(); std::vector<double> sc = clkv; std::sort(sc.begin(), sc.end());
    std::string det = "{\"load\": \"" + std::string(name) + "\", \"seconds\": " + std::to_string((int)load_s) + ", \"smi_clock_mhz_series\": " + vec_json(clkv, "%.0f") + ", \"power_w_series\": " + vec_json(pwr, "%.1f") + ", \"temperature_c_series\": " + vec_json(temp, "%.0f") + ", \"sw_power_cap_and_thermal_last\": \"" + cap + "\", \"probe_mhz_after\": " + std::to_string(probe) + ", \"method\": \"nvidia-smi once per second during the load; the governor is power-limited if SW power cap is active while temperature is flat\"}";
    char q[64]; snprintf(q, sizeof q, "clock_hz_sustained_by_load[%s]", name); std::vector<double> hz; for (double c : clkv) hz.push_back(c * 1e6); emit(q, "Hz", hz, det.c_str()); };
  sustained("fma", [&]() { fma_burst<<<sms * 8, 256>>>(out, clk, ns, load_s); });
  sustained("dram_stream", [&]() { stream_burst<<<sms * 8, 256>>>(a, (1 << 30) / 16, out, load_s); });
  sustained("tensor_bf16", [&]() { tensor_burst<<<sms * 4, 128>>>(out, load_s); });
  sustained("gemm_inner_loop", [&]() { gemm_burst<<<sms, 256>>>(out, load_s); });
  // 4. clock at kernel start after idle gaps
  { std::string curve = "["; bool f = true; std::vector<double> after;
    for (int gap_ms : {0, 1, 10, 100, 1000}) { std::vector<double> s; for (int r = 0; r < 5; r++) { if (gap_ms) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms)); fma_burst<<<sms * 8, 256>>>(out, clk, ns, 0.0002); CK(cudaDeviceSynchronize()); double m = probe_clock_mhz(); s.push_back(m * 1e6); }
      std::sort(s.begin(), s.end()); after.push_back(s[s.size() / 2]); char b[48]; snprintf(b, sizeof b, "%s[%d, %.0f]", f ? "" : ", ", gap_ms, s[s.size() / 2]); curve += b; f = false; }
    curve += "]"; std::string det = "{\"gap_ms_vs_clock_hz\": " + curve + ", \"method\": \"200 us FMA kernel after an idle gap, clock probed right after it; what a reps loop with host gaps experiences\"}";
    emit("clock_after_gap_hz", "Hz", after, det.c_str()); }
  return 0;
}
