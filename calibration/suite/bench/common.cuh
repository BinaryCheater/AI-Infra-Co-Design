// Shared helpers for the calibration microbenchmarks: error checking, event timing, JSON output.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct Timer {
  cudaEvent_t a, b;
  Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  void start() { CK(cudaEventRecord(a)); }
  double stop_ms() { CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};

static int argi(int argc, char** argv, const char* key, int def) {
  for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], key)) return atoi(argv[i + 1]);
  return def;
}
static const char* args(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

// One JSON record per line: {"quantity": ..., "unit": ..., "samples": [...], "detail": {...}}
static void emit(const char* quantity, const char* unit, const std::vector<double>& samples, const char* detail_json) {
  printf("{\"quantity\": \"%s\", \"unit\": \"%s\", \"samples\": [", quantity, unit);
  for (size_t i = 0; i < samples.size(); i++) printf("%s%.9g", i ? ", " : "", samples[i]);
  printf("], \"detail\": %s}\n", detail_json ? detail_json : "{}");
  fflush(stdout);
}

static int sm_count() { cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); return p.multiProcessorCount; }

// ---- SM clock probe: clock64 against %globaltimer, so cycles can be reported beside seconds (clocks unlocked)
__global__ void __clock_probe_kernel(double* mhz) {
  long long c0 = clock64(); unsigned long long t0, t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while (t1 - t0 < 1000000ull);
  long long c1 = clock64(); mhz[0] = (double)(c1 - c0) / (double)(t1 - t0) * 1e3;
}
static double probe_clock_mhz() { double* d; double h = 0; CK(cudaMalloc(&d, 8)); __clock_probe_kernel<<<1, 1>>>(d); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); cudaFree(d); return h; }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned s; asm("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
// emit with the clock recorded: detail gets "sm_clock_mhz" so cycles = seconds * mhz * 1e6 can be read back
static void emit_clocked(const char* quantity, const char* unit, const std::vector<double>& samples, const char* detail_json, double mhz) {
  std::string d = detail_json ? detail_json : "{}";
  char buf[64]; snprintf(buf, sizeof buf, "\"sm_clock_mhz\": %.1f", mhz);
  if (d.size() >= 2 && d[d.size() - 1] == '}') d = d.substr(0, d.size() - 1) + (d.size() > 2 ? ", " : "") + buf + "}";
  emit(quantity, unit, samples, d.c_str());
}
static std::string vec_json(const std::vector<double>& v, const char* fmt = "%.6g") { std::string s = "["; char b[64]; for (size_t i = 0; i < v.size(); i++) { snprintf(b, sizeof b, fmt, v[i]); s += (i ? ", " : "") + std::string(b); } return s + "]"; }
static std::string ivec_json(const std::vector<long long>& v) { std::string s = "["; char b[32]; for (size_t i = 0; i < v.size(); i++) { snprintf(b, sizeof b, "%lld", v[i]); s += (i ? ", " : "") + std::string(b); } return s + "]"; }
