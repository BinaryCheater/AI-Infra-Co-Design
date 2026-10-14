// Shared helpers for the calibration microbenchmarks: error checking, event timing, JSON output.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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
