// BX: kernels for the ncu counter cross-checks (calibration/suite/ncu_crosscheck.py). One kernel per --kernel name,
// launched twice (the first launch is warm-up; ncu profiles the second). Each kernel is the probe of the benchmark it
// cross-checks, reduced to one launch: B13 fill granularity and L2 capacity, B17 issue rate, B21 inner loop, B6/B26 banks, B14 L1.
#include "common.cuh"
__global__ void strided_touch(const float* __restrict__ a, size_t n_words, size_t stride_words, float* out) {
  float acc = 0; size_t nreq = n_words / stride_words;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nreq; i += (size_t)gridDim.x * blockDim.x) acc += a[i * stride_words];
  if (acc == 123456.789f) out[0] = acc;
}
__global__ void footprint_read(const float4* __restrict__ a, size_t n, int passes, float* out) {
  // each block cycles over its own slice (L1 bypassed with .cg), so a hit can only come from the block's earlier pass: a capacity test
  float4 acc = make_float4(0, 0, 0, 0); size_t per = n / gridDim.x; const float4* base = a + blockIdx.x * per;
  for (int p = 0; p < passes; p++) for (size_t i = threadIdx.x; i < per; i += blockDim.x) { float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(base + i)); acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 123456.789f) out[0] = 1;
}
__global__ void fma_issue(float* out, int iters) {
  float x0 = threadIdx.x * 1e-3f, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7; asm volatile("" : "+r"(iters));
  for (int i = 0; i < iters; i++) { x0 = fmaf(x0, 0.999f, 0.001f); x1 = fmaf(x1, 0.999f, 0.001f); x2 = fmaf(x2, 0.999f, 0.001f); x3 = fmaf(x3, 0.999f, 0.001f); x4 = fmaf(x4, 0.999f, 0.001f); x5 = fmaf(x5, 0.999f, 0.001f); x6 = fmaf(x6, 0.999f, 0.001f); x7 = fmaf(x7, 0.999f, 0.001f); }
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
template <int TM, int TN, int TK> __global__ void inner_loop(float* out, int iters) {
  extern __shared__ float sm[]; float* As = sm; float* Bs = sm + TM * TK; for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) As[i] = i * 1e-4f; for (int i = threadIdx.x; i < TK * TN; i += blockDim.x) Bs[i] = i * 2e-4f; __syncthreads();
  const int RM = TM / 16, RN = TN / 16; int tx = threadIdx.x % 16, ty = threadIdx.x / 16; float acc[RM][RN]; for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) acc[i][j] = 0; asm volatile("" : "+r"(iters));
  for (int it = 0; it < iters; it++) {
#pragma unroll 4
    for (int kk = 0; kk < TK; kk++) { float af[RM], bf[RN]; for (int i = 0; i < RM; i++) af[i] = As[(ty + i * 16) * TK + kk]; for (int j = 0; j < RN; j++) bf[j] = Bs[kk * TN + tx + j * 16]; for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]); }
    __syncthreads(); }
  float s = 0; for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) s += acc[i][j]; out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void smem_stride(float* out, int iters, int stride) {
  __shared__ float sf[8192]; for (int i = threadIdx.x; i < 8192; i += blockDim.x) sf[i] = i; __syncthreads(); float acc = 0; int idx = (threadIdx.x * stride) & 8191; asm volatile("" : "+r"(iters));
  for (int k = 0; k < iters; k++) { acc += sf[idx]; idx = (idx + 32 * stride + k) & 8191; }
  if (acc == 123456.789f) out[0] = acc;
}
template <int WIDTH> __global__ void lds_width(float* out, int iters, int stride) {
  extern __shared__ __align__(16) char sb[]; for (int i = threadIdx.x; i < 16384; i += blockDim.x) ((float*)sb)[i] = i; __syncthreads();
  unsigned base = (unsigned)__cvta_generic_to_shared(sb) + (threadIdx.x & 31) * stride * WIDTH; float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; asm volatile("" : "+r"(iters));
  for (int k = 0; k < iters; k += 8) {        // bursts of 8 independent loads, 8 independent consumers (no FADD chain)
#pragma unroll
    for (int i = 0; i < 8; i++) { unsigned a = base + (((k + i) * 512 * stride) & 65535);
      if (WIDTH == 4) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a)); acc[i] += v; }
      else if (WIDTH == 8) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a)); acc[i] += v.x; }
      else { float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); acc[i] += v.x; } } }
  float r = 0; for (int i = 0; i < 8; i++) r += acc[i]; if (r == 123456.789f) out[0] = r;
}
template <int WIDTH> __global__ void __launch_bounds__(128) lds_occ(float* out, int trips) {
  extern __shared__ __align__(16) char sb[]; for (int i = threadIdx.x; i < 2048; i += blockDim.x) ((float*)sb)[i] = i * 1e-6f; __syncthreads();
  unsigned base = (unsigned)__cvta_generic_to_shared(sb) + (threadIdx.x & 31) * WIDTH; float acc[32]; for (int i = 0; i < 32; i++) acc[i] = 0; asm volatile("" : "+r"(trips));
  for (int t = 0; t < trips; t++) { unsigned off = (t & 3) * 2048;
    if (WIDTH == 4) { float s[32];
#pragma unroll
      for (int i = 0; i < 32; i++) asm volatile("ld.shared.f32 %0, [%1];" : "=f"(s[i]) : "r"(base + ((off + i * 128) & 8191)));
#pragma unroll
      for (int i = 0; i < 32; i++) acc[i] += s[i]; }
    else { float4 s[32];
#pragma unroll
      for (int i = 0; i < 32; i++) asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(s[i].x), "=f"(s[i].y), "=f"(s[i].z), "=f"(s[i].w) : "r"(base + ((off + i * 512) & 8191)));
#pragma unroll
      for (int i = 0; i < 32; i++) acc[i] += s[i].x; }
    asm volatile("" : "+f"(acc[0]), "+f"(acc[7]), "+f"(acc[15]), "+f"(acc[31])); }
  float r = 0; for (int i = 0; i < 32; i++) r += acc[i]; if (r == 123456.789f) out[0] = r;
}
__global__ void l1_chase(const unsigned* __restrict__ next, int steps, unsigned* out) {
  extern __shared__ char pad[]; unsigned p = 0; for (int i = 0; i < steps; i++) { unsigned v; asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(next + p)); p = v; } out[0] = p; if (p == 0xFFFFFFFF) pad[0] = 1;
}
int main(int argc, char** argv) {
  const char* kernel = "strided"; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--kernel")) kernel = argv[i + 1];
  long a1 = argi(argc, argv, "--a", 0), a2 = argi(argc, argv, "--b", 0); int sms = sm_count(); float* out; CK(cudaMalloc(&out, (size_t)sms * 16 * 1024 * 4));
  if (!strcmp(kernel, "strided")) {          // --a stride bytes over 1 GiB
    size_t bytes = 1ull << 30; float* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes));
    for (int r = 0; r < 2; r++) { strided_touch<<<sms * 8, 256>>>(d, bytes / 4, (size_t)a1 / 4, out); CK(cudaDeviceSynchronize()); } }
  else if (!strcmp(kernel, "footprint")) {   // --a footprint MiB, cyclic streaming read, 8 passes (the first pass fills)
    size_t bytes = (size_t)a1 << 20; float4* d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 0, bytes));
    for (int r = 0; r < 2; r++) { footprint_read<<<sms * 8, 256>>>(d, bytes / 16, 8, out); CK(cudaDeviceSynchronize()); } }
  else if (!strcmp(kernel, "fma_issue")) {   // --a warps per block, one block per SM (90 KB smem keeps it to one)
    CK(cudaFuncSetAttribute(fma_issue, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    for (int r = 0; r < 2; r++) { fma_issue<<<sms, (int)a1 * 32, 90 * 1024>>>(out, 20000); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); } }
  else if (!strcmp(kernel, "inner_T1") || !strcmp(kernel, "inner_T5") || !strcmp(kernel, "inner_R2")) {
    auto run = [&](auto k, int TM, int TN, int TK) { int smem = (TM * TK + TK * TN) * 4; CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); for (int r = 0; r < 2; r++) { k<<<sms, 256, smem>>>(out, 200); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); } };
    if (!strcmp(kernel, "inner_T1")) run(inner_loop<128, 256, 16>, 128, 256, 16); else if (!strcmp(kernel, "inner_T5")) run(inner_loop<128, 128, 32>, 128, 128, 32); else run(inner_loop<32, 32, 32>, 32, 32, 32); }
  else if (!strcmp(kernel, "smem_stride")) { // --a stride in words
    for (int r = 0; r < 2; r++) { smem_stride<<<sms * 8, 256>>>(out, 4000, (int)a1); CK(cudaDeviceSynchronize()); } }
  else if (!strcmp(kernel, "lds_width")) {   // --a width bytes (4, 8, 16), --b stride words; 16 warps per SM
    CK(cudaFuncSetAttribute(lds_width<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(lds_width<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(lds_width<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    for (int r = 0; r < 2; r++) { if (a1 == 4) lds_width<4><<<sms, 512, 90 * 1024>>>(out, 4000, (int)a2); else if (a1 == 8) lds_width<8><<<sms, 512, 90 * 1024>>>(out, 4000, (int)a2); else lds_width<16><<<sms, 512, 90 * 1024>>>(out, 4000, (int)a2); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); } }
  else if (!strcmp(kernel, "lds_occ")) {     // --a width bytes (4 or 16), --b warps per sub-partition (blocks of 4 warps, padded)
    CK(cudaFuncSetAttribute(lds_occ<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(lds_occ<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); int pad = std::max(8192L, (96L * 1024) / a2);
    for (int r = 0; r < 2; r++) { if (a1 == 4) lds_occ<4><<<sms * (int)a2, 128, pad>>>(out, 400); else lds_occ<16><<<sms * (int)a2, 128, pad>>>(out, 400); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); } }
  else if (!strcmp(kernel, "l1_chase")) {    // --a footprint KiB at 128 B stride, --b dynamic smem bytes
    size_t bytes = (size_t)a1 << 10; unsigned* d; CK(cudaMalloc(&d, bytes)); std::vector<unsigned> h(bytes / 4); size_t n = bytes / 128; for (size_t i = 0; i < n; i++) h[i * 32] = (unsigned)(((i + 1) % n) * 32); CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute(l1_chase, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); int carve = a2 ? std::min(100, (int)(100.0 * a2 / (100 * 1024))) : 0; CK(cudaFuncSetAttribute(l1_chase, cudaFuncAttributePreferredSharedMemoryCarveout, carve));
    for (int r = 0; r < 2; r++) { l1_chase<<<1, 32, (int)a2>>>(d, 20000, (unsigned*)out); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); } }
  else { fprintf(stderr, "unknown kernel %s\n", kernel); return 2; }
  printf("{\"kernel\": \"%s\", \"a\": %ld, \"b\": %ld}\n", kernel, a1, a2);
  return 0;
}
