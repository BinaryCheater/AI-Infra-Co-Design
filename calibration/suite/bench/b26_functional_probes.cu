// B26: the functional semantics the verifier's event model assumes, each as a count over many trials.
// 1 in-flight cp.async visibility (chunk order, window, torn chunks); 2 wait_group N; 3 barrier / named
// barrier / fence visibility; 4 TMA early bytes; 5 FP32 contraction, ftz, tensor accumulate rounding;
// 6 shared-memory bank map at 8 B and 16 B widths; 7 independent thread scheduling and __syncwarp.
#include "common.cuh"
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using namespace nvcuda;
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) { unsigned s = (unsigned)__cvta_generic_to_shared(smem); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem)); }
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n"); }
// 1. warp 0 copies 32 KB; warp 1 polls the 2048 chunks and records first-change time per chunk and torn observations
__global__ void visibility(const unsigned* src, long long* first_seen, int* torn, int* done_flag) {
  extern __shared__ unsigned __align__(16) slot[];        // 32 KB = 8192 words = 2048 chunks of 4 words
  const int chunks = 2048; int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) slot[i] = 0xFFFFFFFFu;
  __syncthreads();
  if (w == 0) { for (int c = lane; c < chunks; c += 32) cp_async16(slot + c * 4, src + c * 4); commit(); }
  if (w == 1) { // poll until all chunks changed, recording first-seen clock per chunk; a chunk with some words new and some sentinel is torn
    long long t0 = clock64(); int seen = 0; int my_torn = 0; for (int c = lane; c < chunks; c += 32) first_seen[c] = -1;
    for (int spin = 0; spin < 200000 && seen < chunks / 32; spin++) {
      for (int c = lane; c < chunks; c += 32) { if (first_seen[c] >= 0) continue; volatile unsigned* p = slot + c * 4; unsigned a = p[0], b = p[1], c2 = p[2], d = p[3]; int changed = (a != 0xFFFFFFFFu) + (b != 0xFFFFFFFFu) + (c2 != 0xFFFFFFFFu) + (d != 0xFFFFFFFFu);
        if (changed == 4) { first_seen[c] = clock64() - t0; seen++; } else if (changed > 0) { my_torn++; } } }
    atomicAdd(torn, my_torn); }
  if (w == 0) { asm volatile("cp.async.wait_group 0;\n"); } __syncthreads(); if (threadIdx.x == 0) done_flag[0] = 1;
}
// 2. wait_group 1 after three groups: groups 0,1 complete (must); group 2 complete too? (allowed)
__global__ void waitgroup(const unsigned* src, int* counts) {
  extern __shared__ unsigned __align__(16) s3[]; for (int i = threadIdx.x; i < 3 * 1024; i += blockDim.x) s3[i] = 0xFFFFFFFFu; __syncthreads();
  for (int g = 0; g < 3; g++) { for (int c = threadIdx.x; c < 256; c += blockDim.x) cp_async16(s3 + g * 1024 + c * 4, src + g * 1024 + c * 4); commit(); }
  asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
  if (threadIdx.x == 0) { int c0 = 1, c1 = 1, c2 = 1; for (int i = 0; i < 1024; i++) { if (s3[i] == 0xFFFFFFFFu) c0 = 0; if (s3[1024 + i] == 0xFFFFFFFFu) c1 = 0; if (s3[2048 + i] == 0xFFFFFFFFu) c2 = 0; } atomicAdd(&counts[0], c0); atomicAdd(&counts[1], c1); atomicAdd(&counts[2], c2); }
  asm volatile("cp.async.wait_group 0;\n"); __syncthreads();
}
// 3. visibility: syncthreads, named barrier arrive-only, fence-without-barrier
__global__ void visibility_barriers(int* counts, long long* fence_cycles, int trial) {
  __shared__ volatile int v[3]; __shared__ volatile int flag; int w = threadIdx.x / 32;
  if (threadIdx.x == 0) { v[0] = v[1] = v[2] = 0; flag = 0; } __syncthreads();
  if (w == 0 && threadIdx.x == 0) v[0] = trial + 1; __syncthreads();
  if (w == 1 && threadIdx.x == 32) atomicAdd(&counts[0], v[0] == trial + 1);
  // named barrier: warp 0 arrives (no wait), warp 1 syncs; both with 64 threads
  if (w == 0) { if (threadIdx.x == 0) v[1] = trial + 1; asm volatile("bar.arrive 1, 64;\n"); }
  if (w == 1) { asm volatile("bar.sync 1, 64;\n"); if (threadIdx.x == 32) atomicAdd(&counts[1], v[1] == trial + 1); }
  __syncthreads();
  // fence without barrier: warp 0 stores + threadfence_block + sets flag; warp 1 polls value then flag
  if (w == 0 && threadIdx.x == 0) { long long t0 = clock64(); v[2] = trial + 1; __threadfence_block(); flag = 1; fence_cycles[0] = clock64() - t0; }
  if (w == 1 && threadIdx.x == 32) { long long t0 = clock64(); int spins = 0; while (v[2] != trial + 1 && spins < 100000) spins++; long long t1 = clock64(); atomicAdd(&counts[2], v[2] == trial + 1); fence_cycles[1] = t1 - t0; while (flag == 0 && spins < 200000) spins++; }
  __syncthreads();
}
// 4. TMA: any byte visible before complete_tx fires?
__global__ void tma_early(const unsigned* src, int* counts) {
  extern __shared__ unsigned __align__(128) t4[]; __shared__ unsigned long long bar; unsigned b = (unsigned)__cvta_generic_to_shared(&bar), dst = (unsigned)__cvta_generic_to_shared(t4);
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) t4[i] = 0xFFFFFFFFu; if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b)); __syncthreads();
  int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(16384) : "memory"); asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(16384), "r"(b) : "memory"); }
  if (w == 1) { // poll chunks and the barrier phase: count chunks seen changed while the barrier phase is still incomplete
    int early = 0; for (int spin = 0; spin < 20000; spin++) { unsigned done = 0; asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], 0;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b) : "memory"); if (done) break;
      volatile unsigned* p = t4 + lane * 128; if (p[0] != 0xFFFFFFFFu) { early++; break; } }
    atomicAdd(&counts[0], early); }
  unsigned done = 0; while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b) : "memory"); } __syncthreads();
}
// 5. FP32: contraction, ftz, tensor rounding
__global__ void fp32_probe(const float* a, const float* b, const float* c, int n, int* counts, float* ulp_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float x = a[i] * b[i] + c[i]; float y = __fadd_rn(__fmul_rn(a[i], b[i]), c[i]); if (x != y) atomicAdd(&counts[0], 1);
  float d = 1.0e-40f * (1.0f + 1e-7f * i); if (d == 0.0f) atomicAdd(&counts[1], 1);       // denormal flushed?
}
__global__ void tensor_round(float* out_bf16, float* out_tf32, float* out_fp16) {
  // A (16x16, row-major) and B (16x16, col-major) built in shared memory with a defined layout, so the host can form the same products
  __shared__ __nv_bfloat16 A[256], B[256]; __shared__ __half HA[256], HB[256]; __shared__ float TA[128], TB[128]; __shared__ float buf[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) { int k = i % 16; float av = 1.0f + (i % 7) * 0.0078125f, bv = (k & 1) ? ldexpf(1.0f + (i % 5) * 0.0625f, -18) : 1.0f + (i % 5) * 0.015625f; /* alternate ~1 and ~2^-18 terms: the exact sum needs > 24 bits */ A[i] = __float2bfloat16(av); B[i] = __float2bfloat16(bv); HA[i] = __float2half(av); HB[i] = __float2half(bv); }
  for (int i = threadIdx.x; i < 128; i += blockDim.x) { int k = i % 8; TA[i] = wmma::__float_to_tf32(1.0f + (i % 7) * 0.0078125f); TB[i] = wmma::__float_to_tf32((k & 1) ? ldexpf(1.0f + (i % 5) * 0.0625f, -18) : 1.0f + (i % 5) * 0.015625f); }
  __syncthreads();
  if (threadIdx.x < 32) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a; wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b; wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::load_matrix_sync(a, A, 16); wmma::load_matrix_sync(b, B, 16); wmma::fill_fragment(c, 0.0f); wmma::mma_sync(c, a, b, c); wmma::store_matrix_sync(buf, c, 16, wmma::mem_row_major); }
  __syncthreads(); if (threadIdx.x < 256) out_bf16[threadIdx.x] = buf[threadIdx.x]; __syncthreads();
  if (threadIdx.x < 32) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> ta; wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> tb; wmma::fragment<wmma::accumulator, 16, 16, 8, float> tc;
    wmma::load_matrix_sync(ta, TA, 8); wmma::load_matrix_sync(tb, TB, 8); wmma::fill_fragment(tc, 0.0f); wmma::mma_sync(tc, ta, tb, tc); wmma::store_matrix_sync(buf, tc, 16, wmma::mem_row_major); }
  __syncthreads(); if (threadIdx.x < 256) out_tf32[threadIdx.x] = buf[threadIdx.x]; __syncthreads();
  if (threadIdx.x < 32) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> ha; wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> hb; wmma::fragment<wmma::accumulator, 16, 16, 16, float> hc;
    wmma::load_matrix_sync(ha, HA, 16); wmma::load_matrix_sync(hb, HB, 16); wmma::fill_fragment(hc, 0.0f); wmma::mma_sync(hc, ha, hb, hc); wmma::store_matrix_sync(buf, hc, 16, wmma::mem_row_major); }
  __syncthreads(); if (threadIdx.x < 256) out_fp16[threadIdx.x] = buf[threadIdx.x];
}
// 6. bank map at 8 B and 16 B widths: time a strided access pattern per width
template <typename T> __global__ void bank_width(float* out, int iters, int stride_elems, long long* cyc) {
  extern __shared__ __align__(16) char sb[]; T* buf = (T*)sb; int n = 16384 / sizeof(T); for (int i = threadIdx.x; i < n; i += blockDim.x) ((float*)buf)[i * (sizeof(T) / 4)] = i;
  __syncthreads(); float acc = 0; int idx = (threadIdx.x * stride_elems) % n; long long t0 = clock64();
  for (int i = 0; i < iters; i++) { T v = buf[idx]; acc += *(float*)&v; idx = (idx + 97 * stride_elems) % n; }
  long long t1 = clock64(); if (acc == 123456.789f) out[0] = acc; if (threadIdx.x == 0) cyc[blockIdx.x] = (t1 - t0) / iters;
}
// 7. independent thread scheduling: divergent halves spin; does the second half start before the first finishes?
__global__ void its_probe(long long* stamps, int* counts) {
  int lane = threadIdx.x % 32; long long t0 = clock64();
  if (lane & 1) { long long s = clock64(); while (clock64() - s < 2000) { } stamps[lane] = clock64() - t0; }
  else { stamps[lane] = clock64() - t0; long long s = clock64(); while (clock64() - s < 2000) { } }
  __syncwarp(); long long after = clock64() - t0; stamps[32 + lane] = after;
  __syncthreads(); if (threadIdx.x == 0) { long long even_start = stamps[0], odd_end = stamps[1]; atomicAdd(&counts[0], even_start < odd_end); long long mn = after, mx = after; for (int l = 0; l < 32; l++) { mn = stamps[32 + l] < mn ? stamps[32 + l] : mn; mx = stamps[32 + l] > mx ? stamps[32 + l] : mx; } stamps[64] = mx - mn; }
}
int main(int argc, char** argv) {
  int trials = argi(argc, argv, "--trials", 10000); double mhz = probe_clock_mhz(); int sms = sm_count();
  unsigned* src; CK(cudaMalloc(&src, 1 << 20)); CK(cudaMemset(src, 0x5A, 1 << 20)); long long* fs; CK(cudaMalloc(&fs, 2048 * 8)); int* counts; CK(cudaMalloc(&counts, 64)); int* done; CK(cudaMalloc(&done, 4));
  CK(cudaFuncSetAttribute(visibility, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024));
  // 1.
  { std::vector<long long> h(2048); std::vector<double> inorder_frac, spread, firsts; long long torn_total = 0; int ntr = std::min(trials, 2000);
    for (int t = 0; t < ntr; t++) { CK(cudaMemset(counts, 0, 64)); visibility<<<1, 64, 32 * 1024>>>(src, fs, counts, done); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), fs, 2048 * 8, cudaMemcpyDeviceToHost)); int torn; CK(cudaMemcpy(&torn, counts, 4, cudaMemcpyDeviceToHost)); torn_total += torn;
      long long mn = 1LL << 60, mx = -1; int inv = 0, seen = 0; long long prev = -1; for (int c = 0; c < 2048; c++) { if (h[c] < 0) continue; seen++; mn = std::min(mn, h[c]); mx = std::max(mx, h[c]); if (prev >= 0 && h[c] < prev) inv++; prev = h[c]; }
      if (seen > 0) { inorder_frac.push_back(1.0 - (double)inv / std::max(1, seen - 1)); spread.push_back((double)(mx - mn)); firsts.push_back((double)mn); } }
    std::sort(spread.begin(), spread.end()); std::sort(inorder_frac.begin(), inorder_frac.end()); std::sort(firsts.begin(), firsts.end());
    char det[2048]; snprintf(det, sizeof det, "{\"trials\": %d, \"tile_bytes\": 32768, \"chunks\": 2048, \"in_order_fraction_median\": %.3f, \"window_cycles_p1\": %.0f, \"window_cycles_p50\": %.0f, \"window_cycles_p99\": %.0f, \"first_chunk_cycles_p50\": %.0f, \"torn_chunks_observed\": %lld, \"method\": \"warp 1 polls all 16 B chunks of a slot warp 0 is filling by cp.async.cg, recording the first clock64 at which each chunk is fully new; torn = a chunk seen with some words new and some sentinel\"}",
             ntr, inorder_frac[inorder_frac.size() / 2], spread[spread.size() / 100], spread[spread.size() / 2], spread[spread.size() * 99 / 100], firsts[firsts.size() / 2], torn_total);
    emit_clocked("cp_async_window_cycles", "cycles", std::vector<double>{spread[spread.size() / 2]}, det, mhz); }
  // 2.
  { CK(cudaMemset(counts, 0, 64)); CK(cudaFuncSetAttribute(waitgroup, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * 1024)); waitgroup<<<trials / 64 + 1, 64, 12 * 1024>>>(src, counts); CK(cudaDeviceSynchronize()); int h[3]; CK(cudaMemcpy(h, counts, 12, cudaMemcpyDeviceToHost)); int n = trials / 64 + 1;
    char det[2048]; snprintf(det, sizeof det, "{\"trials\": %d, \"group0_complete\": %d, \"group1_complete\": %d, \"group2_already_complete\": %d, \"method\": \"three 4 KB groups, wait_group 1: groups 0 and 1 must be complete; group 2 may be\"}", n, h[0], h[1], h[2]);
    emit_clocked("wait_group_overcomplete_observed", "count", std::vector<double>{(double)h[2]}, det, mhz); }
  // 3.
  { CK(cudaMemset(counts, 0, 64)); std::vector<double> fence; for (int t = 0; t < std::min(trials, 2000); t++) { visibility_barriers<<<1, 64>>>(counts, fs, t); CK(cudaDeviceSynchronize()); long long h2[2]; CK(cudaMemcpy(h2, fs, 16, cudaMemcpyDeviceToHost)); fence.push_back((double)h2[1]); }
    int h[3]; CK(cudaMemcpy(h, counts, 12, cudaMemcpyDeviceToHost)); std::sort(fence.begin(), fence.end()); int n = std::min(trials, 2000);
    char det[2048]; snprintf(det, sizeof det, "{\"trials\": %d, \"syncthreads_visible\": %d, \"named_barrier_arrive_only_visible\": %d, \"fence_no_barrier_visible\": %d, \"fence_visibility_cycles_p50\": %.0f, \"fence_visibility_cycles_p99\": %.0f, \"method\": \"warp 0 stores, warp 1 reads after __syncthreads / after bar.sync paired with warp 0's bar.arrive / after polling with only __threadfence_block on the writer\"}", n, h[0], h[1], h[2], fence[fence.size() / 2], fence[fence.size() * 99 / 100]);
    emit_clocked("threadfence_visibility_cycles", "cycles", std::vector<double>{fence[fence.size() / 2]}, det, mhz); }
  // 4.
  { CK(cudaMemset(counts, 0, 64)); CK(cudaFuncSetAttribute(tma_early, cudaFuncAttributeMaxDynamicSharedMemorySize, 20 * 1024)); int n = std::min(trials, 2000); for (int t = 0; t < n; t++) tma_early<<<1, 64, 16 * 1024>>>(src, counts); cudaError_t e = cudaDeviceSynchronize(); int h; CK(cudaMemcpy(&h, counts, 4, cudaMemcpyDeviceToHost));
    char det[2048]; snprintf(det, sizeof det, "{\"trials\": %d, \"early_bytes_observed\": %d, \"status\": \"%s\", \"method\": \"warp 1 polls 16 KB being filled by cp.async.bulk while mbarrier.test_wait says the phase is incomplete\"}", n, h, cudaGetErrorString(e)); emit_clocked("tma_early_bytes_observed", "count", std::vector<double>{(double)h}, det, mhz); }
  // 5.
  { int n = 1 << 20; float *a, *b, *c; CK(cudaMalloc(&a, n * 4)); CK(cudaMalloc(&b, n * 4)); CK(cudaMalloc(&c, n * 4)); std::vector<float> ha(n), hb(n), hc(n); unsigned s = 1; for (int i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; ha[i] = (s >> 8) / 16777216.0f + 0.5f; s = s * 1664525u + 1013904223u; hb[i] = (s >> 8) / 16777216.0f + 0.5f; s = s * 1664525u + 1013904223u; hc[i] = (s >> 8) / 16777216.0f * 1e-3f; }
    CK(cudaMemcpy(a, ha.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(b, hb.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(c, hc.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(counts, 0, 64)); float* ulp; CK(cudaMalloc(&ulp, 4));
    fp32_probe<<<n / 256, 256>>>(a, b, c, n, counts, ulp); CK(cudaDeviceSynchronize()); int h[2]; CK(cudaMemcpy(h, counts, 8, cudaMemcpyDeviceToHost));
    char det[2048]; snprintf(det, sizeof det, "{\"samples\": %d, \"contracted_differs_from_rn_mul_add\": %d, \"denormal_flushed\": %d, \"method\": \"a*b+c vs __fadd_rn(__fmul_rn(a,b),c) on 2^20 random inputs; 1e-40f*x == 0 tests ftz\"}", n, h[0], h[1]);
    emit_clocked("fma_contraction_default", "bool", std::vector<double>{h[0] > 0 ? 1.0 : 0.0}, det, mhz);
    emit_clocked("ftz_default", "bool", std::vector<double>{h[1] > 0 ? 1.0 : 0.0}, det, mhz);
    float *ob, *ot, *oh; CK(cudaMalloc(&ob, 1024)); CK(cudaMalloc(&ot, 1024)); CK(cudaMalloc(&oh, 1024)); tensor_round<<<1, 256>>>(ob, ot, oh); CK(cudaDeviceSynchronize());
    std::vector<float> rb(256), rt(256), rh(256); CK(cudaMemcpy(rb.data(), ob, 1024, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rt.data(), ot, 1024, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rh.data(), oh, 1024, cudaMemcpyDeviceToHost));
    // FP64 reference of the same products (inputs are exactly representable after the bf16/fp16/tf32 rounding applied on device, recomputed here)
    auto bf16r = [](float x) { unsigned u; memcpy(&u, &x, 4); unsigned r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; float y; memcpy(&y, &r, 4); return (double)y; };
    auto tf32r = [](float x) { unsigned u; memcpy(&u, &x, 4); unsigned r = (u + 0xFFF + ((u >> 13) & 1)) & 0xFFFFE000u; float y; memcpy(&y, &r, 4); return (double)y; };
    double worst_b = 0, worst_t = 0, worst_h = 0;
    auto av = [](int i) { return 1.0f + (i % 7) * 0.0078125f; }; auto bv = [](int i, int k) { return (k & 1) ? ldexpf(1.0f + (i % 5) * 0.0625f, -18) : 1.0f + (i % 5) * 0.015625f; };
    for (int r = 0; r < 16; r++) for (int col = 0; col < 16; col++) { double sb = 0, st = 0, sh = 0; for (int k = 0; k < 16; k++) { int ia = r * 16 + k, ib = col * 16 + k; sb += bf16r(av(ia)) * bf16r(bv(ib, ib % 16)); sh += (double)__half2float(__float2half(av(ia))) * (double)__half2float(__float2half(bv(ib, ib % 16))); }
      for (int k = 0; k < 8; k++) { int ia = r * 8 + k, ib = col * 8 + k; st += tf32r(av(ia)) * tf32r(bv(ib, ib % 8)); }
      auto ulps = [](double ref, float got) { float f = (float)ref; float nxt = nextafterf(f, 1e30f); double ulp = (double)nxt - f; return fabs((double)got - ref) / ulp; };
      worst_b = std::max(worst_b, ulps(sb, rb[r * 16 + col])); worst_t = std::max(worst_t, ulps(st, rt[r * 16 + col])); worst_h = std::max(worst_h, ulps(sh, rh[r * 16 + col])); }
    char det2[2048]; snprintf(det2, sizeof det2, "{\"bf16_max_ulp\": %.3f, \"fp16_max_ulp\": %.3f, \"tf32_max_ulp\": %.3f, \"method\": \"wmma 16x16x16 (bf16, fp16) and 16x16x8 (tf32) with FP32 accumulate against an FP64 sum of the rounded inputs' exact products; error in FP32 ulps of the result; the inputs alternate ~1 and ~2^-18 terms so the exact sum needs more than 24 bits and the accumulate must round. A row-major (A[r*16+k]), B col-major (B[col*16+k]) loaded with load_matrix_sync, so host and device use the same elements\"}", worst_b, worst_h, worst_t);
    emit_clocked("tensor_accumulate_ulp_bound", "ulp", std::vector<double>{std::max(worst_b, std::max(worst_t, worst_h))}, det2, mhz); }
  // 6.
  { CK(cudaFuncSetAttribute(bank_width<float2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * 1024)); CK(cudaFuncSetAttribute(bank_width<float4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * 1024)); float* o; CK(cudaMalloc(&o, 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 64));
    std::string t8 = "[", t16 = "["; bool f = true; std::vector<double> base8, base16;
    for (int stride : {1, 2, 4, 8, 16, 32}) { double c8, c16; bank_width<float2><<<1, 256, 16 * 1024>>>(o, 20000, stride, cyc); CK(cudaDeviceSynchronize()); long long h; CK(cudaMemcpy(&h, cyc, 8, cudaMemcpyDeviceToHost)); c8 = (double)h;
      bank_width<float4><<<1, 256, 16 * 1024>>>(o, 20000, stride, cyc); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&h, cyc, 8, cudaMemcpyDeviceToHost)); c16 = (double)h; base8.push_back(c8); base16.push_back(c16);
      char b[48]; snprintf(b, sizeof b, "%s[%d, %.1f]", f ? "" : ", ", stride, c8); t8 += b; snprintf(b, sizeof b, "%s[%d, %.1f]", f ? "" : ", ", stride, c16); t16 += b; f = false; }
    t8 += "]"; t16 += "]";
    std::string det = "{\"cycles_per_access_by_stride_elems_8B\": " + t8 + ", \"cycles_per_access_by_stride_elems_16B\": " + t16 + ", \"penalty_8B_stride16_over_stride1\": " + std::to_string(base8[4] / base8[0]) + ", \"penalty_16B_stride8_over_stride1\": " + std::to_string(base16[3] / base16[0]) + ", \"method\": \"256 threads, 8 B and 16 B shared loads at element strides 1..32 (a 64-bit access that pays no penalty at stride 1 and doubles per stride step from 2 elements implies 8-byte banks / the half-warp rule)\"}";
    emit_clocked("smem_bank_map", "cycles", std::vector<double>{base8[0]}, det.c_str(), mhz); }
  // 7.
  { CK(cudaMemset(counts, 0, 64)); int n = std::min(trials, 2000); std::vector<double> sp; for (int t = 0; t < n; t++) { its_probe<<<1, 32>>>(fs, counts); CK(cudaDeviceSynchronize()); long long h; CK(cudaMemcpy(&h, fs + 64, 8, cudaMemcpyDeviceToHost)); sp.push_back((double)h); }
    int h; CK(cudaMemcpy(&h, counts, 4, cudaMemcpyDeviceToHost)); std::sort(sp.begin(), sp.end());
    char det[2048]; snprintf(det, sizeof det, "{\"trials\": %d, \"interleaved_observed\": %d, \"syncwarp_reconverge_spread_cycles_p50\": %.0f, \"spread_p99\": %.0f, \"method\": \"odd lanes spin 2000 cycles in one branch, even lanes record their start in the other: interleaved if an even lane's start precedes the odd branch's end; spread of clock64 across lanes after __syncwarp\"}", n, h, sp[sp.size() / 2], sp[sp.size() * 99 / 100]);
    emit_clocked("independent_thread_scheduling", "bool", std::vector<double>{h > 0 ? 1.0 : 0.0}, det, mhz); }
  return 0;
}
