// B32: the operand path, ldmatrix.x4 -> mma.sync (round 2, group 1). A 64 x 32 bf16 A-tile and 32 x 64 B-tile in shared memory, swizzled
// (XOR of the 16 B column index with the row's low bits) or not. (a) dependent: each mma waits for its own ldmatrix and the next ldmatrix
// address depends on the mma result; (b) pipelined: fragments for step k + D loaded while step k computes, D in {1, 2, 3}; (c) unswizzled;
// (d) 1, 2, 4 blocks per SM. One warp per sub-partition per block (128 threads), 16 k-steps per trip.
#include "common.cuh"
#include <cuda_bf16.h>
#define MMA_BF16 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
__device__ __forceinline__ unsigned swz(unsigned row, unsigned col16, int SW) { return SW ? (col16 ^ (row & 7)) : col16; }   // 16 B column index swizzle
__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr)); }
__device__ __forceinline__ void fill(char* sm, int n) { for (int i = threadIdx.x; i < n / 2; i += blockDim.x) ((unsigned short*)sm)[i] = (unsigned short)(0x3f80 + (i & 7)); }
// A tile: 64 rows x 32 bf16 (64 B per row = 4 x 16 B columns); B tile: 64 rows (n) x 32 bf16 (k), same layout. Warp w owns A rows 16w..16w+15.
template <int SW, int DEPTH> __global__ void __launch_bounds__(128) pipelined(float* out, int trips, long long* cyc) {
  extern __shared__ __align__(128) char sm[]; char* At = sm; char* Bt = sm + 64 * 64; fill(sm, 2 * 64 * 64); __syncthreads();
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; unsigned abase = (unsigned)__cvta_generic_to_shared(At), bbase = (unsigned)__cvta_generic_to_shared(Bt);
  float c[4][4] = {}; asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) {
    unsigned fa[DEPTH + 1][4], fb[DEPTH + 1][2];
    auto load = [&](int k, int slot) {   // k-step k: A columns 2k, 2k+1 (16 B each) of rows 16w + (lane % 16); B columns of rows (n) lane % 8 + 8 * (lane / 16)
      unsigned arow = 16 * warp + (lane & 15), acol = 2 * (k & 1) + (lane >> 4); ldsm_x4(abase + arow * 64 + swz(arow, acol, SW) * 16, fa[slot][0], fa[slot][1], fa[slot][2], fa[slot][3]);
      unsigned brow = (lane & 7) + 8 * ((lane >> 3) & 1) + 16 * (k & 3) % 64, bcol = 2 * (k & 1) + ((lane >> 4) & 1); unsigned x0, x1, x2, x3; ldsm_x4(bbase + brow * 64 + swz(brow, bcol, SW) * 16, x0, x1, x2, x3); fb[slot][0] = x0; fb[slot][1] = x1; (void)x2; (void)x3; };
#pragma unroll
    for (int d = 0; d < DEPTH; d++) load(d, d);
#pragma unroll
    for (int k = 0; k < 16; k++) { int cur = k % (DEPTH + 1); if (k + DEPTH < 16) load(k + DEPTH, (k + DEPTH) % (DEPTH + 1));
#pragma unroll
      for (int j = 0; j < 4; j++) asm volatile(MMA_BF16 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(fa[cur][0]), "r"(fa[cur][1]), "r"(fa[cur][2]), "r"(fa[cur][3]), "r"(fb[cur][0]), "r"(fb[cur][1])); } }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float s = 0; for (int j = 0; j < 4; j++) s += c[j][0]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
template <int SW> __global__ void __launch_bounds__(128) dependent(float* out, int trips, long long* cyc, int mask) {   // mask is 0 at run time: the address depends on the accumulator without changing
  // chain: ldmatrix -> mma -> the next ldmatrix address depends on the accumulator (through an integer conversion that is zero at run time)
  extern __shared__ __align__(128) char sm[]; fill(sm, 2 * 64 * 64); __syncthreads(); int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; unsigned abase = (unsigned)__cvta_generic_to_shared(sm);
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0; unsigned dep = 0; unsigned b0 = 0x3f803f80u; asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) { unsigned arow = 16 * warp + (lane & 15), acol = (lane >> 4) + (dep & 1); unsigned f0, f1, f2, f3; ldsm_x4(abase + arow * 64 + swz(arow, acol, SW) * 16, f0, f1, f2, f3);
    asm volatile(MMA_BF16 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(f0), "r"(f1), "r"(f2), "r"(f3), "r"(b0), "r"(b0)); dep = __float_as_uint(c0) & (unsigned)mask; }
  long long t1 = clock64(); out[blockIdx.x * blockDim.x + threadIdx.x] = c0 + c1 + c2 + c3; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
__global__ void __launch_bounds__(128) ldsm_only(float* out, int trips, long long* cyc, int x) {   // ldmatrix.x1/.x2/.x4 latency: each result is consumed (XOR) before the next load; wavefronts via ncu
  extern __shared__ __align__(128) char sm[]; fill(sm, 2 * 64 * 64); __syncthreads(); int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; unsigned abase = (unsigned)__cvta_generic_to_shared(sm); unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0; asm volatile("" : "+r"(trips)); long long t0 = clock64();
  for (int t = 0; t < trips; t++) { unsigned arow = 16 * warp + (lane & 15), acol = ((lane >> 4) + t) & 3; unsigned a = abase + arow * 64 + (acol ^ (arow & 7)) * 16; unsigned x0, x1, x2, x3;
    if (x == 1) { asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(x0) : "r"(a)); r0 ^= x0; } else if (x == 2) { asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(x0), "=r"(x1) : "r"(a)); r0 ^= x0; r1 ^= x1; } else { ldsm_x4(a, x0, x1, x2, x3); r0 ^= x0; r1 ^= x1; r2 ^= x2; r3 ^= x3; } }
  asm volatile("" : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3)); long long t1 = clock64(); if ((r0 ^ r1 ^ r2 ^ r3) == 0x1234567u) out[0] = 1; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); float* out; CK(cudaMalloc(&out, (size_t)sms * 4 * 128 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  auto run = [&](auto kern, const char* name, int bps, int steps_per_trip, bool emit, int extra = -1) { int pad = std::max(8192, (96 * 1024) / bps); int grid = sms * bps; int trips = 400; CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    auto go = [&]() { if (extra >= 0) ((void (*)(float*, int, long long*, int))kern)<<<grid, 128, pad>>>(out, trips, cyc, extra); else ((void (*)(float*, int, long long*))kern)<<<grid, 128, pad>>>(out, trips, cyc); CK(cudaGetLastError()); };
    go(); CK(cudaDeviceSynchronize()); if (probe) { go(); CK(cudaDeviceSynchronize()); return 0.0; }
    std::vector<double> s; for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(grid); CK(cudaMemcpy(h.data(), cyc, grid * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[grid / 2] / (trips * (double)steps_per_trip)); }
    std::sort(s.begin(), s.end()); if (emit) { char q[96], d[160]; snprintf(q, sizeof q, "ldmatrix_mma_cycles_per_step[%s,bps=%d]", name, bps); snprintf(d, sizeof d, "{\"variant\": \"%s\", \"blocks_per_sm\": %d, \"mma_per_step_per_warp\": 4, \"ldmatrix_per_step_per_warp\": 2}", name, bps); emit_clocked(q, "cycles", s, d, mhz); } return s[s.size() / 2]; };
  if (probe) { if (!strcmp(probe, "pipelined")) run(pipelined<1, 2>, "swizzled_depth2", 1, 16, false); else if (!strcmp(probe, "unswizzled")) run(pipelined<0, 2>, "unswizzled_depth2", 1, 16, false); else run(ldsm_only, "ldsm", 1, 1, false, atoi(probe)); return 0; }
  for (int bps : {1, 2, 4}) { run(pipelined<1, 1>, "swizzled_depth1", bps, 16, true); run(pipelined<1, 2>, "swizzled_depth2", bps, 16, true); run(pipelined<1, 3>, "swizzled_depth3", bps, 16, true); run(pipelined<0, 2>, "unswizzled_depth2", bps, 16, true); }
  { double c = run(dependent<1>, "dependent_swizzled", 1, 1, true, 0); double u = run(dependent<0>, "dependent_unswizzled", 1, 1, true, 0); (void)c; (void)u; }
  for (int x : {1, 2, 4}) { double c = run(ldsm_only, "x", 1, 1, false, x); char q[64], d[256]; snprintf(q, sizeof q, "ldmatrix_latency_cycles[x%d]", x); snprintf(d, sizeof d, "{\"x\": %d, \"warps_per_sm\": 4, \"method\": \"one warp per sub-partition: ldmatrix.x%d, its result XORed into an accumulator, then the next; cycles per link = the ldmatrix latency plus the XOR and loop (about 4 cycles)\"}", x, x); emit_clocked(q, "cycles", std::vector<double>{c}, d, mhz); }
  return 0;
}
