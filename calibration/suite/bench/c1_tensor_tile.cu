// C1: the group-1 composite (round 2): a BF16 tensor GEMM tile kernel, 128 x 128 x 32, ldmatrix.x4 + mma.sync m16n8k16, swizzled
// shared tiles, double-buffered through cp.async, at 1, 2, 3 blocks per SM (4 x 32 KB does not fit the 100 KB SM). Eight warps in a
// 2 x 4 grid, warp tile 64 x 32: per k-step per warp 12 ldmatrix.x4 (8 A, 4 B) and 32 mma, 64 accumulators per thread. A is [M][K] bf16
// row-major, B is [N][K] (k contiguous), both resident in L2 (8 x 8 tiles over K = 1024: 4 MB each). Cycles per k-step per block from
// clock64 over 8 passes of the K loop. The model predicts this from B31-B34 with nothing fitted.
#include "common.cuh"
#include <cuda_bf16.h>
#define MMA_BF16 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#define TK 32
#define KDIM 1024
#define STAGE_BYTES (2 * 128 * TK * 2)      // A tile + B tile, 8 KB each
__device__ __forceinline__ unsigned swz(unsigned row, unsigned col16) { return col16 ^ ((row >> 1) & 3); }   // 64 B rows: rows 2q, 2q+1 share a 128 B line; 8 rows hit 8 distinct 16 B slots
__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr)); }
__device__ __forceinline__ void cp16(unsigned dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void load_stage(unsigned sbase, const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int k0) {
  // 128 rows x 4 chunks (16 B) per operand: 512 chunks per operand, 2 per thread each
  for (int c = threadIdx.x; c < 512; c += 256) { unsigned row = c >> 2, col = c & 3; cp16(sbase + row * 64 + swz(row, col) * 16, A + (size_t)row * KDIM + k0 + col * 8); cp16(sbase + 8192 + row * 64 + swz(row, col) * 16, B + (size_t)row * KDIM + k0 + col * 8); }
  asm volatile("cp.async.commit_group;\n");
}
// VARIANT 0: the kernel; 1: copies only (no ldmatrix, no mma: the stage copy floor per step); 2: no copies (ldmatrix + mma from a tile filled once: the compute floor)
template <int VARIANT> __global__ void __launch_bounds__(256) tensor_tile(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, float* __restrict__ C, int passes, long long* cyc) {
  extern __shared__ __align__(128) char sm[]; unsigned base = (unsigned)__cvta_generic_to_shared(sm);
  int tile = blockIdx.x & 63; const __nv_bfloat16* At = A + (size_t)(tile & 7) * 128 * KDIM; const __nv_bfloat16* Bt = B + (size_t)(tile >> 3) * 128 * KDIM;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp & 1, wn = warp >> 1;   // warp tile rows 64*wm.., cols 32*wn..
  float c[4][4][4]; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) c[i][j][q] = 0.f;
  const int steps = KDIM / TK; if (VARIANT == 2) { load_stage(base, At, Bt, 0); load_stage(base + STAGE_BYTES, At, Bt, TK); asm volatile("cp.async.wait_group 0;\n" ::: "memory"); __syncthreads(); }
  asm volatile("" : "+r"(passes)); long long t0 = clock64();
  for (int p = 0; p < passes; p++) {
    if (VARIANT != 2) load_stage(base, At, Bt, 0);
    for (int s = 0; s < steps; s++) {
      unsigned cur = base + (s & 1) * STAGE_BYTES;
      if (VARIANT != 2) { if (s + 1 < steps) { load_stage(base + ((s + 1) & 1) * STAGE_BYTES, At, Bt, (s + 1) * TK); asm volatile("cp.async.wait_group 1;\n" ::: "memory"); } else asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
      __syncthreads();
      if (VARIANT != 1)
#pragma unroll
      for (int kk = 0; kk < TK / 16; kk++) {
        unsigned a[4][4], b[2][4];
#pragma unroll
        for (int i = 0; i < 4; i++) { unsigned row = 64 * wm + 16 * i + (lane & 15), col = 2 * kk + (lane >> 4); ldsm_x4(cur + row * 64 + swz(row, col) * 16, a[i][0], a[i][1], a[i][2], a[i][3]); }
#pragma unroll
        for (int j = 0; j < 2; j++) { unsigned row = 32 * wn + 16 * j + (lane & 15), col = 2 * kk + (lane >> 4); ldsm_x4(cur + 8192 + row * 64 + swz(row, col) * 16, b[j][0], b[j][1], b[j][2], b[j][3]); }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int j = 0; j < 4; j++) { unsigned b0 = b[j >> 1][(j & 1) * 1 + 0], b1 = b[j >> 1][(j & 1) * 1 + 2];
            asm volatile(MMA_BF16 : "+f"(c[i][j][0]), "+f"(c[i][j][1]), "+f"(c[i][j][2]), "+f"(c[i][j][3]) : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b0), "r"(b1)); } }
      __syncthreads();
    }
  }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  // the tile store (once per block): C[128 x 128] row-major, fragment layout of m16n8
  float* Ct = C + (size_t)blockIdx.x * 128 * 128;
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { int r = 64 * wm + 16 * i + (lane >> 2), cc = 32 * wn + 8 * j + 2 * (lane & 3); Ct[r * 128 + cc] = c[i][j][0]; Ct[r * 128 + cc + 1] = c[i][j][1]; Ct[(r + 8) * 128 + cc] = c[i][j][2]; Ct[(r + 8) * 128 + cc + 1] = c[i][j][3]; }
  if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz();
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  __nv_bfloat16 *A, *B; CK(cudaMalloc(&A, (size_t)8 * 128 * KDIM * 2)); CK(cudaMalloc(&B, (size_t)8 * 128 * KDIM * 2)); CK(cudaMemset(A, 0x3c, (size_t)8 * 128 * KDIM * 2)); CK(cudaMemset(B, 0x3c, (size_t)8 * 128 * KDIM * 2));
  float* C; CK(cudaMalloc(&C, (size_t)sms * 4 * 128 * 128 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  CK(cudaFuncSetAttribute(tensor_tile<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE_BYTES)); CK(cudaFuncSetAttribute(tensor_tile<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE_BYTES)); CK(cudaFuncSetAttribute(tensor_tile<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE_BYTES));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, tensor_tile<0>, 256, 2 * STAGE_BYTES)); cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, tensor_tile<0>));
  const char* vname[] = {"", "copy_only_", "no_copy_"};
  auto run = [&](int bps, bool emit, int variant = 0) { int grid = sms * bps, passes = 8; auto go = [&]() { if (variant == 1) tensor_tile<1><<<grid, 256, 2 * STAGE_BYTES>>>(A, B, C, passes, cyc); else if (variant == 2) tensor_tile<2><<<grid, 256, 2 * STAGE_BYTES>>>(A, B, C, passes, cyc); else tensor_tile<0><<<grid, 256, 2 * STAGE_BYTES>>>(A, B, C, passes, cyc); CK(cudaGetLastError()); };
    go(); CK(cudaDeviceSynchronize()); if (probe) { go(); CK(cudaDeviceSynchronize()); return; }
    std::vector<double> s; for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(grid); CK(cudaMemcpy(h.data(), cyc, grid * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[grid / 2] / (passes * (double)(KDIM / TK))); }
    if (!emit) return; char q[80], d[400]; snprintf(q, sizeof q, "tensor_tile_%scycles_per_step[bps=%d]", vname[variant], bps);
    snprintf(d, sizeof d, "{\"blocks_per_sm\": %d, \"resident_blocks_per_sm\": %d, \"occupancy_limit_blocks\": %d, \"warps_per_block\": 8, \"ldmatrix_x4_per_step_per_warp\": 12, \"mma_per_step_per_warp\": 32, \"cp_async_chunks_per_step_per_thread\": 4, \"smem_bytes\": %d, \"regs\": %d, \"k_steps_per_pass\": %d, \"tile\": \"128x128x32 bf16\"}", bps, std::min(bps, occ), occ, 2 * STAGE_BYTES, fa.numRegs, KDIM / TK);
    emit_clocked(q, "cycles", s, d, mhz); };
  if (probe) { run(atoi(probe), false); return 0; }
  for (int bps : {1, 2, 3}) { run(bps, true); run(bps, true, 1); run(bps, true, 2); }
  return 0;
}
