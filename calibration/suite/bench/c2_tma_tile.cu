// C2: the group-2 composite (round 2): the T5 GEMM tile (128 x 128, FP32 FFMA, 256 threads in 16 x 16, 8 x 8 accumulators per thread)
// with its tiles brought in by 2-D TMA (tensor maps, one mbarrier per stage) instead of cp.async, double-buffered, leading and
// trailing barriers as the template generates them. TK = 16 (16 KB per stage, 32 KB per block) at 1, 2, 3 blocks per SM and TK = 32
// (64 KB per block) at 1. A is [M][K] fp32 row-major, B is [K][N]; 8 x 8 tiles over K = 1024, L2 resident. Cycles per k-step per block.
#include "tma_common.cuh"
#define KDIM 1024
#define NDIM 1024
template <int TK, int VARIANT> __global__ void __launch_bounds__(256) tma_tile(const CUtensorMap* mapA, const CUtensorMap* mapB, float* __restrict__ C, int passes, long long* cyc) {
  // VARIANT 0: the kernel; 1: copies only; 2: compute only (tiles filled once)
  constexpr int TM = 128, TN = 128, TX = 16, TY = 16, RM = TM / TY, RN = TN / TX, STAGE = (TM * TK + TK * TN) * 4;
  extern __shared__ __align__(128) char sm[]; __shared__ unsigned long long bar[2]; unsigned b0 = saddr(&bar[0]), b1 = saddr(&bar[1]);
  if (threadIdx.x == 0) { mbar_init(b0, 1); mbar_init(b1, 1); } __syncthreads();
  int tile = blockIdx.x & 63; int rowA = (tile & 7) * TM, colB = (tile >> 3) * TN; int tx = threadIdx.x % TX, ty = threadIdx.x / TX; const int steps = KDIM / TK;
  float acc[RM][RN]; for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) acc[i][j] = 0.f;
  auto load = [&](int s, int slot) { if (threadIdx.x == 0) { unsigned b = slot ? b1 : b0, dst = saddr(sm + slot * STAGE); mbar_expect_tx(b, STAGE); tma_load_2d(dst, mapA, s * TK, rowA, b); tma_load_2d(dst + TM * TK * 4, mapB, colB, s * TK, b); } };
  if (VARIANT == 2) { load(0, 0); load(1, 1); mbar_wait(b0, 0); mbar_wait(b1, 0); __syncthreads(); }
  int phase_count[2] = {0, 0}; asm volatile("" : "+r"(passes)); long long t0 = clock64();
  for (int p = 0; p < passes; p++) {
    if (VARIANT != 2) load(0, 0);
    for (int s = 0; s < steps; s++) { int slot = s & 1;
      if (VARIANT != 2) { if (s + 1 < steps) load(s + 1, slot ^ 1); mbar_wait(slot ? b1 : b0, phase_count[slot] & 1); phase_count[slot]++; }
      __syncthreads();                                                   // leading barrier
      if (VARIANT != 1) { const float* a = (const float*)(sm + slot * STAGE); const float* bt = a + TM * TK;
#pragma unroll 4
        for (int kk = 0; kk < TK; kk++) { float af[RM], bf[RN];
          for (int i = 0; i < RM; i++) af[i] = a[(ty + i * TY) * TK + kk];
          for (int j = 0; j < RN; j++) bf[j] = bt[kk * TN + tx + j * TX];
          for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) acc[i][j] = fmaf(af[i], bf[j], acc[i][j]); } }
      __syncthreads();                                                   // trailing barrier
    } }
  long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  float* Ct = C + (size_t)blockIdx.x * TM * TN; for (int i = 0; i < RM; i++) for (int j = 0; j < RN; j++) Ct[(ty + i * TY) * TN + tx + j * TX] = acc[i][j];
  if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz();
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  float *A, *B, *C; CK(cudaMalloc(&A, (size_t)1024 * KDIM * 4)); CK(cudaMalloc(&B, (size_t)KDIM * NDIM * 4)); CK(cudaMemset(A, 0, (size_t)1024 * KDIM * 4)); CK(cudaMemset(B, 0, (size_t)KDIM * NDIM * 4)); CK(cudaMalloc(&C, (size_t)sms * 4 * 128 * 128 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096));
  CUtensorMap mA16, mB16, mA32, mB32; bool ok = make_map_2d(&mA16, A, 1024, KDIM, 128, 16) && make_map_2d(&mB16, B, KDIM, NDIM, 16, 128) && make_map_2d(&mA32, A, 1024, KDIM, 128, 32) && make_map_2d(&mB32, B, KDIM, NDIM, 32, 128);
  if (!ok) { printf("{\"quantity\": \"tma_tile_cycles_per_step[tk=16,bps=1]\", \"unit\": \"cycles\", \"samples\": [], \"detail\": {\"measured\": false, \"reason\": \"cuTensorMapEncodeTiled failed\"}}\n"); return 0; }
  CUtensorMap* d; CK(cudaMalloc(&d, 4 * sizeof(CUtensorMap))); CUtensorMap hm[4] = {mA16, mB16, mA32, mB32}; CK(cudaMemcpy(d, hm, sizeof hm, cudaMemcpyHostToDevice));
  const char* vname[] = {"", "copy_only_", "compute_only_"};
  auto run = [&](int tk, int bps, int variant, bool emit) { int stage = (128 * tk + tk * 128) * 4, smem = 2 * stage; int grid = sms * bps, passes = tk == 16 ? 4 : 2; cudaFuncAttributes fa; int occ = 0;
    auto launch = [&]() { if (tk == 16) { if (variant == 0) tma_tile<16, 0><<<grid, 256, smem>>>(d, d + 1, C, passes, cyc); else if (variant == 1) tma_tile<16, 1><<<grid, 256, smem>>>(d, d + 1, C, passes, cyc); else tma_tile<16, 2><<<grid, 256, smem>>>(d, d + 1, C, passes, cyc); }
      else { if (variant == 0) tma_tile<32, 0><<<grid, 256, smem>>>(d + 2, d + 3, C, passes, cyc); else if (variant == 1) tma_tile<32, 1><<<grid, 256, smem>>>(d + 2, d + 3, C, passes, cyc); else tma_tile<32, 2><<<grid, 256, smem>>>(d + 2, d + 3, C, passes, cyc); } CK(cudaGetLastError()); };
    if (tk == 16) { CK(cudaFuncSetAttribute(tma_tile<16, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncSetAttribute(tma_tile<16, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncSetAttribute(tma_tile<16, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncGetAttributes(&fa, tma_tile<16, 0>)); CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, tma_tile<16, 0>, 256, smem)); }
    else { CK(cudaFuncSetAttribute(tma_tile<32, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncSetAttribute(tma_tile<32, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncSetAttribute(tma_tile<32, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncGetAttributes(&fa, tma_tile<32, 0>)); CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, tma_tile<32, 0>, 256, smem)); }
    launch(); CK(cudaDeviceSynchronize()); if (probe) { launch(); CK(cudaDeviceSynchronize()); return; }
    std::vector<double> s; for (int r = 0; r < reps; r++) { launch(); CK(cudaDeviceSynchronize()); std::vector<long long> h(grid); CK(cudaMemcpy(h.data(), cyc, grid * 8, cudaMemcpyDeviceToHost)); std::sort(h.begin(), h.end()); s.push_back((double)h[grid / 2] / (passes * (double)(KDIM / tk))); }
    if (!emit) return; char q[80], dd[400]; snprintf(q, sizeof q, "tma_tile_%scycles_per_step[tk=%d,bps=%d]", vname[variant], tk, bps);
    snprintf(dd, sizeof dd, "{\"tk\": %d, \"blocks_per_sm\": %d, \"resident_blocks_per_sm\": %d, \"occupancy_limit_blocks\": %d, \"stage_bytes\": %d, \"smem_bytes\": %d, \"regs\": %d, \"threads\": 256, \"RM\": 8, \"RN\": 8, \"design\": \"T5 128x128 FP32, S=2, P=1, TMA 2-D\"}", tk, bps, std::min(bps, occ), occ, stage, smem, fa.numRegs); emit_clocked(q, "cycles", s, dd, mhz); };
  if (probe) { int bps = atoi(probe); run(16, bps, 0, false); return 0; }
  for (int bps : {1, 2, 3}) for (int v : {0, 1, 2}) run(16, bps, v, true);
  for (int v : {0, 1, 2}) run(32, 1, v, true);
  return 0;
}
