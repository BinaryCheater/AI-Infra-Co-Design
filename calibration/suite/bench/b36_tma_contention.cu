// B36: TMA with cp.async and with shared loads on the same SM (round 2, group 2). Roles by wave (consecutive blocks land on
// consecutive SMs, B16; %smid recorded). (a) TMA and cp.async pipelined 8 KB streams in ratios 1:1, 1:3, 3:1 at 4 resident blocks,
// chip bytes/s and each role's cycles per tile; (b) TMA blocks beside LDS.128 loop blocks, 2 + 2; (c) 2-D tensor-map copies vs 1-D of
// the same bytes, one block per SM; (d) TMA store (shared to global) beside the TMA load stream.
#include "tma_common.cuh"
#define TILE 8192
__device__ __forceinline__ void tma_stream(const char* src, int iters, long long* cyc, int bidx) {
  extern __shared__ __align__(128) char sm[]; __shared__ unsigned long long bar[2]; unsigned b0 = saddr(&bar[0]), b1 = saddr(&bar[1]);
  if (threadIdx.x == 0) { mbar_init(b0, 1); mbar_init(b1, 1); } __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) { const char* base = src + (size_t)((i + bidx) & 63) * TILE; unsigned dst = saddr(sm + (i & 1) * TILE), b = (i & 1) ? b1 : b0;
    if (threadIdx.x == 0) { mbar_expect_tx(b, TILE); tma_load_1d(dst, base, TILE, b); } mbar_wait(b, (i >> 1) & 1); __syncthreads(); }
  if (threadIdx.x == 0) cyc[bidx] = (clock64() - t0) / iters;
}
__device__ __forceinline__ void cp_stream(const char* src, int iters, long long* cyc, int bidx) {
  extern __shared__ __align__(128) char sm[]; int per = TILE / 16 / blockDim.x; long long t0 = clock64();
  for (int i = 0; i < iters; i++) { const char* base = src + (size_t)((i + bidx) & 63) * TILE; unsigned dst = saddr(sm + (i & 1) * TILE);
    for (int c = 0; c < per; c++) { int off = (c * blockDim.x + threadIdx.x) * 16; cp_async16(dst + off, base + off); } asm volatile("cp.async.commit_group;\n cp.async.wait_group 1;\n" ::: "memory"); __syncthreads(); }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory"); if (threadIdx.x == 0) cyc[bidx] = (clock64() - t0) / iters;
}
__device__ __forceinline__ void lds_stream(int iters, long long* cyc, int bidx, float* out) {
  extern __shared__ __align__(128) char sm[]; unsigned base = saddr(sm); float acc = 0; int lane = threadIdx.x & 31; long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int k = 0; k < 16; k++) { float4 v; unsigned a = base + ((lane * 16 + k * 512 + i * 16) & 8191); asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); acc += v.x; } asm volatile("" : "+f"(acc)); }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc; if (threadIdx.x == 0) cyc[bidx] = (clock64() - t0) / (iters * 16);
}
__device__ __forceinline__ void tma_store_stream(char* dst_g, int iters, long long* cyc, int bidx) {   // TMA store of the block's 8 KB tile, back to back
  extern __shared__ __align__(128) char sm[]; long long t0 = clock64();
  for (int i = 0; i < iters; i++) { if (threadIdx.x == 0) { tma_store_1d(dst_g + (size_t)((i + bidx) & 63) * TILE, saddr(sm), TILE); tma_store_wait(); } __syncthreads(); }
  if (threadIdx.x == 0) cyc[bidx] = (clock64() - t0) / iters;
}
__device__ __forceinline__ void tma_2d_stream(const CUtensorMap* map, int iters, long long* cyc, int bidx) {   // 64 x 32 fp32 boxes = 8 KB
  extern __shared__ __align__(128) char sm[]; __shared__ unsigned long long bar[2]; unsigned b0 = saddr(&bar[0]), b1 = saddr(&bar[1]);
  if (threadIdx.x == 0) { mbar_init(b0, 1); mbar_init(b1, 1); } __syncthreads(); long long t0 = clock64();
  for (int i = 0; i < iters; i++) { int x = ((i + bidx) & 31) * 32, y = (((i + bidx) >> 5) & 15) * 64; unsigned dst = saddr(sm + (i & 1) * TILE), b = (i & 1) ? b1 : b0;
    if (threadIdx.x == 0) { mbar_expect_tx(b, TILE); tma_load_2d(dst, map, x, y, b); } mbar_wait(b, (i >> 1) & 1); __syncthreads(); }
  if (threadIdx.x == 0) cyc[bidx] = (clock64() - t0) / iters;
}
// role per wave: roles[wave & 3]; 0 tma load, 1 cp.async, 2 lds, 3 tma store, 4 tma 2-d, 5 idle (exit)
__global__ void __launch_bounds__(256) mixed(const char* src, char* dst_g, const CUtensorMap* map, float* out, long long* cyc, int* smid, int iters, int sms, int r0, int r1, int r2, int r3) {
  int wave = blockIdx.x / sms; int role = (wave & 3) == 0 ? r0 : (wave & 3) == 1 ? r1 : (wave & 3) == 2 ? r2 : r3;
  if (threadIdx.x == 0) { unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id)); smid[blockIdx.x] = (int)id; }
  if (role == 0) tma_stream(src, iters, cyc, blockIdx.x); else if (role == 1) cp_stream(src, iters, cyc, blockIdx.x); else if (role == 2) lds_stream(iters, cyc, blockIdx.x, out);
  else if (role == 3) tma_store_stream(dst_g, iters, cyc, blockIdx.x); else if (role == 4) tma_2d_stream(map, iters, cyc, blockIdx.x); else if (threadIdx.x == 0) cyc[blockIdx.x] = 0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6;
  char *src, *dst_g; CK(cudaMalloc(&src, 64 * 65536)); CK(cudaMemset(src, 7, 64 * 65536)); CK(cudaMalloc(&dst_g, 64 * 65536)); float* out; CK(cudaMalloc(&out, (size_t)sms * 4 * 256 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096)); int* smid; CK(cudaMalloc(&smid, 4 * 4096));
  float* mat; CK(cudaMalloc(&mat, 1024 * 1024 * 4)); CK(cudaMemset(mat, 0, 1024 * 1024 * 4)); CUtensorMap map; bool have_map = make_map_2d(&map, mat, 1024, 1024, 64, 32); CUtensorMap* dmap; CK(cudaMalloc(&dmap, sizeof map)); CK(cudaMemcpy(dmap, &map, sizeof map, cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(mixed, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  const char* rname[] = {"tma", "cp_async", "lds128", "tma_store", "tma_2d", "idle"};
  // one experiment: 4 waves (4 resident blocks per SM, 24 KB pad), roles per wave; emit per-role cycles per tile and the chip's copy bytes/s
  auto run = [&](const char* label, int r0, int r1, int r2, int r3, const char* what) {
    int waves = 4, grid = sms * waves, iters = 400, pad = 24 * 1024; auto go = [&]() { mixed<<<grid, 256, pad>>>(src, dst_g, dmap, out, cyc, smid, iters, sms, r0, r1, r2, r3); CK(cudaGetLastError()); };
    go(); CK(cudaDeviceSynchronize()); if (probe) { go(); CK(cudaDeviceSynchronize()); return; }
    int roles[4] = {r0, r1, r2, r3}; std::vector<double> per_role[6]; std::vector<double> bw; Timer t; int mixed_sms = 0;
    for (int r = 0; r < reps; r++) { t.start(); go(); double ms = t.stop_ms(); std::vector<long long> h(grid); CK(cudaMemcpy(h.data(), cyc, grid * 8, cudaMemcpyDeviceToHost)); std::vector<int> id(grid); CK(cudaMemcpy(id.data(), smid, grid * 4, cudaMemcpyDeviceToHost));
      long long copy_bytes = 0; for (int w = 0; w < waves; w++) { int role = roles[w]; std::vector<long long> v(h.begin() + w * sms, h.begin() + (w + 1) * sms); std::sort(v.begin(), v.end()); per_role[role].push_back((double)v[sms / 2]); if (role == 0 || role == 1 || role == 3 || role == 4) copy_bytes += (long long)TILE * iters * sms; }
      bw.push_back(copy_bytes / (ms * 1e-3));
      if (r == 0) { std::vector<int> cnt(sms, 0); for (int b = 0; b < grid; b++) cnt[id[b]]++; for (int i = 0; i < sms; i++) mixed_sms += cnt[i] == waves; } }
    std::sort(bw.begin(), bw.end()); char q[96], d[400];
    snprintf(q, sizeof q, "tma_mix_bytes_per_s[%s]", label); snprintf(d, sizeof d, "{\"roles_by_wave\": [\"%s\",\"%s\",\"%s\",\"%s\"], \"tile_bytes\": %d, \"resident_blocks_per_sm\": 4, \"sms_with_all_four_blocks\": %d, \"what\": \"%s\"}", rname[r0], rname[r1], rname[r2], rname[r3], TILE, mixed_sms, what); emit_clocked(q, "byte/s", bw, d, mhz);
    for (int role = 0; role < 6; role++) if (!per_role[role].empty()) { snprintf(q, sizeof q, "tma_mix_cycles_per_tile[%s,role=%s]", label, rname[role]); emit_clocked(q, "cycles", per_role[role], d, mhz); } };
  if (probe) { if (!strcmp(probe, "1to1")) run("tma1_cp1", 0, 1, 0, 1, ""); else if (!strcmp(probe, "tma_lds")) run("tma2_lds2", 0, 2, 0, 2, ""); else run("tma4", 0, 0, 0, 0, ""); return 0; }
  run("tma4", 0, 0, 0, 0, "(a) four TMA streams per SM: the reference"); run("cp4", 1, 1, 1, 1, "(a) four cp.async streams per SM: the reference");
  run("tma1_cp1", 0, 1, 0, 1, "(a) TMA : cp.async = 1 : 1 at four resident"); run("tma1_cp3", 0, 1, 1, 1, "(a) 1 : 3"); run("tma3_cp1", 0, 0, 0, 1, "(a) 3 : 1");
  run("tma2_idle2", 0, 5, 0, 5, "(b) two TMA streams alone"); run("tma2_lds2", 0, 2, 0, 2, "(b) two TMA streams beside two LDS.128 loop blocks"); run("lds2_idle2", 2, 5, 2, 5, "(b) the LDS.128 blocks alone");
  if (have_map) { run("tma1d_1", 0, 5, 5, 5, "(c) one 1-D TMA stream per SM"); run("tma2d_1", 4, 5, 5, 5, "(c) one 2-D tensor-map stream per SM, 64 x 32 fp32 boxes = 8 KB"); }
  run("tmastore2_idle2", 3, 5, 3, 5, "(d) two TMA store streams alone"); run("tma2_store2", 0, 3, 0, 3, "(d) two TMA load streams beside two TMA store streams");
  return 0;
}
