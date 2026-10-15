// B33: the tensor path and the LSU queue (round 2, group 1). Tensor blocks run B32's pipelined loop (swizzled, depth 2); load blocks on
// the same SM run (a) LDS.128 loops, (b) cp.async loops, (c) both. Mixes 1+1, 2+2, 3+3 blocks per SM (block role by blockIdx parity).
// Per block: cycles per k-step (tensor blocks) or per step (load blocks) from clock64. One warp per sub-partition per block.
#include "common.cuh"
#include <cuda_bf16.h>
#define MMA_BF16 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr)); }
template <int ROLE> __global__ void __launch_bounds__(128) mixed(const char* __restrict__ g, float* out, int trips, long long* cyc, int role_of_odd, int sms, int* smid_out) {
  // ROLE: 0 tensor loop; 1 LDS.128 loop; 2 cp.async loop; 3 both loads. The block scheduler places consecutive blocks on consecutive SMs
  // (B16), so blocks b and b + sms share an SM: waves alternate roles (even waves tensor, odd waves role_of_odd). Parity of blockIdx
  // would put two blocks of the same role on one SM. %smid is recorded so the host can check the mix.
  extern __shared__ __align__(128) char sm[]; for (int i = threadIdx.x; i < 4096; i += blockDim.x) ((unsigned short*)sm)[i] = 0x3f80 + (i & 7); __syncthreads();
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; unsigned base = (unsigned)__cvta_generic_to_shared(sm); int role = ((blockIdx.x / sms) & 1) ? role_of_odd : 0;
  if (threadIdx.x == 0) { unsigned id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id)); smid_out[blockIdx.x] = (int)id; }
  float c[4][4] = {}; float acc = 0; asm volatile("" : "+r"(trips)); long long t0 = clock64();
  if (role == 0) {
    for (int t = 0; t < trips; t++) { unsigned fa[3][4], fb[3][2];
      auto load = [&](int k, int slot) { unsigned arow = 16 * warp + (lane & 15), acol = 2 * (k & 1) + (lane >> 4); ldsm_x4(base + arow * 64 + ((acol ^ (arow & 7)) * 16), fa[slot][0], fa[slot][1], fa[slot][2], fa[slot][3]);
        unsigned brow = (lane & 7) + 8 * ((lane >> 3) & 1) + 16 * (k & 3), bcol = 2 * (k & 1) + ((lane >> 4) & 1); unsigned x0, x1, x2, x3; ldsm_x4(base + 4096 + brow * 64 + ((bcol ^ (brow & 7)) * 16), x0, x1, x2, x3); fb[slot][0] = x0; fb[slot][1] = x1; };
      load(0, 0); load(1, 1);
#pragma unroll
      for (int k = 0; k < 16; k++) { int cur = k % 3; if (k + 2 < 16) load(k + 2, (k + 2) % 3);
#pragma unroll
        for (int j = 0; j < 4; j++) asm volatile(MMA_BF16 : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(fa[cur][0]), "r"(fa[cur][1]), "r"(fa[cur][2]), "r"(fa[cur][3]), "r"(fb[cur][0]), "r"(fb[cur][1])); } } }
  else {
    for (int t = 0; t < trips; t++) {
#pragma unroll
      for (int k = 0; k < 16; k++) {
        if (role & 1) { float4 v; unsigned a = base + ((lane * 16 + k * 512) & 8191); asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); acc += v.x; }
        if (role & 2) { unsigned dst = base + 8192 + ((k & 3) * 2048 + threadIdx.x * 16); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(g + ((t * 16 + k) & 255) * 2048 + threadIdx.x * 16)); if ((k & 3) == 3) asm volatile("cp.async.commit_group;\n cp.async.wait_group 1;\n" ::: "memory"); } }
      asm volatile("" : "+f"(acc)); } }
  long long t1 = clock64(); float s = acc; for (int j = 0; j < 4; j++) s += c[j][0]; out[blockIdx.x * blockDim.x + threadIdx.x] = s; if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(); float* out; CK(cudaMalloc(&out, (size_t)sms * 8 * 128 * 4)); long long* cyc; CK(cudaMalloc(&cyc, 8 * 4096)); char* g; CK(cudaMalloc(&g, 1 << 20)); CK(cudaMemset(g, 1, 1 << 20));
  CK(cudaFuncSetAttribute(mixed<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); const char* roles[] = {"tensor_only", "lds128", "cp_async", "both"}; int* smid; CK(cudaMalloc(&smid, 4 * 4096));
  const char* probe = nullptr; for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--probe")) probe = argv[i + 1];
  for (int role : {0, 1, 2, 3}) for (int pairs : {1, 2, 3}) { int bps = role == 0 ? pairs : 2 * pairs; int pad = std::max(16384, (96 * 1024) / bps); int grid = sms * bps; int trips = 200;
    if (probe && (pairs != 1 || strcmp(probe, roles[role]))) continue;
    auto go = [&]() { mixed<0><<<grid, 128, pad>>>(g, out, trips, cyc, role, sms, smid); CK(cudaGetLastError()); }; go(); CK(cudaDeviceSynchronize());
    if (probe) { go(); CK(cudaDeviceSynchronize()); continue; }
    std::vector<double> st, sl; int mixed_sms = 0;
    for (int r = 0; r < reps; r++) { go(); CK(cudaDeviceSynchronize()); std::vector<long long> h(grid); CK(cudaMemcpy(h.data(), cyc, grid * 8, cudaMemcpyDeviceToHost)); std::vector<int> id(grid); CK(cudaMemcpy(id.data(), smid, grid * 4, cudaMemcpyDeviceToHost));
      std::vector<long long> tb, lb; for (int b = 0; b < grid; b++) (((b / sms) & 1) && role ? lb : tb).push_back(h[b]); std::sort(tb.begin(), tb.end()); st.push_back((double)tb[tb.size() / 2] / (trips * 16.0)); if (!lb.empty()) { std::sort(lb.begin(), lb.end()); sl.push_back((double)lb[lb.size() / 2] / (trips * 16.0)); }
      if (r == 0 && role) { std::vector<int> t(sms, 0), l(sms, 0); for (int b = 0; b < grid; b++) (((b / sms) & 1) ? l : t)[id[b]]++; for (int i = 0; i < sms; i++) mixed_sms += (t[i] > 0 && l[i] > 0); } }
    std::sort(st.begin(), st.end()); char q[96], d[240]; snprintf(q, sizeof q, "tensor_under_load[load=%s,tensor_blocks=%d]", roles[role], pairs); snprintf(d, sizeof d, "{\"load\": \"%s\", \"tensor_blocks_per_sm\": %d, \"load_blocks_per_sm\": %d, \"mma_per_step_per_warp\": 4, \"ldmatrix_per_step_per_warp\": 2, \"sms_with_both_roles\": %d, \"sms\": %d}", roles[role], pairs, role ? pairs : 0, mixed_sms, sms); emit_clocked(q, "cycles", st, d, mhz);
    if (!sl.empty()) { std::sort(sl.begin(), sl.end()); snprintf(q, sizeof q, "load_block_under_tensor[load=%s,tensor_blocks=%d]", roles[role], pairs); emit_clocked(q, "cycles", sl, d, mhz); } }
  return 0;
}
