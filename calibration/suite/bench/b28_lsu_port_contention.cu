// B28: LSU and shared-memory ports shared across resident blocks (B27_plus.md §1). Each block runs the template's per-warp
// pattern per k-step (4 x ld.shared.128 + 8 x ld.shared.32 per thread, T5's widths) and one 16 B cp.async per thread per phase
// into a second buffer with wait_group 1 (phase = 8 k-steps). Residency b in {1,2,3,4,6} forced by dynamic-smem padding.
// Mixes: lds only, cp.async only, both. Per block: cycles per k-step; per SM: aggregate shared-memory bytes per cycle and cp.async bytes/s.
#include "common.cuh"
template <int MIX> __global__ void pattern(const char* __restrict__ src, int phases, long long* cyc, long long* t_in, float* out) {
  extern __shared__ __align__(16) char smem[]; float* tile = (float*)smem; char* buf = smem + 8192;     // 8 KB tile + 2 x 4 KB copy buffers
  for (int i = threadIdx.x; i < 2048; i += blockDim.x) tile[i] = i * 1e-4f; __syncthreads();
  float acc = 0; int lane = threadIdx.x & 31, warp = threadIdx.x >> 5; unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
  if (threadIdx.x == 0) t_in[blockIdx.x] = clock64();
  long long t0 = clock64();
  for (int p = 0; p < phases; p++) {
    if (MIX != 0) { unsigned dst = sbase + 8192 + (p & 1) * 4096 + threadIdx.x * 16; const char* g = src + ((size_t)(blockIdx.x * phases + p) * 4096 + threadIdx.x * 16) % (8u << 20);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n cp.async.commit_group;\n" :: "r"(dst), "l"(g)); }
    if (MIX != 1) {
#pragma unroll 1
      for (int k = 0; k < 8; k++) { float4 a0, a1, a2, a3; float b0, b1, b2, b3, b4, b5, b6, b7; unsigned ab = sbase + ((warp & 1) * 64 + k * 4) * 4 * 4; unsigned bb = sbase + (k * 128 + lane) * 4;
        asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a0.x), "=f"(a0.y), "=f"(a0.z), "=f"(a0.w) : "r"(ab)); asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a1.x), "=f"(a1.y), "=f"(a1.z), "=f"(a1.w) : "r"(ab + 1024));
        asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a2.x), "=f"(a2.y), "=f"(a2.z), "=f"(a2.w) : "r"(ab + 2048)); asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a3.x), "=f"(a3.y), "=f"(a3.z), "=f"(a3.w) : "r"(ab + 3072));
        asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b0) : "r"(bb)); asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b1) : "r"(bb + 128)); asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b2) : "r"(bb + 256)); asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b3) : "r"(bb + 384));
        asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b4) : "r"(bb + 512)); asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b5) : "r"(bb + 640)); asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b6) : "r"(bb + 768)); asm volatile("ld.shared.f32 %0, [%1];" : "=f"(b7) : "r"(bb + 896));
        acc += (a0.x + a1.y + a2.z + a3.w) * (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7); } }
    if (MIX != 0) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    __syncthreads(); }
  asm volatile("" : "+f"(acc)); long long t1 = clock64(); __shared__ long long smin, smax; if (threadIdx.x == 0) { smin = t0; smax = t1; } __syncthreads(); atomicMin(&smin, t0); atomicMax(&smax, t1); __syncthreads();
  if (acc == 123456.789f) out[0] = acc; if (threadIdx.x == 0) cyc[blockIdx.x] = smax - smin;
}
int main(int argc, char** argv) {
  int reps = argi(argc, argv, "--reps", 5), sms = sm_count(); double mhz = probe_clock_mhz(), hz = mhz * 1e6; char* src; CK(cudaMalloc(&src, 8 << 20)); CK(cudaMemset(src, 1, 8 << 20));
  long long *cyc, *tin; CK(cudaMalloc(&cyc, 8 * 2048)); CK(cudaMalloc(&tin, 8 * 2048)); float* out; CK(cudaMalloc(&out, 4)); int phases = 64; const double LDS_BYTES_PER_STEP = 8 * (4 * 512.0 + 8 * 128.0) * 1;   // per block per step: 8 warps x (4 x 512 B + 8 x 128 B)
  CK(cudaFuncSetAttribute(pattern<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(pattern<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); CK(cudaFuncSetAttribute(pattern<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  const char* names[] = {"lds", "copy", "both"}; std::string table = "["; bool tf = true; double lds_b1 = 0, copy_b1 = 0;
  for (int mix = 0; mix < 3; mix++) for (int b : {1, 2, 3, 4, 6}) {
    int smem = std::max(16384, (96 * 1024) / b); int grid = sms * b; std::vector<double> step_cyc, agg_lds, agg_copy, share_cv;
    auto launch = [&]() { if (mix == 0) pattern<0><<<grid, 256, smem>>>(src, phases, cyc, tin, out); else if (mix == 1) pattern<1><<<grid, 256, smem>>>(src, phases, cyc, tin, out); else pattern<2><<<grid, 256, smem>>>(src, phases, cyc, tin, out); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); };
    launch();
    for (int r = 0; r < reps; r++) { launch(); std::vector<long long> h(grid); CK(cudaMemcpy(h.data(), cyc, grid * 8, cudaMemcpyDeviceToHost)); std::vector<long long> hs = h; std::sort(hs.begin(), hs.end());
      double med = (double)hs[grid / 2]; step_cyc.push_back(med / (phases * 8.0));
      double mean = 0; for (auto x : h) mean += x; mean /= grid; double var = 0; for (auto x : h) var += (x - mean) * (x - mean); share_cv.push_back(sqrt(var / grid) / mean);
      // aggregate per SM: b blocks each moving LDS_BYTES_PER_STEP per step over its own time -> bytes per SM cycle
      double lds_bytes_per_cycle = (mix == 1) ? 0 : b * LDS_BYTES_PER_STEP * phases * 8 / med; agg_lds.push_back(lds_bytes_per_cycle);
      double copy_bps = (mix == 0) ? 0 : b * 4096.0 * phases / (med / hz); agg_copy.push_back(copy_bps); }
    std::sort(step_cyc.begin(), step_cyc.end()); std::sort(agg_lds.begin(), agg_lds.end()); std::sort(agg_copy.begin(), agg_copy.end()); std::sort(share_cv.begin(), share_cv.end());
    double sc = step_cyc[reps / 2], al = agg_lds[reps / 2], ac = agg_copy[reps / 2], cv = share_cv[reps / 2];
    if (mix == 0 && b == 1) lds_b1 = al; if (mix == 1 && b == 1) copy_b1 = ac;
    char bb[200]; snprintf(bb, sizeof bb, "%s[\"%s\", %d, %.1f, %.1f, %.4g, %.3f]", tf ? "" : ", ", names[mix], b, sc, al, ac, cv); table += bb; tf = false;
    char q[96]; snprintf(q, sizeof q, "lsu_contention[mix=%s,b=%d]", names[mix], b); char det[400]; snprintf(det, sizeof det, "{\"mix\": \"%s\", \"blocks_per_sm\": %d, \"smem_bytes\": %d, \"cycles_per_step_per_block\": %.1f, \"smem_bytes_per_cycle_per_sm\": %.1f, \"copy_bytes_per_s_per_sm\": %.4g, \"per_block_time_cv\": %.3f}", names[mix], b, smem, sc, al, ac, cv);
    emit_clocked(q, "cycles", step_cyc, det, mhz); }
  table += "]";
  std::string det = "{\"table_mix_b_cyclesPerStep_smemBytesPerCycle_copyBytesPerS_cv\": " + table + ", \"lds_bytes_per_step_per_block\": " + std::to_string((long long)LDS_BYTES_PER_STEP) + ", \"method\": \"T5-width ld.shared pattern (4 x 128-bit + 8 x 32-bit per thread per step), one 16 B cp.async per thread per phase (8 steps) with wait_group 1, 256 threads; residency forced by dynamic-smem padding; aggregate = b x per-block bytes over the median block's time; cv = spread of per-block times (0 = fair share)\"}";
  emit_clocked("smem_port_contention_table", "table", std::vector<double>{lds_b1}, det.c_str(), mhz);
  return 0;
}
