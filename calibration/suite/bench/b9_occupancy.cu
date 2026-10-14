// B9: occupancy limits. (a) cudaOccupancyMaxActiveBlocksPerMultiprocessor for a 256-thread kernel as
// dynamic shared memory grows 0 .. 96 KB; (b) a measured sweep: blocks spin on a host-set flag, so the
// number that become resident per SM before the flag is the real concurrency limit. (c) The register
// cliff: the same with kernels forced to 32, 64, 128, 255 registers per thread via __launch_bounds__.
#include "common.cuh"
#include <unistd.h>
__global__ void resident(int* cnt, int* mx, int extra) {
  // peak concurrency per SM: count on entry, atomicMax of the post-increment value, hold ~20 ms, count out
  extern __shared__ char smem[];
  __shared__ int slot;
  unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm));
  if (threadIdx.x == 0) { int c = atomicAdd(&cnt[sm], 1) + 1; atomicMax(&mx[sm], c); slot = c; }
  __syncthreads();
  long long t0 = clock64(); while (clock64() - t0 < 40000000LL) { }       // ~20 ms at 2 GHz
  if (extra > 0 && threadIdx.x == 0) smem[0] = (char)slot;
  __syncthreads();
  if (threadIdx.x == 0) atomicSub(&cnt[sm], 1);
}
template <int MINB> __global__ void __launch_bounds__(256, MINB) reg_kernel(float* out, int iters) {
  float v[16]; for (int i = 0; i < 16; i++) v[i] = threadIdx.x + i;
  for (int k = 0; k < iters; k++) for (int i = 0; i < 16; i++) v[i] = fmaf(v[i], 1.0001f, v[(i + 1) & 15]);
  float s = 0; for (int i = 0; i < 16; i++) s += v[i]; if (s == 123456.789f) out[0] = s;
}
int main(int argc, char** argv) {
  int sms = sm_count(); int *cnt, *mx; CK(cudaMalloc(&cnt, 1024 * 4)); CK(cudaMalloc(&mx, 1024 * 4));   // %smid may exceed the SM count
  std::vector<int> hmx(1024);
  CK(cudaFuncSetAttribute(resident, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  for (int smem : {0, 8192, 16384, 32768, 49152, 65536, 98304}) {
    int api = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api, resident, 256, smem));
    CK(cudaMemset(cnt, 0, 1024 * 4)); CK(cudaMemset(mx, 0, 1024 * 4));
    resident<<<sms * 40, 256, smem>>>(cnt, mx, smem); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hmx.data(), mx, 1024 * 4, cudaMemcpyDeviceToHost));
    int peak = 0; for (int i = 0; i < 1024; i++) peak = std::max(peak, hmx[i]);
    char d[128]; snprintf(d, sizeof d, "{\"smem_bytes\": %d, \"threads\": 256, \"occupancy_api\": %d, \"measured_peak_resident\": %d}", smem, api, peak);
    char q[64]; snprintf(q, sizeof q, "max_blocks_per_sm[smem=%d]", smem); emit(q, "blocks", std::vector<double>{(double)peak}, d);
  }
  float* out; CK(cudaMalloc(&out, 4)); cudaFuncAttributes fa;
  struct { const char* name; int minb; } cfg[] = {{"minblocks1", 1}, {"minblocks2", 2}, {"minblocks4", 4}, {"minblocks8", 8}};
  int apis[4]; int regs[4];
  CK(cudaFuncGetAttributes(&fa, reg_kernel<1>)); regs[0] = fa.numRegs; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&apis[0], reg_kernel<1>, 256, 0));
  CK(cudaFuncGetAttributes(&fa, reg_kernel<2>)); regs[1] = fa.numRegs; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&apis[1], reg_kernel<2>, 256, 0));
  CK(cudaFuncGetAttributes(&fa, reg_kernel<4>)); regs[2] = fa.numRegs; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&apis[2], reg_kernel<4>, 256, 0));
  CK(cudaFuncGetAttributes(&fa, reg_kernel<8>)); regs[3] = fa.numRegs; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&apis[3], reg_kernel<8>, 256, 0));
  for (int i = 0; i < 4; i++) { char d[128]; snprintf(d, sizeof d, "{\"regs_per_thread\": %d, \"threads\": 256, \"launch_bounds_min_blocks\": %d}", regs[i], cfg[i].minb);
    char q[64]; snprintf(q, sizeof q, "max_blocks_per_sm[regs=%d]", regs[i]); emit(q, "blocks", std::vector<double>{(double)apis[i]}, d); }
  return 0;
}
