// B1: declared device properties (cudaGetDeviceProperties / cudaDeviceGetAttribute). Not timed.
#include "common.cuh"
int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int clk = 0, mclk = 0, busw = 0, l2 = 0, optin = 0, drv = 0, rt = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0); cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, 0); cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("{\"quantity\": \"device_properties\", \"unit\": \"\", \"samples\": [], \"detail\": {"
         "\"name\": \"%s\", \"cc\": \"%d.%d\", \"sms\": %d, \"warp_size\": %d, \"smem_bytes_per_block\": %zu, "
         "\"smem_bytes_per_block_optin\": %d, \"smem_bytes_per_sm\": %zu, \"regs_per_sm\": %d, \"regs_per_block\": %d, "
         "\"l2_bytes\": %d, \"max_threads_per_block\": %d, \"max_blocks_per_sm\": %d, \"clock_khz\": %d, \"mem_clock_khz\": %d, "
         "\"bus_width_bits\": %d, \"total_mem_bytes\": %zu, \"managed\": %d, \"pageable_access\": %d, \"integrated\": %d, "
         "\"driver_version\": %d, \"runtime_version\": %d}}\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.warpSize, p.sharedMemPerBlock, optin, p.sharedMemPerMultiprocessor,
         p.regsPerMultiprocessor, p.regsPerBlock, l2, p.maxThreadsPerBlock, p.maxBlocksPerMultiProcessor, clk, mclk, busw,
         p.totalGlobalMem, p.managedMemory, p.pageableMemoryAccess, p.integrated, drv, rt);
  return 0;
}
