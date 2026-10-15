// Shared TMA / mbarrier helpers for B35-B37 and C2 (round 2, group 2).
#pragma once
#include "common.cuh"
#include <cuda.h>
__device__ __forceinline__ unsigned saddr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned b, unsigned count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(count)); }
__device__ __forceinline__ void mbar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(unsigned b, unsigned bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(unsigned b, unsigned phase) { unsigned done = 0; while (!done) { asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(b), "r"(phase) : "memory"); } }
__device__ __forceinline__ void tma_load_1d(unsigned dst, const void* src, unsigned bytes, unsigned b) { asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(b) : "memory"); }
__device__ __forceinline__ void tma_load_2d(unsigned dst, const CUtensorMap* map, int x, int y, unsigned b) { asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" :: "r"(dst), "l"(map), "r"(x), "r"(y), "r"(b) : "memory"); }
__device__ __forceinline__ void tma_store_1d(void* dst, unsigned src, unsigned bytes) { asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" :: "l"(dst), "r"(src), "r"(bytes) : "memory"); asm volatile("cp.async.bulk.commit_group;\n"); }
__device__ __forceinline__ void tma_store_wait() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); }
static bool make_map_2d(CUtensorMap* map, void* base, int rows, int cols, int box_rows, int box_cols) {   // fp32 row-major [rows][cols], box [box_rows][box_cols]
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows}; cuuint64_t strides[1] = {(cuuint64_t)cols * 4}; cuuint32_t box[2] = {(cuuint32_t)box_cols, (cuuint32_t)box_rows}; cuuint32_t estr[2] = {1, 1};
  return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
