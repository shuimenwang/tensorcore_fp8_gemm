#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "common.h"

#ifndef TILE_K
#define TILE_K 32
#endif

namespace smem_opt {

// 1. Swizzle 寻址物理映射
template<int BITS_SHIFTS = 3>
__device__ __forceinline__ int get_swizzled_smem_offset(int row, int col) {
    int swizzled_col = col ^ (row >> BITS_SHIFTS);
    return row * TILE_K + swizzled_col;
}

// 🌟 2. 【新增】：向外部暴露 launcher 接口声明
void launch_swizzle_bank_free_gemm(
    const __half* d_A, 
    const __half* d_B, 
    __half* d_C, 
    int M, int N, int K, 
    cudaStream_t stream = nullptr
);

} // namespace smem_opt