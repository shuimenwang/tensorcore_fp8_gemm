#ifndef ASYNC_PIPELINE_CUH
#define ASYNC_PIPELINE_CUH

#include <common.h>


namespace pipeline {

// 1. 手写 Inline PTX 汇编：异步 16 字节 (128-bit) 搬运并带边界 Guard
__device__ __forceinline__ void cp_async_16bytes(void* smem_ptr, const void* gmem_ptr, bool src_valid) {
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    // 使用 PTX 指令 cp.async.ca 进行 16 字节异步拷贝，当 src_valid 为 false 时自动补 0（避免越界）
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
        "}\n"
        : 
        : "r"(smem_addr), "l"(gmem_ptr), "r"((int)src_valid)
        : "memory"
    );
}

// 2. 提交当前异步拷贝任务为一个 Group
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

// 3. 阻塞等待，直到未完成的 Group 数量 <= N
template <int N>
__device__ __forceinline__ void cp_async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

} // namespace pipeline

// Launch 声明
void launch_async_pipeline_gemm_stage(const float* d_A, const float* d_B, float* d_C,
                                     int M, int N, int K, cudaStream_t stream = 0);

#endif // ASYNC_PIPELINE_CUH