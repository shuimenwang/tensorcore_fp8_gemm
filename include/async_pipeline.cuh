#ifndef ASYNC_PIPELINE_CUH
#define ASYNC_PIPELINE_CUH

#include "common.h"

namespace pipeline {

// 封装 Inline PTX 指令：执行 16 字节从 Global Memory 到 Shared Memory 的硬件级异步拷贝
__device__ __forceinline__ void cp_async_16bytes(void* smem_ptr, const void* gmem_ptr, bool predicate) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    
    // 如果 predicate 为 true（未越界），执行 cp.async；否则填充 0，保证安全性
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.b128 [%0], {0, 0, 0, 0};\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"((int)predicate)
    );
}

// 提交当前异步拷贝组（相当于告诉硬件 DMA：“这批任务下发完毕”）
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// 阻塞等待，直到后台只剩下最多 N 组异步拷贝还在进行
template <int N>
__device__ __forceinline__ void cp_async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

} // namespace pipeline

// 对外暴露的测试 Kernel 声明
void launch_async_copy_test(const float* d_in, float* d_out, int M, int N, cudaStream_t stream);

#endif // ASYNC_PIPELINE_CUH