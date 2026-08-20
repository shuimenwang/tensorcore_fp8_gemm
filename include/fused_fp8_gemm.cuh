#pragma once
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ptx {

// 16 字节严格对齐加载
__device__ __forceinline__ void load_smem_16bytes(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  const void* smem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

// mma.sync.m16n8k32 FP8 指令规范
__device__ __forceinline__ void mma_m16n8k32_fp8(float& c0, float& c1, float& c2, float& c3,
                                                 uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}

}  // namespace ptx

void launch_fused_fp8_tensor_core_gemm(const __nv_fp8_e4m3* d_A,
                                       const __nv_fp8_e4m3* d_B,
                                       float* d_C,
                                       float scale_a, float scale_b,
                                       int M, int N, int K,
                                       cudaStream_t stream);