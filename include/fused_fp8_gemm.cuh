#pragma once
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ptx {

// 16 字节严格对齐加载 (Shared Memory)
__device__ __forceinline__ void load_smem_16bytes(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  const void* smem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

// =============================================================================
// ldmatrix PTX 指令：warp 协同的 smem -> register 加载
// =============================================================================
// ldmatrix.x4: 加载 4 个 8x8 的 16-bit 矩阵 (512 字节)
//   每个 thread 提供 1 个 16 字节地址, 共 32 个地址覆盖 4 矩阵 x 8 行
//   输出: 每个 thread 4 个 uint32_t (a0/a1/a2/a3)
//   适用: FP8 m16n8k32 的 A 操作数 (16x32 FP8 = 16x16 16-bit, 沿 K 方向配对)
__device__ __forceinline__ void ldmatrix_x4(uint32_t* dst, const void* smem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(smem_addr)
    );
}

// ldmatrix.x2.trans: 加载 2 个 8x8 的 16-bit 矩阵 (256 字节), 加载时硬件转置
//   关键: .trans 让硬件把 smem 中按行排列的 16-bit 元素加载为列优先布局
//   适用: FP8 m16n8k32 的 B 操作数 (32x8 FP8 = 16x8 16-bit, 沿 K 方向配对)
//   每个 thread 输出 2 个 uint32_t (b0/b1), 每个 uint32_t 含 4 个 FP8 (沿 K 方向)
// =============================================================================
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* dst, const void* smem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(smem_addr)
    );
}

// mma.sync.m16n8k32 FP8 指令规范 (.row.col.f32.e4m3.e4m3.f32)
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