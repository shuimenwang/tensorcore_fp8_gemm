#include "fused_fp8_gemm.cuh"

#define TILE_M    128
#define TILE_N    128
#define TILE_K    64
#define PADDING_K 16

namespace pipeline {

__device__ __forceinline__ void cp_async_16bytes_zfill(void* smem_ptr, const void* gmem_ptr, bool predicate) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.b32 [%0], {%3, %3, %3, %3};\n"
        "}\n"
        :
        : "r"(smem_addr),
          "l"(gmem_ptr),
          "r"(static_cast<int>(predicate)),
          "r"(0)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

}  // namespace pipeline

__global__ void __launch_bounds__(256, 2) fused_fp8_tensor_core_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ gmem_A,
    const __nv_fp8_e4m3* __restrict__ gmem_B,
    float* __restrict__ gmem_C,
    float scale_a, float scale_b,
    int M, int N, int K) {

    // smem_A: Row-Major [TILE_M][TILE_K + PADDING_K]
    __shared__ alignas(16) __nv_fp8_e4m3 smem_A[2][TILE_M * (TILE_K + PADDING_K)];
    // smem_B: 保持按 TILE_K 方向连续存放 [TILE_N][TILE_K + PADDING_K]，确保 cp_async 16 字节对齐
    __shared__ alignas(16) __nv_fp8_e4m3 smem_B[2][TILE_N * (TILE_K + PADDING_K)];

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int block_row   = blockIdx.y * TILE_M;
    int block_col   = blockIdx.x * TILE_N;
    int num_tiles   = (K + TILE_K - 1) / TILE_K;
    int write_stage = 0;

    auto load_async_fp8 = [&](int stage, int k_offset) {
        // 1. 加载 A 到 Shared Memory (16 字节对齐)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int vec_idx = tid + i * 256;
            int a_row   = vec_idx / 4;
            int a_col   = (vec_idx % 4) * 16;  // 保证 16 的倍数，满足对齐

            int g_r = block_row + a_row;
            int g_c = k_offset + a_col;

            bool valid     = (g_r < M) && (g_c < K);
            const void* src = gmem_A + g_r * K + g_c;
            void* dst       = &smem_A[stage][a_row * (TILE_K + PADDING_K) + a_col];

            pipeline::cp_async_16bytes_zfill(dst, src, valid);
        }

        // 2. 加载 B 到 Shared Memory (按 N 维度拆分，K 维度连续读 16 字节)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int vec_idx = tid + i * 256;
            int b_n     = vec_idx / 4;        // 0 ~ 127 (对应 TILE_N 轴)
            int b_k     = (vec_idx % 4) * 16; // 0, 16, 32, 48 (对应 TILE_K 轴，16字节连续对齐)

            int g_n = block_col + b_n;        // Global Memory 中 B^T 的 行索引 (N 轴)
            int g_k = k_offset + b_k;         // Global Memory 中 B^T 的 列索引 (K 轴)

            bool valid = (g_n < N) && (g_k < K);

            // 修改点：此时 gmem_B 是 [N, K] 形状，连续访问 K 维度 (以 K 为 stride)
            const void* src = gmem_B + g_n * K + g_k;

            // 写入 Shared Memory，布局保持 [TILE_N][TILE_K + PADDING_K]
            void* dst = &smem_B[stage][b_n * (TILE_K + PADDING_K) + b_k];

            pipeline::cp_async_16bytes_zfill(dst, src, valid);
        }
    };

    load_async_fp8(write_stage, 0);
    pipeline::cp_async_commit();
    write_stage ^= 1;

    int warp_row_idx = warp_id / 2;
    int warp_col_idx = warp_id % 2;

    int warp_row_offset = warp_row_idx * 32;
    int warp_col_offset = warp_col_idx * 64;

    float accum[2][8][4] = {0.0f};

    auto compute_k_step = [&](int read_stage, int k_step) {
        uint32_t reg_a[2][4];
        uint32_t reg_b[8][2];

        // 1. 读取 Matrix A (16 字节对齐读取)
        #pragma unroll
        for (int m_sub = 0; m_sub < 2; ++m_sub) {
            // 每 2 个线程负责同一行 (0,0, 1,1, 2,2 ... 15,15)
            int a_row = warp_row_offset + m_sub * 16 + (lane_id / 2);
            // 偶数线程读 K 轴前 16 Byte，奇数线程读 K 轴后 16 Byte
            int a_col = k_step * 32 + (lane_id % 2) * 16;

            const void* ptr_a = &smem_A[read_stage][a_row * (TILE_K + PADDING_K) + a_col];
            ptx::load_smem_16bytes(reg_a[m_sub][0], reg_a[m_sub][1], reg_a[m_sub][2], reg_a[m_sub][3], ptr_a);
        }

        // 2. 读取 Matrix B (采用 8 字节连续读取，严格匹配对齐)
        #pragma unroll
        for (int n_sub = 0; n_sub < 8; ++n_sub) {
            int b_n = warp_col_offset + n_sub * 8 + (lane_id % 8);
            int b_k = k_step * 32 + (lane_id / 8) * 8; // 8 字节对齐

            const void* ptr_b = &smem_B[read_stage][b_n * (TILE_K + PADDING_K) + b_k];

            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_b));
            asm volatile(
                "ld.shared.v2.b32 {%0, %1}, [%2];\n"
                : "=r"(reg_b[n_sub][0]), "=r"(reg_b[n_sub][1])
                : "r"(smem_addr)
            );
        }

        // 3. 执行 MMA 计算
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                ptx::mma_m16n8k32_fp8(
                    accum[i][j][0], accum[i][j][1], accum[i][j][2], accum[i][j][3],
                    reg_a[i][0], reg_a[i][1], reg_a[i][2], reg_a[i][3],
                    reg_b[j][0], reg_b[j][1]
                );
            }
        }
    };

    // 流水线主循环
    for (int tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        int next_k_offset = (tile_idx + 1) * TILE_K;

        load_async_fp8(write_stage, next_k_offset);
        pipeline::cp_async_commit();

        pipeline::cp_async_wait_pending<1>();
        __syncthreads();

        int read_stage = write_stage ^ 1;

        #pragma unroll
        for (int k_step = 0; k_step < TILE_K / 32; ++k_step) {
            compute_k_step(read_stage, k_step);
        }

        write_stage ^= 1;
        __syncthreads();
    }

    // 处理最后一个 Tile
    pipeline::cp_async_wait_pending<0>();
    __syncthreads();

    int read_stage = write_stage ^ 1;
    #pragma unroll
    for (int k_step = 0; k_step < TILE_K / 32; ++k_step) {
        compute_k_step(read_stage, k_step);
    }

    // Epilogue 写回 Global Memory
    float total_scale   = scale_a * scale_b;
    int group_id        = lane_id / 4;
    int lane_in_group   = lane_id % 4;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int row0 = block_row + warp_row_offset + i * 16 + group_id;
            int row1 = row0 + 8;

            int col0 = block_col + warp_col_offset + j * 8 + lane_in_group * 2;
            int col1 = col0 + 1;

            if (row0 < M && col0 < N) gmem_C[row0 * N + col0] = accum[i][j][0] * total_scale;
            if (row0 < M && col1 < N) gmem_C[row0 * N + col1] = accum[i][j][1] * total_scale;
            if (row1 < M && col0 < N) gmem_C[row1 * N + col0] = accum[i][j][2] * total_scale;
            if (row1 < M && col1 < N) gmem_C[row1 * N + col1] = accum[i][j][3] * total_scale;
        }
    }
}

void launch_fused_fp8_tensor_core_gemm(const __nv_fp8_e4m3* d_A,
                                       const __nv_fp8_e4m3* d_B,
                                       float* d_C,
                                       float scale_a, float scale_b,
                                       int M, int N, int K,
                                       cudaStream_t stream) {
    dim3 block(256, 1);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    fused_fp8_tensor_core_gemm_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, scale_a, scale_b, M, N, K
    );
}