#include "tiled_gemm_bank.cuh"
#include "async_pipeline.cuh"

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

__global__ void swizzle_bank_free_gemm_kernel(
    const __half* __restrict__ gmem_A,
    const __half* __restrict__ gmem_B,
    __half* __restrict__ gmem_C,
    int M, int N, int K) {

    // 声明片上 Shared Memory 双缓冲区
    __shared__ alignas(16) __half smem_A[2][TILE_M * TILE_K];
    __shared__ alignas(16) __half smem_B[2][TILE_K * TILE_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 ~ 255

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int write_stage = 0;

    // =========================================================================
    // 🌟 统一 Swizzle 寻址函数 (以 16 字节 / 8 个 __half 为单位向量)
    // =========================================================================
    // A 矩阵: 64 行 x 4 个向量列 (TILE_K / 8 = 4)。取 (a_row & 0x3) 做 4 行周期 Swizzle
    auto get_smem_A_ptr = [&](int stage, int row, int col) {
        int col_vec = col / 8;
        int col_off = col % 8;
        int swizzled_col_vec = (col_vec ^ (row & 0x3)) & 0x3;
        return &smem_A[stage][row * TILE_K + swizzled_col_vec * 8 + col_off];
    };

    // B 矩阵: 32 行 x 8 个向量列 (TILE_N / 8 = 8)。取 (b_row & 0x7) 做 8 行周期 Swizzle
    auto get_smem_B_ptr = [&](int stage, int row, int col) {
        int col_vec = col / 8;
        int col_off = col % 8;
        int swizzled_col_vec = (col_vec ^ (row & 0x7)) & 0x7;
        return &smem_B[stage][row * TILE_N + swizzled_col_vec * 8 + col_off];
    };

    // 🌟 256 线程协同搬运函数 (A 和 B 各有 256 个向量，各只需搬 1 次)
    auto load_async_swizzled = [&](int stage, int k_offset) {
        // --- 1. 搬运 A 矩阵 (64x32 FP16 = 256 个 16-Byte 向量) ---
        int a_row = tid / 4;        // 0 ~ 63
        int a_col_vec = tid % 4;    // 0 ~ 3
        int g_r_a = block_row + a_row;
        int g_c_a = k_offset + a_col_vec * 8;
        bool valid_a = (g_r_a < M) && (g_c_a + 7 < K);

        pipeline::cp_async_16bytes(
            get_smem_A_ptr(stage, a_row, a_col_vec * 8),
            gmem_A + g_r_a * K + g_c_a,
            valid_a
        );

        // --- 2. 搬运 B 矩阵 (32x64 FP16 = 256 个 16-Byte 向量) ---
        int b_row = tid / 8;        // 0 ~ 31
        int b_col_vec = tid % 8;    // 0 ~ 7
        int g_r_b = k_offset + b_row;
        int g_c_b = block_col + b_col_vec * 8;
        bool valid_b = (g_r_b < K) && (g_c_b + 7 < N);

        pipeline::cp_async_16bytes(
            get_smem_B_ptr(stage, b_row, b_col_vec * 8),
            gmem_B + g_r_b * N + g_c_b,
            valid_b
        );
    };

    // Prologue 阶段
    load_async_swizzled(write_stage, 0);
    pipeline::cp_async_commit();
    write_stage ^= 1;

    // 🌟 每个线程维护 4x4 = 16 个 C 矩阵元素的累加器
    float accum[4][4] = {0.0f};

    // 计算核函数内部内联计算 Lambda
    auto compute_stage = [&](int stage) {
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_vals[4];
            float b_vals[4];

            // 读取 A 的 4 个元素 (跨步 16 覆盖 64 行)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row_a = threadIdx.y + i * 16;
                a_vals[i] = __half2float(*get_smem_A_ptr(stage, row_a, k));
            }

            // 读取 B 的 4 个元素 (跨步 16 覆盖 64 列)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int col_b = threadIdx.x + j * 16;
                b_vals[j] = __half2float(*get_smem_B_ptr(stage, k, col_b));
            }

            // 外积计算更新 16 个点
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    accum[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
    };

    // Main Loop
    for (int tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        int next_k_offset = (tile_idx + 1) * TILE_K;

        // 1. 后台预取下一 Tile
        load_async_swizzled(write_stage, next_k_offset);
        pipeline::cp_async_commit();

        // 2. 等待上一 Tile 搬运完成
        pipeline::cp_async_wait_pending<1>();
        __syncthreads();

        // 3. 读取计算
        int read_stage = write_stage ^ 1;
        compute_stage(read_stage);

        write_stage ^= 1;
        __syncthreads();
    }

    // Epilogue 阶段
    pipeline::cp_async_wait_pending<0>();
    __syncthreads();

    int read_stage = write_stage ^ 1;
    compute_stage(read_stage);

    // 🌟 写回 Global Memory (覆盖整个 64x64 领地)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int global_r = block_row + threadIdx.y + i * 16;
            int global_c = block_col + threadIdx.x + j * 16;

            if (global_r < M && global_c < N) {
                gmem_C[global_r * N + global_c] = __float2half(accum[i][j]);
            }
        }
    }
}

namespace smem_opt {

void launch_swizzle_bank_free_gemm(
    const __half* d_A, 
    const __half* d_B, 
    __half* d_C, 
    int M, int N, int K, 
    cudaStream_t stream) {

    // 🌟 修正：一个 Block 256 线程，按 TILE_M(64) 和 TILE_N(64) 划分 Grid！
    dim3 block(16, 16); // 256 线程
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    swizzle_bank_free_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

} // namespace smem_opt