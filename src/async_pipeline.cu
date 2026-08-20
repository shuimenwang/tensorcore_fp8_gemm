#include "async_pipeline.cuh"

// 设定 Tile 尺寸与 Thread 维度
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void async_double_buffer_pipeline_kernel(
    const float* __restrict__ gmem_A,
    const float* __restrict__ gmem_B,
    float* __restrict__ gmem_C,
    int M, int N, int K) {

    // 声明片上 Shared Memory 双缓冲区 [2][TILE_M][TILE_K]
    __shared__ alignas(16) float smem_A[2][TILE_M][TILE_K];
    __shared__ alignas(16) float smem_B[2][TILE_K][TILE_N];

    int tx = threadIdx.x; // 0..31
    int ty = threadIdx.y; // 0..31

    // 🌟【关键新增】：拿到当前 Block 内拍平的 1D 线程 ID (0 .. 1023)
    int tid = ty * blockDim.x + tx; 

    // 当前 Block 负责的全局矩阵 C 的 Tile 锚点
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    // 当前线程用于最后 C 矩阵计算与写回的全局坐标
    int global_a_row = block_row + ty;
    int global_b_col = block_col + tx;

    // 累加器（寄存器）
    float accum = 0.0f;

    // 计算总共需要的 K 维分块步数
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    // 双缓冲阶段指示器 (0 或 1)
    int write_stage = 0;

    // =========================================================================
    // 🌟【核心助手 Lambda 函数】：16 字节完全对齐的 Shared Memory 搬运逻辑
    // =========================================================================
    auto load_gmem_to_smem_async = [&](int stage, int k_tile_offset) {
        // --- 1. 搬运 A 矩阵 (TILE_M * TILE_K = 1024 个 float = 256 个 16-byte 块) ---
        // 前 256 个线程 (tid < 256) 专职搬运 A，每个线程搬运 4 个 float
        if (tid < 256) {
            int a_copy_idx = tid * 4; // 保证 index 必定是 0, 4, 8, 12... (16字节强对齐)
            int a_row = a_copy_idx / TILE_K;
            int a_col = a_copy_idx % TILE_K;

            int g_a_r = block_row + a_row;
            int g_a_c = k_tile_offset + a_col;

            bool valid_A = (g_a_r < M) && (g_a_c + 3 < K); // 确保连续 4 个 float 不越界
            const float* ptr_A = gmem_A + g_a_r * K + g_a_c;

            pipeline::cp_async_16bytes(&smem_A[stage][a_row][a_col], ptr_A, valid_A);
        }

        // --- 2. 搬运 B 矩阵 (TILE_K * TILE_N = 1024 个 float = 256 个 16-byte 块) ---
        // 接下来的 256 个线程 (256 <= tid < 512) 专职搬运 B，每个线程搬运 4 个 float
        if (tid >= 256 && tid < 512) {
            int b_copy_idx = (tid - 256) * 4; // 保证 index 必定是 0, 4, 8, 12... (16字节强对齐)
            int b_row = b_copy_idx / TILE_N;
            int b_col = b_copy_idx % TILE_N;

            int g_b_r = k_tile_offset + b_row;
            int g_b_c = block_col + b_col;

            bool valid_B = (g_b_r < K) && (g_b_c + 3 < N); // 确保连续 4 个 float 不越界
            const float* ptr_B = gmem_B + g_b_r * N + g_b_c;

            pipeline::cp_async_16bytes(&smem_B[stage][b_row][b_col], ptr_B, valid_B);
        }
    };

    // =========================================================================
    // 【Phase 1: Prologue 前导预取】加载第 0 块数据到 smem[*][0]
    // =========================================================================
    load_gmem_to_smem_async(write_stage, 0);

    // 提交第 0 组异步请求
    pipeline::cp_async_commit();

    // 翻转写入 Stage：下一块将写入 Buffer 1
    write_stage ^= 1;

    // =========================================================================
    // 【Phase 2: Main Loop 双缓冲流水线主循环】
    // =========================================================================
    for (int tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        // 1. 在后台异步发起【下一块 (tile_idx + 1)】数据的加载
        int next_k_offset = (tile_idx + 1) * TILE_K;
        load_gmem_to_smem_async(write_stage, next_k_offset);

        // 提交下一组请求
        pipeline::cp_async_commit();

        // 2. 核心掩盖点：等待上一次搬运完成
        pipeline::cp_async_wait_pending<1>();
        __syncthreads();

        // 3. 执行【当前块 (tile_idx)】的矩阵计算（逻辑完全未变！）
        int read_stage = write_stage ^ 1;
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            accum += smem_A[read_stage][ty][k] * smem_B[read_stage][k][tx];
        }

        // 4. 翻转 Buffer，准备下一轮
        write_stage ^= 1;
        __syncthreads();
    }

    // =========================================================================
    // 【Phase 3: Epilogue 收尾阶段】处理最后一块预取好的数据
    // =========================================================================
    pipeline::cp_async_wait_pending<0>();
    __syncthreads();

    int read_stage = write_stage ^ 1;
    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
        accum += smem_A[read_stage][ty][k] * smem_B[read_stage][k][tx];
    }

    // 写回全局显存
    if (global_a_row < M && global_b_col < N) {
        gmem_C[global_a_row * N + global_b_col] = accum;
    }
}

void launch_async_pipeline_gemm_stage(const float* d_A, const float* d_B, float* d_C,
                                      int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M); // 32x32 = 1024 线程/Block
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    async_double_buffer_pipeline_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}