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

    // 当前 Block 负责的全局矩阵 C 的 Tile 锚点
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    // 当前线程对应的全局坐标与 Guardian
    int global_a_row = block_row + ty;
    int global_b_col = block_col + tx;

    // 累加器（寄存器）
    float accum = 0.0f;

    // 计算总共需要的 K 维分块步数
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    // 双缓冲阶段指示器 (0 或 1)
    int write_stage = 0;

    // =========================================================================
    // 【Phase 1: Prologue 前导预取】加载第 0 块数据到 smem[*][0]
    // =========================================================================
    int k_offset = 0;
    
    // A 矩阵全局指针与 Guard (以 16 字节对齐考虑，这里演示元素级安全 Guard)
    bool valid_A_0 = (global_a_row < M) && (k_offset + tx < K);
    const float* ptr_A_0 = gmem_A + global_a_row * K + (k_offset + tx);
    pipeline::cp_async_16bytes(&smem_A[write_stage][ty][tx], ptr_A_0, valid_A_0);

    // B 矩阵全局指针与 Guard
    bool valid_B_0 = (k_offset + ty < K) && (global_b_col < N);
    const float* ptr_B_0 = gmem_B + (k_offset + ty) * N + global_b_col;
    pipeline::cp_async_16bytes(&smem_B[write_stage][ty][tx], ptr_B_0, valid_B_0);

    // 提交第 0 组异步请求
    pipeline::cp_async_commit();

    // 翻转写入 Stage：下一块将写入 Buffer 1
    write_stage ^= 1;

    // =========================================================================
    // 【Phase 2: Main Loop 双缓冲流水线主循环】
    // =========================================================================
    for (int tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        // 1. 在后台异步发起【下一块 (tile_idx + 1)】数据的加载到 smem[*][write_stage]
        int next_k_offset = (tile_idx + 1) * TILE_K;

        bool valid_A_next = (global_a_row < M) && (next_k_offset + tx < K);
        const float* ptr_A_next = gmem_A + global_a_row * K + (next_k_offset + tx);
        pipeline::cp_async_16bytes(&smem_A[write_stage][ty][tx], ptr_A_next, valid_A_next);

        bool valid_B_next = (next_k_offset + ty < K) && (global_b_col < N);
        const float* ptr_B_next = gmem_B + (next_k_offset + ty) * N + global_b_col;
        pipeline::cp_async_16bytes(&smem_B[write_stage][ty][tx], ptr_B_next, valid_B_next);

        // 提交下一组请求
        pipeline::cp_async_commit();

        // 2. 核心掩盖点：只等待【当前计算需要的读 Stage】(后台保持 1 组仍在异步搬运)
        pipeline::cp_async_wait_pending<1>();
        __syncthreads();

        // 3. 执行【当前块 (tile_idx)】的矩阵计算，从 read_stage 中读取
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
    // 等待所有后台任务完成
    pipeline::cp_async_wait_pending<0>();
    __syncthreads();

    int read_stage = write_stage ^ 1;
    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
        accum += smem_A[read_stage][ty][k] * smem_B[read_stage][k][tx];
    }

    // 写回全局显存，带边界 Protection
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