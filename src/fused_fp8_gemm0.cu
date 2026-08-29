#include "fused_fp8_gemm0.cuh"

#define TILE_M    128
#define TILE_N    128
#define TILE_K    64
#define PADDING_K 16
#define PADDING_N 16

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

// 4 字节拷贝回退路径: 当 K/N 不是 16 的倍数时, 全局地址无法保证 16B 对齐,
// 而 cp.async 要求源/目的地址按 cp-size 对齐, 因此退化为 4B 粒度的 cp.async.ca
__device__ __forceinline__ void cp_async_4bytes_zfill(void* smem_ptr, const void* gmem_ptr, bool predicate) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 4;\n"
        "  @!p st.shared.b32 [%0], %3;\n"
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

// 按照 NVIDIA PTX 官方规范 (Figure 88) 从 SMEM 加载 16x32 的 Matrix A 片段
// 寄存器象限顺序: a0=左上, a1=左下, a2=右上, a3=右下 (与 ldmatrix.x4 输出顺序一致)
__device__ __forceinline__ void load_a_m16k32_fp8(const __nv_fp8_e4m3* tile_ptr, int ldA, uint32_t a[4], int lane_id) {
    int group = lane_id / 4;
    int row0  = group;          // 行 0~7
    int row1  = group + 8;      // 行 8~15
    int col   = (lane_id % 4) * 4;

    const void* ptr_r0_c0 = tile_ptr + row0 * ldA + col;
    const void* ptr_r0_c1 = tile_ptr + row0 * ldA + col + 16;
    const void* ptr_r1_c0 = tile_ptr + row1 * ldA + col;
    const void* ptr_r1_c1 = tile_ptr + row1 * ldA + col + 16;

    a[0] = *reinterpret_cast<const uint32_t*>(ptr_r0_c0);
    a[1] = *reinterpret_cast<const uint32_t*>(ptr_r1_c0);
    a[2] = *reinterpret_cast<const uint32_t*>(ptr_r0_c1);
    a[3] = *reinterpret_cast<const uint32_t*>(ptr_r1_c1);
}

// 按照 NVIDIA PTX 官方规范 (Figure 90 & 91)：从 SMEM (Row-Major [TILE_K][TILE_N]) 加载 32x8 的 Matrix B 片段
__device__ __forceinline__ void load_b_k32n8_fp8(const __nv_fp8_e4m3* tile_ptr, int ldB, uint32_t b[2], int lane_id) {
    // 1. 计算当前线程负责的 N 轴列号 (0~7)
    int col = lane_id / 4; 
    int k_group = lane_id % 4;

    // 2. 计算 K 轴索引基址
    int k0 = k_group * 4;       // K 轴前 16 步长 (Figure 90)
    int k1 = k_group * 4 + 16;  // K 轴后 16 步长 (Figure 91)

    // 3. 沿 K 轴提取同一列上连续 4 个 FP8 字节，并拼接为 uint32_t
    auto pack_k4 = [&](int k_start) -> uint32_t {
        uint8_t bytes[4] = {
            *reinterpret_cast<const uint8_t*>(tile_ptr + (k_start + 0) * ldB + col),
            *reinterpret_cast<const uint8_t*>(tile_ptr + (k_start + 1) * ldB + col),
            *reinterpret_cast<const uint8_t*>(tile_ptr + (k_start + 2) * ldB + col),
            *reinterpret_cast<const uint8_t*>(tile_ptr + (k_start + 3) * ldB + col)
        };
        return *reinterpret_cast<uint32_t*>(bytes);
    };

    b[0] = pack_k4(k0);
    b[1] = pack_k4(k1);
}

__global__ void __launch_bounds__(256, 2) fused_fp8_tensor_core_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ gmem_A,
    const __nv_fp8_e4m3* __restrict__ gmem_B,
    float* __restrict__ gmem_C,
    float scale_a, float scale_b,
    int M, int N, int K) {

    // A: Row-Major [TILE_M][TILE_K + PADDING_K]
    __shared__ alignas(16) __nv_fp8_e4m3 smem_A[2][TILE_M * (TILE_K + PADDING_K)];
    // B: Row-Major [TILE_K][TILE_N + PADDING_N]
    __shared__ alignas(16) __nv_fp8_e4m3 smem_B[2][TILE_K * (TILE_N + PADDING_N)];

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int block_row   = blockIdx.y * TILE_M;
    int block_col   = blockIdx.x * TILE_N;
    int num_tiles   = (K + TILE_K - 1) / TILE_K;

    auto load_async_fp8 = [&](int stage, int k_offset) {
        // 仅当 K、N 均为 16 的倍数时, 全局内存每次 16B 访存才保证不跨行且对齐;
        // 否则退化为 4B 粒度拷贝 (要求 K、N 为 4 的倍数, 此时谓词零填充仍能精确切齐边界)
        bool vec16 = (K % 16 == 0) && (N % 16 == 0);

        if (vec16) {
            // 1. 加载 A 到 Shared Memory
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int vec_idx = tid + i * 256;
                int a_row   = vec_idx / 4;
                int a_col   = (vec_idx % 4) * 16;

                int g_r = block_row + a_row;
                int g_c = k_offset + a_col;

                bool valid      = (g_r < M) && (g_c < K);
                const void* src = gmem_A + g_r * K + g_c;
                void* dst       = &smem_A[stage][a_row * (TILE_K + PADDING_K) + a_col];

                pipeline::cp_async_16bytes_zfill(dst, src, valid);
            }

            // 2. 加载 B 到 Shared Memory: gmem_B 尺寸 [K, N]，在 N 轴连续读 16 字节
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int vec_idx = tid + i * 256;
                int b_row   = vec_idx / 4;           // 0..63 (TILE_K)
                int b_col   = (vec_idx % 4) * 16;   // 0,16,32,48 (TILE_N)

                int g_k = k_offset + b_row;
                int g_n = block_col + b_col;

                bool valid      = (g_k < K) && (g_n < N);
                const void* src = gmem_B + g_k * N + g_n;
                void* dst       = &smem_B[stage][b_row * (TILE_N + PADDING_N) + b_col];

                pipeline::cp_async_16bytes_zfill(dst, src, valid);
            }
        } else {
            // 非 vec16: 退化为 4B 粒度拷贝 (要求 K、N 为 4 的倍数, 谓词零填充处理边界)
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int vec_idx = tid + i * 256;
                int a_row   = vec_idx / 16;
                int a_col   = (vec_idx % 16) * 4;

                int g_r = block_row + a_row;
                int g_c = k_offset + a_col;

                bool valid      = (g_r < M) && (g_c < K);
                const void* src = gmem_A + g_r * K + g_c;
                void* dst       = &smem_A[stage][a_row * (TILE_K + PADDING_K) + a_col];

                pipeline::cp_async_4bytes_zfill(dst, src, valid);
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int vec_idx = tid + i * 256;
                int b_row   = vec_idx / 32;
                int b_col   = (vec_idx % 32) * 4;

                int g_k = k_offset + b_row;
                int g_n = block_col + b_col;

                bool valid      = (g_k < K) && (g_n < N);
                const void* src = gmem_B + g_k * N + g_n;
                void* dst       = &smem_B[stage][b_row * (TILE_N + PADDING_N) + b_col];

                pipeline::cp_async_4bytes_zfill(dst, src, valid);
            }
        }
        pipeline::cp_async_commit();
    };

    // ================= Warp 划分 =================
    // 8 warps = 4 (M 方向) x 2 (N 方向), 每个 warp 负责 32 行 x 64 列输出 tile
    //   内部再分 2 个 m_sub x 8 个 n_sub, 每个 m_sub 处理 16 行, 每个 n_sub 处理 8 列
    int warp_row_idx    = warp_id / 2;     // 0..3
    int warp_col_idx    = warp_id % 2;     // 0..1
    int warp_row_offset = warp_row_idx * 32;   // 0, 32, 64, 96
    int warp_col_offset = warp_col_idx * 64;   // 0, 64

    // ================= 累加器 =================
    // 2 m_sub x 8 n_sub x 4 f32 (每 mma 输出 16x8 = 4 elements/thread)
    float accum[2][8][4] = {0.0f};

    // ================= 计算单步 (k_step) =================
    auto compute_k_step = [&](int read_stage, int k_step) {
        uint32_t reg_a[2][4];
        uint32_t reg_b[8][2];

        // A 片段: 2 次 load_a_m16k32_fp8 (m_sub=0,1)
        //   每个 m_sub 负责 16 行, K 方向偏移 k_step*32
        #pragma unroll
        for (int m_sub = 0; m_sub < 2; ++m_sub) {
            const __nv_fp8_e4m3* ptr_a =
                &smem_A[read_stage][(warp_row_offset + m_sub * 16) * (TILE_K + PADDING_K) + k_step * 32];
            load_a_m16k32_fp8(ptr_a, TILE_K + PADDING_K, reg_a[m_sub], lane_id);
        }

        // B 片段: 8 次 load_b_k32n8_fp8 (n_sub=0..7)
        //   每个 n_sub 负责 8 列, K 方向偏移 k_step*32
        #pragma unroll
        for (int n_sub = 0; n_sub < 8; ++n_sub) {
            int n_base = warp_col_offset + n_sub * 8;
            const __nv_fp8_e4m3* ptr_b =
                &smem_B[read_stage][k_step * 32 * (TILE_N + PADDING_N) + n_base];
            load_b_k32n8_fp8(ptr_b, TILE_N + PADDING_N, reg_b[n_sub], lane_id);
        }

        // MMA: 2 x 8 = 16 次 mma.sync.m16n8k32
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

    // ================= Prologue: 预加载第一个 tile =================
    int write_stage = 0;
    load_async_fp8(write_stage, 0);
    pipeline::cp_async_wait_pending<0>();
    __syncthreads();

    // ================= Main Loop: 双缓冲 (朴素 baseline) =================
    for (int tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        int read_stage = write_stage;
        write_stage ^= 1;
        int next_k_offset = (tile_idx + 1) * TILE_K;

        // 发起下一阶段加载 (异步, 不等待)
        load_async_fp8(write_stage, next_k_offset);

        // 计算当前阶段
        #pragma unroll
        for (int k_step = 0; k_step < TILE_K / 32; ++k_step) {
            compute_k_step(read_stage, k_step);
        }

        // 等待下一阶段加载完成
        pipeline::cp_async_wait_pending<0>();
        __syncthreads();
    }

    // ================= 收尾 Tile =================
    #pragma unroll
    for (int k_step = 0; k_step < TILE_K / 32; ++k_step) {
        compute_k_step(write_stage, k_step);
    }

    // ================= Epilogue: 写回 C =================
    // D 片段布局 (Figure 92/93): lane t, group=t>>2, tid_in_group=t%4
    //   d0: (row=group, col=tid*2+0),  d1: (row=group, col=tid*2+1)
    //   d2: (row=group+8, col=tid*2+0), d3: (row=group+8, col=tid*2+1)
    float total_scale = scale_a * scale_b;
    int group_id      = lane_id / 4;
    int lane_in_group = lane_id % 4;

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

void launch_fused_fp8_tensor_core_gemm_v0(const __nv_fp8_e4m3* d_A,
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
           