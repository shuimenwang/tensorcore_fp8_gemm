#include "fused_fp8_gemm.cuh"

// =============================================================================
// Tile 尺寸
// =============================================================================
#define TILE_M      128
#define TILE_N      128
#define TILE_K      64

// A 矩阵 smem 布局: [M][K] of FP8, 行 stride = TILE_K + PADDING_K
#define PADDING_K   16
#define A_STRIDE   (TILE_K + PADDING_K)            // 80 bytes/row

// B 矩阵 smem 布局: [K/2][N] of uint16 (2 FP8 沿 K 配对)
//   每个 16-bit 的低字节 = K=2r, 高字节 = K=2r+1
//   如此 ldmatrix.x2.trans 可一次加载 32x8 的 B 片段
// PADDING_N_B = 8: stride_16 = 136, stride_bytes = 272 (16 字节对齐, ldmatrix 要求)
//   272 = 17*16, 故 k_pair*272 始终 16 字节对齐
#define PADDING_N_B  8
#define B_STRIDE_16 (TILE_N + PADDING_N_B)          // 136 个 16-bit = 272 bytes/row
#define K_PAIRS     (TILE_K / 2)                     // 32

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

// =============================================================================
// B 矩阵 gmem -> smem 加载: 拆分为预取/落盘两相, 消除主循环中的 gmem 延迟暴露
//   布局: [K/2][N] of uint16 (2 FP8 沿 K 配对, 低字节=K=2r, 高字节=K=2r+1)
//   该交错布局无法用 cp.async 直写, 故拆成:
//     Phase 1 prefetch_b_tile : 8 x ld.global.b32 -> 寄存器 (发射后不等)
//     Phase 2 store_b_tile    : __byte_perm + st.shared.v2 (寄存器就绪后执行)
//   预取发射后紧接 mma 计算阶段, gmem 延迟被计算完全掩盖
// =============================================================================
constexpr int B_N_PER_CHUNK = 4;
constexpr int B_N_CHUNKS    = TILE_N / B_N_PER_CHUNK;                    // 32
constexpr int B_ITERS       = (K_PAIRS * B_N_CHUNKS + 255) / 256;        // 4

__device__ __forceinline__ uint32_t b_load_row4(
    const __nv_fp8_e4m3* __restrict__ gmem_B, int g_k, int g_n, int N, int K)
{
    uint32_t v = 0;
    if (g_k < K && g_n + 4 <= N) {
        v = *reinterpret_cast<const uint32_t*>(&gmem_B[g_k * N + g_n]);
    } else if (g_k < K) {
        uint8_t b0 = (g_n + 0 < N) ? *reinterpret_cast<const uint8_t*>(&gmem_B[g_k * N + g_n + 0]) : 0;
        uint8_t b1 = (g_n + 1 < N) ? *reinterpret_cast<const uint8_t*>(&gmem_B[g_k * N + g_n + 1]) : 0;
        uint8_t b2 = (g_n + 2 < N) ? *reinterpret_cast<const uint8_t*>(&gmem_B[g_k * N + g_n + 2]) : 0;
        uint8_t b3 = (g_n + 3 < N) ? *reinterpret_cast<const uint8_t*>(&gmem_B[g_k * N + g_n + 3]) : 0;
        v = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8)
          | (static_cast<uint32_t>(b2) << 16) | (static_cast<uint32_t>(b3) << 24);
    }
    return v;
}

// Phase 1: 全部 8 个 4 字节行加载发射到寄存器 (raw[2i]=K=2r 行, raw[2i+1]=K=2r+1 行)
__device__ __forceinline__ void prefetch_b_tile(
    const __nv_fp8_e4m3* __restrict__ gmem_B,
    int k_offset, int block_col, int N, int K, int tid,
    uint32_t (&raw)[2 * B_ITERS])
{
    #pragma unroll
    for (int i = 0; i < B_ITERS; ++i) {
        int flat_idx = tid + i * 256;
        int k_pair   = flat_idx / B_N_CHUNKS;       // 0..31
        int n_chunk  = flat_idx % B_N_CHUNKS;       // 0..31
        int n_start  = n_chunk * B_N_PER_CHUNK;

        int g_k0 = k_offset + 2 * k_pair;
        int g_n  = block_col + n_start;

        raw[2 * i]     = b_load_row4(gmem_B, g_k0,     g_n, N, K);
        raw[2 * i + 1] = b_load_row4(gmem_B, g_k0 + 1, g_n, N, K);
    }
}

// Phase 2: 寄存器数据 -> K 向配对打包 -> 写 smem [k_pair][n_start..n_start+3] of uint16
__device__ __forceinline__ void store_b_tile_to_smem(
    __nv_fp8_e4m3* __restrict__ smem_B, const uint32_t (&raw)[2 * B_ITERS], int tid)
{
    #pragma unroll
    for (int i = 0; i < B_ITERS; ++i) {
        int flat_idx = tid + i * 256;
        int k_pair   = flat_idx / B_N_CHUNKS;
        int n_start  = (flat_idx % B_N_CHUNKS) * B_N_PER_CHUNK;

        // 打包: 选择子 LSB-first (nibble i 选输出字节 i)
        //   out0 = (a[0], c[0], a[1], c[1]), out1 = (a[2], c[2], a[3], c[3])
        uint32_t out0 = __byte_perm(raw[2 * i], raw[2 * i + 1], 0x5140);
        uint32_t out1 = __byte_perm(raw[2 * i], raw[2 * i + 1], 0x7362);

        uint32_t byte_offset = (k_pair * B_STRIDE_16 + n_start) * 2;
        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_B + byte_offset));
        asm volatile(
            "st.shared.v2.b32 [%0], {%1, %2};\n"
            :
            : "r"(smem_addr), "r"(out0), "r"(out1)
        );
    }
}

// =============================================================================
// A 矩阵 smem -> register: 使用 ldmatrix.x4
//   A 在 smem 中为 [M][K] of FP8 row-major, 可直接被 ldmatrix.x4 读取
//   thread t (lane_id 0..31) 的地址指向: row=t%16, col=(t/16)*16 处的 16 字节
// =============================================================================
__device__ __forceinline__ void load_a_m16k32_ldmatrix(
    const __nv_fp8_e4m3* smem_A_base,
    uint32_t a[4],
    int lane_id)
{
    int row = lane_id % 16;
    int col = (lane_id / 16) * 16;
    const void* ptr = smem_A_base + row * A_STRIDE + col;
    ptx::ldmatrix_x4(&a[0], ptr);
}

// =============================================================================
// B 矩阵 smem -> register: 使用 ldmatrix.x2.trans
//   B 在 smem 中为 [K/2][N] of uint16 (K 方向配对)
//   ldmatrix.x2.trans 一次加载 2 个 8x8 的 16-bit 矩阵, 硬件转置
//   thread t (lane_id 0..15) 的地址指向: k_pair = k_pair_base + t, n = n_base 处的 16 字节
//   (lane 16..31 地址可任意, 此处复用 lane%16 的地址)
// =============================================================================
__device__ __forceinline__ void load_b_k32n8_ldmatrix(
    const __nv_fp8_e4m3* smem_B_base,
    uint32_t b[2],
    int k_pair_base,            // 该 k_step 的起始 k_pair (0 或 16)
    int n_base,                 // 该 n_sub 的起始 N (warp_col_offset + n_sub*8)
    int lane_id)
{
    int row = lane_id % 16;
    int k_pair = k_pair_base + row;
    // byte 偏移 = (k_pair * B_STRIDE_16 + n_base) * 2
    const void* ptr = smem_B_base + (k_pair * B_STRIDE_16 + n_base) * 2;
    ptx::ldmatrix_x2_trans(&b[0], ptr);
}

// =============================================================================
// 主 Kernel
// =============================================================================
__global__ void __launch_bounds__(256, 2) fused_fp8_tensor_core_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ gmem_A,
    const __nv_fp8_e4m3* __restrict__ gmem_B,
    float* __restrict__ gmem_C,
    float scale_a, float scale_b,
    int M, int N, int K) {

    // A: [TILE_M][A_STRIDE] of FP8 (row-major, K is inner)
    __shared__ alignas(16) __nv_fp8_e4m3 smem_A[2][TILE_M * A_STRIDE];
    // B: [K_PAIRS][B_STRIDE_16] of uint16 (2 FP8 packed along K)
    //   总字节 = K_PAIRS * B_STRIDE_16 * 2
    __shared__ alignas(16) __nv_fp8_e4m3 smem_B[2][K_PAIRS * B_STRIDE_16 * 2];

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    // ---- A 加载 (cp.async, 16 字节/次, 保持异步流水) ----
    auto load_a_async = [&](int stage, int k_offset) {
        bool vec16 = (K % 16 == 0) && (N % 16 == 0);
        if (vec16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int vec_idx = tid + i * 256;
                int a_row   = vec_idx / 4;          // 0..127
                int a_col   = (vec_idx % 4) * 16;    // 0,16,32,48

                int g_r = block_row + a_row;
                int g_c = k_offset + a_col;

                bool valid = (g_r < M) && (g_c < K);
                const void* src = gmem_A + g_r * K + g_c;
                void* dst = &smem_A[stage][a_row * A_STRIDE + a_col];
                pipeline::cp_async_16bytes_zfill(dst, src, valid);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int vec_idx = tid + i * 256;
                int a_row   = vec_idx / 16;
                int a_col   = (vec_idx % 16) * 4;

                int g_r = block_row + a_row;
                int g_c = k_offset + a_col;

                bool valid = (g_r < M) && (g_c < K);
                const void* src = gmem_A + g_r * K + g_c;
                void* dst = &smem_A[stage][a_row * A_STRIDE + a_col];
                pipeline::cp_async_4bytes_zfill(dst, src, valid);
            }
        }
    };

    // ---- B 加载 (两相: 预取到寄存器 -> 计算间隙后写 smem, 掩盖 gmem 延迟) ----
    uint32_t b_raw[2 * B_ITERS];

    // 组合加载: A (cp.async 异步) + B 预取 (ld.global 发射不等待)
    auto load_tile = [&](int stage, int k_offset) {
        load_a_async(stage, k_offset);
        prefetch_b_tile(gmem_B, k_offset, block_col, N, K, tid, b_raw);
        pipeline::cp_async_commit();
    };

    // B 落盘: 寄存器 -> 打包 -> smem (prefetch 之后任意时刻调用)
    auto store_b_tile = [&](int stage) {
        store_b_tile_to_smem(smem_B[stage], b_raw, tid);
    };

    int warp_row_idx = warp_id / 2;     // 0..3
    int warp_col_idx = warp_id % 2;     // 0..1
    int warp_row_offset = warp_row_idx * 32;
    int warp_col_offset = warp_col_idx * 64;

    float accum[2][8][4] = {0.0f};

    auto compute_k_step = [&](int read_stage, int k_step) {
        uint32_t reg_a[2][4];
        uint32_t reg_b[8][2];

        int k_pair_base = k_step * (TILE_K / 2 / 2);  // k_step=0 -> 0, k_step=1 -> 16

        // 1. A: 2 次 ldmatrix.x4 (m_sub=0,1)
        #pragma unroll
        for (int m_sub = 0; m_sub < 2; ++m_sub) {
            const __nv_fp8_e4m3* ptr_a =
                &smem_A[read_stage][(warp_row_offset + m_sub * 16) * A_STRIDE + k_step * 32];
            load_a_m16k32_ldmatrix(ptr_a, reg_a[m_sub], lane_id);
        }

        // 2. B: 8 次 ldmatrix.x2.trans (n_sub=0..7)
        #pragma unroll
        for (int n_sub = 0; n_sub < 8; ++n_sub) {
            int n_base = warp_col_offset + n_sub * 8;
            const __nv_fp8_e4m3* ptr_b = smem_B[read_stage];
            load_b_k32n8_ldmatrix(ptr_b, reg_b[n_sub], k_pair_base, n_base, lane_id);
        }

        // 3. MMA: 2 x 8 = 16 次 mma.sync.m16n8k32
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

    // ================= 预加载 (Prologue) =================
    int write_stage = 0;
    load_tile(write_stage, 0);
    store_b_tile(write_stage);
    pipeline::cp_async_wait_pending<0>();
    __syncthreads();

    // ================= 主循环 (Main Loop) =================
    for (int tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        int read_stage = write_stage;
        write_stage ^= 1;
        int next_k_offset = (tile_idx + 1) * TILE_K;

        // 发起下一阶段加载: A cp.async 异步; B 预取仅发射 ld.global (不等待)
        load_tile(write_stage, next_k_offset);

        // 计算当前阶段: mma 与 B 预取的 gmem 延迟、A 的 cp.async 重叠
        #pragma unroll
        for (int k_step = 0; k_step < TILE_K / 32; ++k_step) {
            compute_k_step(read_stage, k_step);
        }

        // B 落盘: 此时预取寄存器已就绪, 仅剩 perm + st.shared (无 gmem 等待)
        store_b_tile(write_stage);

        // 等待 A 的 cp.async 完成
        pipeline::cp_async_wait_pending<0>();
        __syncthreads();
    }

    // ================= 收尾 Tile =================
    #pragma unroll
    for (int k_step = 0; k_step < TILE_K / 32; ++k_step) {
        compute_k_step(write_stage, k_step);
    }

    // ================= Epilogue: 写回 C 矩阵 =================
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
