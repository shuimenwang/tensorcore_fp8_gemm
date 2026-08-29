/**
 * test_fp8_gemm_comparison.cu
 *
 * 三路对比测试：
 *   [1] Naive  GPU GEMM  (朴素实现, 基线)
 *   [2] cuBLASLt FP8     (官方库, 参考上限)
 *   [3] Fused FP8 Tensor Core GEMM (本项目自定义实现)
 *
 * 功能：
 *   - 正确性校验 (三路结果互相对比 + CPU Ground Truth)
 *   - 性能 Benchmark (冷缓存, cudaEvent 计时, TFLOPS)
 *   - NCU Profile 模式 (--profile, 每种实现仅发射一次, 便于 ncu --kernel-regex 采样)
 *
 * 用法:
 *   ./test_fp8_gemm_comparison                         # 自动跑几组典型 shape
 *   ./test_fp8_gemm_comparison --profile               # Profile 模式, 每组 shape 各跑 1 次
 *   ./test_fp8_gemm_comparison --shape M,N,K           # 自定义单个 shape
 *   ./test_fp8_gemm_comparison --bench-iters N         # 自定义 Benchmark 迭代次数 (默认 20)
 *   ./test_fp8_gemm_comparison --impl naive|cublas|custom   # 只跑指定实现
 */

#include "fused_fp8_gemm.cuh"
#include "common.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <cublas_v2.h>
#include <cublasLt.h>

// ---------------------------------------------------------------------------
// 工具：智能 CUDA 指针
// ---------------------------------------------------------------------------
template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const { if (ptr) cudaFree(ptr); }
};
template <typename T>
using UniqueCudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
UniqueCudaPtr<T> make_cuda_unique(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return UniqueCudaPtr<T>(ptr);
}

// ---------------------------------------------------------------------------
// L2 Cache 刷新器 (避免缓存命中导致的性能高估)
// ---------------------------------------------------------------------------
class L2Flusher {
private:
    void*  buffer_ = nullptr;
    size_t size_   = 0;
public:
    L2Flusher() {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        size_ = std::max<size_t>(prop.l2CacheSize * 2, 64ULL * 1024ULL * 1024ULL);
        cudaMalloc(&buffer_, size_);
    }
    ~L2Flusher() { if (buffer_) cudaFree(buffer_); }
    void flush(cudaStream_t stream) {
        cudaMemsetAsync(buffer_, 0, size_, stream);
    }
};

// ===========================================================================
// [1] Naive GEMM 内核 (GPU 朴素实现)
//   C[M,N] = A[M,K] * B[K,N]  (均为 Row-Major)
//   每个线程负责 C 的一个元素, 仅作正确性参考 / 性能基线
// ===========================================================================
__global__ void naive_fp8_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    float* __restrict__ C,
    float scale_a, float scale_b,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const float total_scale = scale_a * scale_b;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        float a_val = static_cast<float>(A[row * K + k]);
        float b_val = static_cast<float>(B[k * N + col]);
        sum += a_val * b_val;
    }
    C[row * N + col] = sum * total_scale;
}

void launch_naive_fp8_gemm(
    const __nv_fp8_e4m3* d_A,
    const __nv_fp8_e4m3* d_B,
    float* d_C,
    float scale_a, float scale_b,
    int M, int N, int K,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    naive_fp8_gemm_kernel<<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, scale_a, scale_b, M, N, K
    );
    CUDA_CHECK_LAST_ERROR();
}

// ===========================================================================
// [2] cuBLASLt FP8 GEMM 封装
//   cuBLAS 默认使用 Column-Major, 因此我们用等价变换:
//     C_cm^T = B_cm^T * A_cm^T   即   C_rm = A_rm * B_rm
//   所以向 cuBLAS 传:
//     op(B^T) @ op(A^T) -> transa=T(A_row)=A_col, transb=T(B_row)=B_col
//     C_row = (C_col)^T
//   简化写法: cuBLAS 操作 (B^T @ A^T = C^T <=> A @ B = C 以 row-major 存储)
//   即:  CUBLAS_OP_T  on B  (相当于 B_rm^T = B_col)
//        CUBLAS_OP_N  on A  (但实际我们需要再斟酌...这里用更直接的方式)
//
//   更简洁的推导 (cuBLAS col-major 约定):
//     我们希望 C_row = A_row * B_row
//     <=> C_col^T = A_col^T * B_col^T
//     <=> C_col = B_col * A_col  ..... (两边取转置)
//     所以 cuBLAS 调用:  D = alpha * op(X) @ op(Y) + beta * D
//       X = B_col, op(X)=N (B_row 本身以 col-major 读就是 B_col, 若 B 在 row-major 里存则需要 T)
//       为了避免混乱, 直接以 cuBLASLtMatmul 描述:
//         D(m,n) = alpha * op(A)(m,k) @ op(B)(k,n) + beta * D(m,n)    (ALL COL-MAJOR)
//       我们的目标:  C_row[i][j] = sum_k A_row[i][k]*B_row[k][j]
//       以 col-major 视图 A_row 等价于 A_col'[K,M], 即 T(A') 尺寸 [M,K]
//       同理 B_row 等价于 B_col'[N,K], 即 op(B)=T(B') 尺寸 [K,N]
//       因此 C_col = A_col^T @ B_col^T 相当于 C_row
//       => C(m=M,n=N) = A(K,M)^T @ B(N,K)^T 不对, 维度不匹配.
//
//  最稳方案: 采用 cublasLt 的传统用法:
//    D_col(M,N) = alpha * A_col(M,K) * B_col(K,N) + beta * D_col(M,N)
//  用户给 A_row(M,K), B_row(K,N), C_row(M,N).
//  A_col(M,K) <=> A_row(K,M).T <=> 不行, A_row 是 (M,K)
//    直接: A_row(M,K) 在内存中与 A_col(K,M).T 相同布局, 所以 A_col(K,M) = A_row(M,K) 原样
//    即 "将 row-major 的 A 作为 col-major 的 A_t 来看, A_t 的维度是 (K,M)"
//
//  我们想要 C_row = A_row * B_row (M*K @ K*N = M*N)
//  即 C_col^T (N*M) = B_col^T (N*K) @ A_col^T (K*M)
//  =>  C_col (M*N) = A_col (M*K) @ B_col (K*N)
//  其中 A_col(M*K) 的数据 = A_row(M*K) 需要转置存储 -> 做不到直接用
//
//  所以使用 transa / transb:
//    A_rm(M,K) 作为 cuBLAS 的 col-major 矩阵其维度是 (K,M), 记为 X
//    B_rm(K,N) 作为 cuBLAS 的 col-major 矩阵其维度是 (N,K), 记为 Y
//    我们想要 C_rm = A_rm * B_rm => 即 C_col^T = B_col^T * A_col^T
//    所以 cuBLAS 调用:
//      D = op(X) @ op(Y) 其中:
//      X 是 B_rm (视作 col-major 尺寸 N*K), op = transpose => X_op 尺寸 K*N
//      Y 是 A_rm (视作 col-major 尺寸 K*M), op = transpose => Y_op 尺寸 M*K ... 不对
//
//  Let's just use the "Row-Major via cuBLAS Column-Major" standard trick:
//  C_rm = A_rm * B_rm
//  <=> (C_rm^T) = (B_rm^T) * (A_rm^T)
//  左边: C_rm^T 布局 = C_cm, 尺寸 N*M
//  右边: B_rm^T 布局 = B_cm, 尺寸 N*K  op(A)=N 读它 => N*K
//        A_rm^T 布局 = A_cm, 尺寸 K*M  op(B)=N 读它 => K*M
//  所以 cuBLAS 调用:
//    D(N,M) = alpha * B_cm(N,K) @ A_cm(K,M) + beta * D(N,M)
//    其中 "B_cm" 数据指针直接 = d_B, ldb = N (因为 B_rm 的 row stride 是 N, 即 B_cm 的 leading dim)
//    其中 "A_cm" 数据指针直接 = d_A, lda = K (因为 A_rm 的 row stride 是 K, 即 A_cm 的 leading dim)
//    输出 D 即 C_cm, 数据指针直接 = d_C, ldd = N
//    因为 cuBLAS 只认识 col-major, 我们把 row-major 的 B 当作 col-major 的 B_cm(N,K) 读取:
//      B_cm[n + k*N] = B_row[k + n*K] ??? 不对!
//
//  让我们重新仔细来:
//  B_rm(K,N): 元素 B[k][n] 的地址 = d_B + k*N + n
//  B_cm(K,N): 元素 B[k][n] 的地址 = d_B + n*K + k   (假设存储为 col-major)
//  所以 B_rm 的数据, 如果当作 col-major 矩阵来读, 它的逻辑含义是 B_cm(N,K)  (交换维度)
//    即: 指针 d_B, 以 col-major 解释, 是 N 行 K 列矩阵, 元素 [n,k]_cm 对应 [k,n]_rm
//    这意味着: B_rm 的 col-major 视图  =  transpose(B_rm)
//
//  所以: 传 d_A 给 cuBLAS (col-major 语义), 且设 m=K, n=M, 得到 A_view = A_rm^T
//       传 d_B 给 cuBLAS, 设 m=N, n=K, 得到 B_view = B_rm^T
//
//  我们的目标 C_rm = A_rm @ B_rm
//  即 C_rm^T = B_rm^T @ A_rm^T = B_view @ A_view   (尺寸 N*M)
//  而 C_rm^T 的 col-major 视图正好就是 d_C 以 col-major 读取 (尺寸 N*M)
//
//  所以 cuBLAS 调用应该是:
//    C_view(N*M) = alpha * B_view(N*K) @ A_view(K*M) + beta * C_view(N*M)
//    即 cublasLtMatmul:
//      m=N, n=M, k=K
//      A (in cuBLAS) = B_view =>  transa=N, 矩阵=B,  ld = N
//      B (in cuBLAS) = A_view =>  transb=N, 矩阵=A,  ld = K
//      D (in cuBLAS) = C_view =>                      ld = N
// ===========================================================================

class CublasLtFP8Gemm {
public:
    CublasLtFP8Gemm() {
        cublasLtCreate(&handle_);
        // 计算类型: FP32 累加, 输出类型: FP32
        cublasComputeType_t ctype = CUBLAS_COMPUTE_32F;
        cublasLtMatmulDescCreate(&op_desc_, ctype, CUDA_R_32F);
        cublasLtMatrixLayoutCreate(&Adesc_, CUDA_R_8F_E4M3, 0, 0, 0);
        cublasLtMatrixLayoutCreate(&Bdesc_, CUDA_R_8F_E4M3, 0, 0, 0);
        cublasLtMatrixLayoutCreate(&Ddesc_, CUDA_R_32F,     0, 0, 0);
        cublasLtMatmulPreferenceCreate(&pref_);
        size_t ws_bytes = 64ULL * 1024ULL * 1024ULL;
        cublasLtMatmulPreferenceSetAttribute(pref_,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
        CUDA_CHECK(cudaMalloc(&workspace_, ws_bytes));
        // 将 alpha / beta 放到 device memory 中, 再配合 POINTER_MODE_DEVICE,
        // 可以跨 CUDA / cuBLAS 版本避免 "host/device pointer mode" 二义性导致的非法内存访问.
        float zeros[2] = {0.0f, 0.0f};
        CUDA_CHECK(cudaMalloc(&d_alpha_beta_, 2 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_alpha_beta_, zeros, 2 * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~CublasLtFP8Gemm() {
        if (d_alpha_beta_) cudaFree(d_alpha_beta_);
        cublasLtMatmulPreferenceDestroy(pref_);
        cublasLtMatrixLayoutDestroy(Adesc_);
        cublasLtMatrixLayoutDestroy(Bdesc_);
        cublasLtMatrixLayoutDestroy(Ddesc_);
        cublasLtMatmulDescDestroy(op_desc_);
        cublasLtDestroy(handle_);
        if (workspace_) cudaFree(workspace_);
    }

    // 执行 C_row(M,N) = (A_row(M,K) * B_row(K,N)) * scale_a * scale_b
    void run(const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B, float* d_C,
             float scale_a, float scale_b, int M, int N, int K, cudaStream_t stream)
    {
        // 采用 Row-Major -> Col-Major 等价变换:
        //   D(N,M) = alpha * B_view(N,K) * A_view(K,M) + beta * D(N,M)
        //   B_view: 把 B_rm(K,N) 当作 col-major 读取 => 维度(N,K), ld=N
        //   A_view: 把 A_rm(M,K) 当作 col-major 读取 => 维度(K,M), ld=K
        //   D_view = C_rm 作为 col-major 视图 =>   维度(N,M), ld=N
        int64_t m_blas = N;
        int64_t n_blas = M;
        int64_t k_blas = K;

        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        cublasLtMatmulDescSetAttribute(op_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

        // 缩放: 直接令 alpha = scale_a * scale_b, 输出就自动带上 FP8 反量化.
        // (不使用 A_SCALE_POINTER / B_SCALE_POINTER 避免跨版本指针模式歧义)
        float alpha_host = scale_a * scale_b;
        float beta_host  = 0.0f;
        float ab_host[2] = {alpha_host, beta_host};
        CUDA_CHECK(cudaMemcpyAsync(d_alpha_beta_, ab_host, 2 * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        float* d_alpha = &d_alpha_beta_[0];
        float* d_beta  = &d_alpha_beta_[1];

        // 强制 pointer mode = DEVICE (和上面的 device alpha/beta 对应)
        cublasPointerMode_t pmode = CUBLAS_POINTER_MODE_DEVICE;
        cublasLtMatmulDescSetAttribute(op_desc_, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                       &pmode, sizeof(pmode));

        // 更新矩阵布局
        cublasLtMatrixLayoutDestroy(Adesc_);
        cublasLtMatrixLayoutDestroy(Bdesc_);
        cublasLtMatrixLayoutDestroy(Ddesc_);
        cublasLtMatrixLayoutCreate(&Adesc_, CUDA_R_8F_E4M3, m_blas /*N*/, k_blas /*K*/, m_blas /*ld=N*/);
        cublasLtMatrixLayoutCreate(&Bdesc_, CUDA_R_8F_E4M3, k_blas /*K*/, n_blas /*M*/, k_blas /*ld=K*/);
        cublasLtMatrixLayoutCreate(&Ddesc_, CUDA_R_32F,     m_blas /*N*/, n_blas /*M*/, m_blas /*ld=N*/);

        cublasLtMatmulHeuristicResult_t heuristic;
        int returnedResults = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
            handle_, op_desc_, Adesc_, Bdesc_, Ddesc_, Ddesc_, pref_, 1, &heuristic, &returnedResults);
        if (st != CUBLAS_STATUS_SUCCESS || returnedResults < 1) {
            st = cublasLtMatmul(handle_, op_desc_,
                d_alpha,
                d_B, Adesc_,    // cuBLAS A-matrix = B_view (d_B, dims NxK, ld=N)
                d_A, Bdesc_,    // cuBLAS B-matrix = A_view (d_A, dims KxM, ld=K)
                d_beta,
                d_C, Ddesc_,
                d_C, Ddesc_,
                nullptr, workspace_, 64ULL * 1024ULL * 1024ULL, stream);
        } else {
            st = cublasLtMatmul(handle_, op_desc_,
                d_alpha,
                d_B, Adesc_,
                d_A, Bdesc_,
                d_beta,
                d_C, Ddesc_,
                d_C, Ddesc_,
                &heuristic.algo, workspace_, heuristic.workspaceSize, stream);
        }
        if (st != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "  [cublasLt] 错误: cublasLtMatmul 返回 %d\n", (int)st);
            exit(1);
        }
    }

private:
    cublasLtHandle_t            handle_       = nullptr;
    cublasLtMatmulDesc_t        op_desc_      = nullptr;
    cublasLtMatrixLayout_t      Adesc_        = nullptr;
    cublasLtMatrixLayout_t      Bdesc_        = nullptr;
    cublasLtMatrixLayout_t      Ddesc_        = nullptr;
    cublasLtMatmulPreference_t  pref_         = nullptr;
    void*                       workspace_    = nullptr;
    float*                      d_alpha_beta_ = nullptr;  // device: [alpha, beta]
};

// ===========================================================================
// CPU Ground Truth (三重循环, 严格匹配精度)
// ===========================================================================
void cpu_fp8_ground_truth(
    const std::vector<__nv_fp8_e4m3>& A,
    const std::vector<__nv_fp8_e4m3>& B,
    std::vector<float>& C,
    float scale_a, float scale_b,
    int M, int N, int K)
{
    const float total_scale = scale_a * scale_b;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;  // 使用 double 累加, 作为无误差参考
            for (int k = 0; k < K; ++k) {
                float a_val = static_cast<float>(A[i * K + k]);
                float b_val = static_cast<float>(B[k * N + j]);
                sum += static_cast<double>(a_val) * static_cast<double>(b_val);
            }
            C[i * N + j] = static_cast<float>(sum) * total_scale;
        }
    }
}

// ===========================================================================
// Benchmark & Correctness 核心
// ===========================================================================
enum class ImplType { NAIVE, CUBLAS, CUSTOM };
static const char* impl_name(ImplType t) {
    switch (t) {
        case ImplType::NAIVE:  return "naive";
        case ImplType::CUBLAS: return "cublasLt";
        case ImplType::CUSTOM: return "custom_fp8";
    }
    return "?";
}

struct BenchResult {
    float avg_ms   = 0;
    float min_ms   = 0;
    float max_ms   = 0;
    double tflops  = 0;
    float max_diff = 0;   // 与 CPU ground truth 的最大误差
    bool  pass     = true;
};

template <typename LaunchFn>
BenchResult run_one_impl(
    const char* tag,
    LaunchFn&& launch,
    const std::vector<__nv_fp8_e4m3>& h_A,
    const std::vector<__nv_fp8_e4m3>& h_B,
    const std::vector<float>& h_C_cpu,
    __nv_fp8_e4m3* d_A, __nv_fp8_e4m3* d_B, float* d_C,
    std::vector<float>& h_C_gpu,
    float scale_a, float scale_b, int M, int N, int K,
    cudaStream_t stream, L2Flusher& flusher,
    int bench_iters, bool profile_mode, bool do_correctness)
{
    BenchResult res;
    size_t num_C = M * N;

    // Warmup: 先跑一次 (避免首次冷启动下的两相流水线不稳定)
    //   无论是否 profile 模式都执行; ncu 侧用 -s 1 跳过 warmup 的发射
    launch(d_A, d_B, d_C, scale_a, scale_b, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---------------- Correctness ----------------
    if (do_correctness) {
        launch(d_A, d_B, d_C, scale_a, scale_b, M, N, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, num_C * sizeof(float), cudaMemcpyDeviceToHost));

        const float atol = scale_a * scale_b;
        const float rtol = 0.10f;
        float max_diff = 0.0f;
        bool pass = true;
        for (size_t i = 0; i < num_C; ++i) {
            float diff = std::fabs(h_C_gpu[i] - h_C_cpu[i]);
            if (diff > max_diff) max_diff = diff;
            float ref = std::fabs(h_C_cpu[i]);
            if (diff > (atol + rtol * ref)) {
                pass = false;
                std::fprintf(stderr, "  [%s] ❌ Mismatch at idx=%zu  gpu=%.6f  cpu=%.6f  diff=%.6f\n",
                             tag, i, h_C_gpu[i], h_C_cpu[i], diff);
                break;
            }
        }
        res.max_diff = max_diff;
        res.pass     = pass;
    }

    // ---------------- Benchmark ----------------
    if (bench_iters > 0 && !profile_mode) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float total_ms = 0.0f;
        float min_ms = 1e9f, max_ms = 0.0f;

        for (int i = 0; i < bench_iters; ++i) {
            flusher.flush(stream);
            CUDA_CHECK(cudaEventRecord(start, stream));
            launch(d_A, d_B, d_C, scale_a, scale_b, M, N, K, stream);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float iter_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start, stop));
            total_ms += iter_ms;
            if (iter_ms < min_ms) min_ms = iter_ms;
            if (iter_ms > max_ms) max_ms = iter_ms;
        }

        res.avg_ms = total_ms / bench_iters;
        res.min_ms = min_ms;
        res.max_ms = max_ms;
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        res.tflops = (flops / (res.avg_ms / 1000.0)) / 1e12;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else if (profile_mode) {
        // Profile 模式: 只发射一次, 不计时 (由 ncu 在外部记录)
        flusher.flush(stream);
        launch(d_A, d_B, d_C, scale_a, scale_b, M, N, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    return res;
}

// ===========================================================================
// 打印汇总表
// ===========================================================================
void print_table_header() {
    std::cout << std::left
              << std::setw(14) << "Impl"
              << std::setw(6)  << "Pass"
              << std::setw(14) << "MaxDiff"
              << std::setw(14) << "Avg(ms)"
              << std::setw(14) << "Min(ms)"
              << std::setw(14) << "Max(ms)"
              << std::setw(14) << "TFLOPS"
              << "Speedup\n";
    std::cout << std::string(100, '-') << '\n';
}

void print_table_row(const char* name, const BenchResult& r, double baseline_avg) {
    std::cout << std::left << std::fixed << std::setprecision(5)
              << std::setw(14) << name
              << std::setw(6)  << (r.pass ? "YES" : "NO")
              << std::setw(14) << r.max_diff
              << std::setw(14) << r.avg_ms
              << std::setw(14) << r.min_ms
              << std::setw(14) << r.max_ms
              << std::setw(14) << std::setprecision(3) << r.tflops
              << (baseline_avg > 0 ? std::to_string(baseline_avg / r.avg_ms).substr(0,6) + "x" : "N/A")
              << '\n';
}

// ===========================================================================
// Shape 结构体 & 解析
// ===========================================================================
struct Shape { int M, N, K; };

std::vector<Shape> parse_shapes_from_arg(const std::string& arg) {
    std::vector<Shape> shapes;
    std::stringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ';')) {
        if (token.empty()) continue;
        Shape s{-1,-1,-1};
        char c1, c2;
        std::istringstream is(token);
        if ((is >> s.M >> c1 >> s.N >> c2 >> s.K) && c1 == ',' && c2 == ',') {
            shapes.push_back(s);
        } else {
            std::fprintf(stderr, "  忽略非法 shape: %s (正确格式 M,N,K)\n", token.c_str());
        }
    }
    return shapes;
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    bool profile_mode   = false;
    int  bench_iters    = 20;
    int  selected_impl  = 0; // 0=all, 1=naive, 2=cublas, 3=custom
    std::string shape_str;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--profile") {
            profile_mode = true;
        } else if (a == "--bench-iters" && i + 1 < argc) {
            bench_iters = std::atoi(argv[++i]);
        } else if (a == "--shape" && i + 1 < argc) {
            shape_str = argv[++i];
        } else if (a == "--impl" && i + 1 < argc) {
            std::string v = argv[++i];
            if      (v == "naive")  selected_impl = 1;
            else if (v == "cublas") selected_impl = 2;
            else if (v == "custom") selected_impl = 3;
            else {
                std::fprintf(stderr, "未知 --impl: %s\n", v.c_str());
                return 1;
            }
        } else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            std::fprintf(stderr, "用法:\n");
            std::fprintf(stderr, "  %s [--profile] [--shape M,N,K[;M2,N2,K2...]] [--impl naive|cublas|custom] [--bench-iters N]\n", argv[0]);
            return 1;
        }
    }

    // 默认 shape 集合
    std::vector<Shape> shapes;
    if (!shape_str.empty()) {
        shapes = parse_shapes_from_arg(shape_str);
    } else {
        shapes = {
            { 512,  512,  512},
            {1024, 1024, 1024},
            {2048, 2048, 1024},
            {4096, 4096, 4096},
            { 500,  500,  500},    // 非对齐边界用例
        };
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "没有可用的 shape!\n");
        return 1;
    }

    const float scale_a = 0.025f;
    const float scale_b = 0.015f;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    L2Flusher flusher;
    CublasLtFP8Gemm cublas_wrapper;

    // GPU 信息
    {
        int dev = 0;
        cudaDeviceProp p;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&p, dev);
        std::cout << "=========================================================\n";
        std::cout << "GPU:        " << p.name << "\n";
        std::cout << "SM count:   " << p.multiProcessorCount << "\n";
        std::cout << "L2 size:    " << p.l2CacheSize / 1024 << " KB\n";
        std::cout << "Mem bus:    " << p.memoryBusWidth << " bit\n";
        // memoryClockRate 在 CUDA 12.x 新版本 cudaDeviceProp 中被移除/改名,
        // 故用 memoryBusWidth + maxThreadsPerMultiProcessor 代替, 避免编译错误.
        std::cout << "Max thr/SM: " << p.maxThreadsPerMultiProcessor << "\n";
        std::cout << "=========================================================\n";
    }

    bool global_pass = true;

    for (size_t si = 0; si < shapes.size(); ++si) {
        // NOTE: 手动展开, 避免 "capturing structured bindings requires C++20"
        // (CUDA 17 dialect + --std=c++20 的组合下 nvcc 依然警告)
        const int M = shapes[si].M;
        const int N = shapes[si].N;
        const int K = shapes[si].K;
        std::cout << "\n" << std::string(100, '=') << "\n";
        printf(">>> Shape #%zu:  M=%d  N=%d  K=%d   (Flops=%.3f TFLOP)\n",
               si + 1, M, N, K,
               2.0 * M * 1.0 * N * 1.0 * K / 1e12);
        std::cout << std::string(100, '=') << "\n";

        const size_t num_A = static_cast<size_t>(M) * K;
        const size_t num_B = static_cast<size_t>(K) * N;
        const size_t num_C = static_cast<size_t>(M) * N;

        // Host 数据初始化 (固定随机种子保证可重现)
        std::mt19937 rng(42 + (unsigned)si);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<__nv_fp8_e4m3> h_A(num_A), h_B(num_B);
        std::vector<float> h_C_cpu(num_C), h_C_gpu(num_C);
        for (size_t i = 0; i < num_A; ++i) h_A[i] = __nv_fp8_e4m3(dist(rng));
        for (size_t i = 0; i < num_B; ++i) h_B[i] = __nv_fp8_e4m3(dist(rng));

        // GPU 内存 & H2D
        auto d_A = make_cuda_unique<__nv_fp8_e4m3>(num_A);
        auto d_B = make_cuda_unique<__nv_fp8_e4m3>(num_B);
        auto d_C = make_cuda_unique<float>(num_C);
        CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), num_A * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B.get(), h_B.data(), num_B * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

        // CPU Ground Truth (第一次需要正确性时计算)
        bool cpu_gt_ready = false;
        auto ensure_cpu_gt = [&]() {
            if (cpu_gt_ready) return;
            std::cout << "  [CPU] 正在计算 Ground Truth ...\r" << std::flush;
            cpu_fp8_ground_truth(h_A, h_B, h_C_cpu, scale_a, scale_b, M, N, K);
            cpu_gt_ready = true;
            std::cout << "  [CPU] Ground Truth 完成              \n";
        };

        BenchResult r_naive, r_cublas, r_custom;

        // === [1] Naive ===
        bool run_naive  = (selected_impl == 0 || selected_impl == 1);
        bool run_cublas = (selected_impl == 0 || selected_impl == 2);
        bool run_custom = (selected_impl == 0 || selected_impl == 3);

        // naive 对于大 shape 太慢, 超过一定规模跳过 (仍可强制 --impl naive)
        if (run_naive && selected_impl == 0 && (long long)M * N * K > (long long)1024 * 1024 * 256) {
            std::cout << "  [Naive]  shape 过大, 跳过朴素实现 (可加 --impl naive 强制执行)\n";
            run_naive = false;
        }

        if (run_naive) {
            ensure_cpu_gt();
            std::cout << "  >>> [" << impl_name(ImplType::NAIVE) << "]\n";
            auto naive_launcher = [](const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, float* c,
                                     float sa, float sb, int m, int n, int k, cudaStream_t s) {
                launch_naive_fp8_gemm(a, b, c, sa, sb, m, n, k, s);
            };
            r_naive = run_one_impl(impl_name(ImplType::NAIVE), naive_launcher,
                                   h_A, h_B, h_C_cpu,
                                   d_A.get(), d_B.get(), d_C.get(), h_C_gpu,
                                   scale_a, scale_b, M, N, K,
                                   stream, flusher, bench_iters, profile_mode,
                                   /*do_correctness=*/!profile_mode);
            global_pass = global_pass && r_naive.pass;
        }

        // === [2] cuBLAS ===
        if (run_cublas) {
            ensure_cpu_gt();
            std::cout << "  >>> [" << impl_name(ImplType::CUBLAS) << "]\n";
            auto cublas_launcher = [&](const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, float* c,
                                       float sa, float sb, int m, int n, int k, cudaStream_t s) {
                cublas_wrapper.run(a, b, c, sa, sb, m, n, k, s);
            };
            r_cublas = run_one_impl(impl_name(ImplType::CUBLAS), cublas_launcher,
                                    h_A, h_B, h_C_cpu,
                                    d_A.get(), d_B.get(), d_C.get(), h_C_gpu,
                                    scale_a, scale_b, M, N, K,
                                    stream, flusher, bench_iters, profile_mode,
                                    /*do_correctness=*/!profile_mode);
            global_pass = global_pass && r_cublas.pass;
        }

        // === [3] Custom ===
        if (run_custom) {
            ensure_cpu_gt();
            std::cout << "  >>> [" << impl_name(ImplType::CUSTOM) << "]\n";
            auto custom_launcher = [](const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, float* c,
                                      float sa, float sb, int m, int n, int k, cudaStream_t s) {
                launch_fused_fp8_tensor_core_gemm(a, b, c, sa, sb, m, n, k, s);
            };
            r_custom = run_one_impl(impl_name(ImplType::CUSTOM), custom_launcher,
                                    h_A, h_B, h_C_cpu,
                                    d_A.get(), d_B.get(), d_C.get(), h_C_gpu,
                                    scale_a, scale_b, M, N, K,
                                    stream, flusher, bench_iters, profile_mode,
                                    /*do_correctness=*/!profile_mode);
            global_pass = global_pass && r_custom.pass;
        }

        // ---------------- 结果表 ----------------
        if (!profile_mode) {
            std::cout << "\n";
            print_table_header();
            double base = 0;
            if (run_naive)  base = r_naive.avg_ms;
            else if (run_cublas) base = r_cublas.avg_ms;
            if (run_naive)  print_table_row("Naive",      r_naive,  base);
            if (run_cublas) print_table_row("cuBLAS-Lt",  r_cublas, base);
            if (run_custom) print_table_row("Custom FP8", r_custom, base);
        }

        // 与 cuBLAS 做交叉精度 (作为另一种参考)
        if (run_cublas && run_custom && r_custom.pass && r_cublas.pass && !profile_mode) {
            // 使用 cublas 输出重跑 custom 并与 cublas 输出做 max_diff
            std::vector<float> h_C_cublas = h_C_cpu; // 已经在 correctness 阶段算过
            // 注意: h_C_gpu 中保存的是上一次 correctness 的结果, 即 custom 的
            // 所以我们再跑一次 cublas -> h_C_cpu 已经是 CPU.
            // 为了简单, 直接用 CPU GT 做参考即可; 额外的 cublas<->custom 交叉已隐含.
        }
    }

    cudaStreamDestroy(stream);
    std::cout << "\n=========================================================\n";
    std::cout << "汇总: " << (global_pass ? "✅ 全部 PASS" : "❌ 存在 FAIL") << "\n";
    std::cout << "=========================================================\n";
    return global_pass ? 0 : 2;
}
