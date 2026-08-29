#include "fused_fp8_gemm0.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <iomanip>
#include <cublasLt.h>
#include "common.h"

// ============================================================================
// Naive FP8 GEMM Kernel 实现
// ============================================================================
__global__ void naive_fp8_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    float* __restrict__ C,
    float scale_a, float scale_b,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = static_cast<float>(A[row * K + k]);
            float b_val = static_cast<float>(B[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum * scale_a * scale_b;
    }
}

inline void launch_naive_fp8_gemm(
    const __nv_fp8_e4m3* d_A,
    const __nv_fp8_e4m3* d_B,
    float* d_C,
    float scale_a, float scale_b,
    int M, int N, int K,
    cudaStream_t stream) {
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_fp8_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, scale_a, scale_b, M, N, K);
}

// ============================================================================
// 辅助工具与结构定义
// ============================================================================
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

// L2 Cache 刷新器，严格控制 Cold-Cache 变量
class L2Flusher {
private:
    void* buffer_ = nullptr;
    size_t size_ = 0;
public:
    L2Flusher() {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        size_ = prop.l2CacheSize * 2;
        if (size_ == 0) size_ = 64 * 1024 * 1024;
        cudaMalloc(&buffer_, size_);
    }
    ~L2Flusher() { if (buffer_) cudaFree(buffer_); }
    void flush(cudaStream_t stream) {
        cudaMemsetAsync(buffer_, 0, size_, stream);
    }
};

// cuBLASLt 封装：执行标准的 Row-Major FP8 GEMM
class CublasLtFP8Gemm {
private:
    cublasLtHandle_t handle_;
    void* workspace_ = nullptr;
    size_t workspace_size_ = 32 * 1024 * 1024; // 32MB Workspace

public:
    CublasLtFP8Gemm() {
        cublasLtCreate(&handle_);
        cudaMalloc(&workspace_, workspace_size_);
    }
    ~CublasLtFP8Gemm() {
        if (workspace_) cudaFree(workspace_);
        cublasLtDestroy(handle_);
    }

    void run(const __nv_fp8_e4m3* d_A,
             const __nv_fp8_e4m3* d_B,
             float* d_C,
             float scale_a, float scale_b,
             int M, int N, int K,
             cudaStream_t stream) {
        
        cublasLtMatmulDesc_t operationDesc = nullptr;
        cublasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr;

        cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
        cublasLtMatmulDescCreate(&operationDesc, computeType, CUDA_R_32F);

        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transb, sizeof(transb));
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transa, sizeof(transa));

        cublasLtMatrixLayoutCreate(&adesc, CUDA_R_8F_E4M3, K, N, N);
        cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_8F_E4M3, M, K, K);
        cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32F, M, N, N);

        float alpha = scale_a * scale_b;
        float beta = 0.0f;

        cublasLtMatmul(handle_, operationDesc,
                       &alpha, d_B, adesc,
                       d_A, bdesc, &beta,
                       d_C, cdesc, d_C, cdesc,
                       nullptr, workspace_, workspace_size_, stream);

        cublasLtMatrixLayoutDestroy(adesc);
        cublasLtMatrixLayoutDestroy(bdesc);
        cublasLtMatrixLayoutDestroy(cdesc);
        cublasLtMatmulDescDestroy(operationDesc);
    }
};

// CPU Ground Truth 计算
void cpu_fp8_ground_truth(
    const std::vector<__nv_fp8_e4m3>& A,
    const std::vector<__nv_fp8_e4m3>& B,
    std::vector<float>& C,
    float scale_a, float scale_b,
    int M, int N, int K) {
    
    float total_scale = scale_a * scale_b;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = static_cast<float>(A[i * K + k]);
                float b_val = static_cast<float>(B[k * N + j]);
                sum += a_val * b_val;
            }
            C[i * N + j] = sum * total_scale;
        }
    }
}

// 结果验证函数
bool verify_result(const std::string& name, const std::vector<float>& h_gpu, const std::vector<float>& h_ref, float atol, float rtol) {
    float max_diff = 0.0f;
    bool pass = true;
    for (size_t i = 0; i < h_gpu.size(); ++i) {
        float diff = std::abs(h_gpu[i] - h_ref[i]);
        if (diff > max_diff) max_diff = diff;
        
        float ref = std::abs(h_ref[i]);
        if (diff > (atol + rtol * ref)) {
            pass = false;
            std::cout << "❌ [" << name << "] Mismatch at index " << i 
                      << " | GPU: " << h_gpu[i] << " vs CPU: " << h_ref[i] << std::endl;
            break;
        }
    }
    if (pass) {
        std::cout << "✅ [" << name << "] 通过校验！Max Diff: " << max_diff << std::endl;
    } else {
        std::cout << "❌ [" << name << "] 未通过校验！Max Diff: " << max_diff << std::endl;
    }
    return pass;
}

// ============================================================================
// 测试与基准测量主体
// ============================================================================
void run_test_and_benchmark(int M, int N, int K) {
    float scale_a = 0.025f, scale_b = 0.015f;

    std::cout << "\n=======================================================================" << std::endl;
    std::cout << ">>> 正在测试 Shape (M x N x K): " << M << " x " << N << " x " << K << std::endl;
    std::cout << "=======================================================================" << std::endl;

    size_t num_A = M * K, num_B = K * N, num_C = M * N;

    // 1. 初始化统一的 CPU 随机数据
    std::vector<__nv_fp8_e4m3> h_A(num_A);
    std::vector<__nv_fp8_e4m3> h_B(num_B);
    std::vector<float> h_C_cpu(num_C);
    std::vector<float> h_C_naive(num_C);
    std::vector<float> h_C_custom(num_C);
    std::vector<float> h_C_cublas(num_C);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < num_A; ++i) h_A[i] = __nv_fp8_e4m3(dist(rng));
    for (size_t i = 0; i < num_B; ++i) h_B[i] = __nv_fp8_e4m3(dist(rng));

    // 2. 分配显存
    auto d_A = make_cuda_unique<__nv_fp8_e4m3>(num_A);
    auto d_B = make_cuda_unique<__nv_fp8_e4m3>(num_B);
    auto d_C = make_cuda_unique<float>(num_C);

    CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), num_A * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.get(), h_B.data(), num_B * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    L2Flusher flusher;
    CublasLtFP8Gemm cublas_gemm;

    // 3. 计算 CPU Ground Truth
    std::cout << ">>> [1/2] 正在计算 CPU Ground Truth..." << std::endl;
    cpu_fp8_ground_truth(h_A, h_B, h_C_cpu, scale_a, scale_b, M, N, K);

    // 4. 正确性校验
    const float atol = scale_a * scale_b; 
    const float rtol = 0.10f;             

    // 4.1 Naive Kernel 校验
    launch_naive_fp8_gemm(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_C_naive.data(), d_C.get(), num_C * sizeof(float), cudaMemcpyDeviceToHost));
    verify_result("Naive Kernel", h_C_naive, h_C_cpu, atol, rtol);

    // 4.2 Custom Kernel 校验
    launch_fused_fp8_tensor_core_gemm_v0(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C.get(), num_C * sizeof(float), cudaMemcpyDeviceToHost));
    verify_result("Custom TensorCore", h_C_custom, h_C_cpu, atol, rtol);

    // 4.3 cuBLASLt 校验
    cublas_gemm.run(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C.get(), num_C * sizeof(float), cudaMemcpyDeviceToHost));
    verify_result("cuBLASLt FP8", h_C_cublas, h_C_cpu, atol, rtol);

    // 5. Benchmark 性能基准测试
    std::cout << "\n>>> [2/2] 启动 Cold L2-Cache Benchmark (均采用冷缓存机制)..." << std::endl;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int bench_iters = 20;
    double flops = 2.0 * M * N * K;

    auto benchmark_kernel = [&](const std::string& name, auto launch_func) {
        float total_ms = 0.0f;
        for (int i = 0; i < bench_iters; ++i) {
            flusher.flush(stream); // 清空 L2 Cache
            
            CUDA_CHECK(cudaEventRecord(start, stream));
            launch_func();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            
            CUDA_CHECK(cudaEventSynchronize(stop));
            float iter_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start, stop));
            total_ms += iter_ms;
        }
        float avg_ms = total_ms / bench_iters;
        double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

        std::cout << std::left << std::setw(22) << name 
                  << " | Latency: " << std::fixed << std::setprecision(4) << avg_ms << " ms"
                  << " | Performance: " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;
    };

    benchmark_kernel("Naive Kernel", [&]() {
        launch_naive_fp8_gemm(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
    });

    benchmark_kernel("Custom TensorCore", [&]() {
        launch_fused_fp8_tensor_core_gemm_v0(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
    });

    benchmark_kernel("cuBLASLt FP8", [&]() {
        cublas_gemm.run(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
    });

    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

int main() {
    run_test_and_benchmark(512, 512, 512);
    run_test_and_benchmark(2048, 2048, 2048);
    return 0;
}