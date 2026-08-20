#include "async_pipeline.cuh"
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>

// CPU Ground Truth
void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 朴素 Baseline Kernel
__global__ void naive_gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_naive_gemm_local(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError()); // 抓取启动错误
}

int main() {
    const int num_iters = 20;

    // 为了排查问题，我们先用简单对齐的标准 512x512x512 Shape 校验
    int M = 512; 
    int N = 512; 
    int K = 512;

    std::cout << "=================================================" << std::endl;
    std::cout << ">>> Running Async GEMM Debugging & Diagnostics..." << std::endl;
    std::cout << ">>> Dimensions (M x N x K): " << M << " x " << N << " x " << K << std::endl;
    std::cout << "=================================================" << std::endl;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_gpu(M * N, 0.0f);
    std::vector<float> h_C_cpu(M * N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) h_B[i] = dist(rng);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C)); // 明确清空 C 内存

    // ------------------------------------------------------------------------
    // 1. 测 Baseline
    // ------------------------------------------------------------------------
    launch_naive_gemm_local(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_base, stop_base;
    CUDA_CHECK(cudaEventCreate(&start_base));
    CUDA_CHECK(cudaEventCreate(&stop_base));

    CUDA_CHECK(cudaEventRecord(start_base));
    for (int it = 0; it < num_iters; ++it) {
        launch_naive_gemm_local(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop_base));
    CUDA_CHECK(cudaEventSynchronize(stop_base));

    float total_base_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_base_time_ms, start_base, stop_base));
    float avg_base_time_ms = total_base_time_ms / num_iters;

    // ------------------------------------------------------------------------
    // 2. 测 Async Pipeline (带全套 Error 捕获)
    // ------------------------------------------------------------------------
    std::cout << ">>> Launching Async Pipeline Kernel..." << std::endl;
    launch_async_pipeline_gemm_stage(d_A, d_B, d_C, M, N, K);
    
    // 🌟【关键捕获】：查看你的 Kernel 抛出了什么异常！
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "❌ [CUDA KERNEL LAUNCH ERROR]: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_async, stop_async;
    CUDA_CHECK(cudaEventCreate(&start_async));
    CUDA_CHECK(cudaEventCreate(&stop_async));

    CUDA_CHECK(cudaEventRecord(start_async));
    for (int it = 0; it < num_iters; ++it) {
        launch_async_pipeline_gemm_stage(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop_async));
    CUDA_CHECK(cudaEventSynchronize(stop_async));

    float total_async_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_async_time_ms, start_async, stop_async));
    float avg_async_time_ms = total_async_time_ms / num_iters;

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // 3. 结果校验
    // ------------------------------------------------------------------------
    std::cout << ">>> Running CPU Verification..." << std::endl;
    cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);

    bool pass = true;
    float max_diff = 0.0f;
    int error_count = 0;

    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > max_diff) max_diff = diff;

        if (diff > 1e-3) {
            if (error_count < 5) {
                std::cout << "❌ Mismatch at index " << i 
                          << ": GPU=" << h_C_gpu[i] << ", CPU=" << h_C_cpu[i] 
                          << " (Diff: " << diff << ")" << std::endl;
            }
            pass = false;
            error_count++;
        }
    }

    std::cout << "\n==================== BENCHMARK RESULTS ====================" << std::endl;
    if (pass) {
        std::cout << "✅ [PASS] Correctness Verified!" << std::endl;
    } else {
        std::cout << "❌ [FAIL] Error Count: " << error_count << std::endl;
    }

    double flops = 2.0 * static_cast<double>(M) * N * K;
    double base_tflops = (flops / (avg_base_time_ms / 1000.0)) / 1e12;
    double async_tflops = (flops / (avg_async_time_ms / 1000.0)) / 1e12;
    float speedup = avg_base_time_ms / avg_async_time_ms;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Naive Baseline Time : " << avg_base_time_ms << " ms (" << base_tflops << " TFLOPS)" << std::endl;
    std::cout << "Async Pipeline Time : " << avg_async_time_ms << " ms (" << async_tflops << " TFLOPS)" << std::endl;
    std::cout << "Speedup             : " << speedup << " x" << std::endl;
    std::cout << "==========================================================" << std::endl;

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    return pass ? 0 : 1;
}