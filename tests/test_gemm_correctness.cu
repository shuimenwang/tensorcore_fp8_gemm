#include "fused_fp8_gemm.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include "common.h"



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

void run_test_and_benchmark(int M, int N, int K) {
    float scale_a = 0.025f, scale_b = 0.015f;

    std::cout << "\n=================================================" << std::endl;
    std::cout << ">>> 正在测试 Shape (M x N x K): " << M << " x " << N << " x " << K << std::endl;
    std::cout << "=================================================" << std::endl;

    size_t num_A = M * K, num_B = K * N, num_C = M * N;

    std::vector<__nv_fp8_e4m3> h_A(num_A);
    std::vector<__nv_fp8_e4m3> h_B(num_B);             // 原始 B [K, N]，供 CPU Ground Truth 使用
    std::vector<__nv_fp8_e4m3> h_B_transposed(num_B);  // 转置后的 B [N, K]，供 GPU 使用
    
    std::vector<float> h_C_gpu(num_C);
    std::vector<float> h_C_cpu(num_C);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < num_A; ++i) h_A[i] = __nv_fp8_e4m3(dist(rng));
    for (size_t i = 0; i < num_B; ++i) h_B[i] = __nv_fp8_e4m3(dist(rng));

    // 1. 在 CPU 端完成 B 矩阵的转置 [K, N] -> [N, K]
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            h_B_transposed[n * K + k] = h_B[k * N + n];
        }
    }

    auto d_A = make_cuda_unique<__nv_fp8_e4m3>(num_A);
    auto d_B = make_cuda_unique<__nv_fp8_e4m3>(num_B);
    auto d_C = make_cuda_unique<float>(num_C);

    // 2. 将转置后的 h_B_transposed 拷贝至 HBM
    CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), num_A * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.get(), h_B_transposed.data(), num_B * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    L2Flusher flusher;

    std::cout << ">>> [1/2] 正在执行 CPU Ground Truth 校验..." << std::endl;
    // 3. CPU Ground Truth 保持传入未经转置的 h_B，确保计算逻辑不变
    cpu_fp8_ground_truth(h_A, h_B, h_C_cpu, scale_a, scale_b, M, N, K);

    launch_fused_fp8_tensor_core_gemm(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C.get(), num_C * sizeof(float), cudaMemcpyDeviceToHost));





    bool pass = true;
    float max_diff = 0.0f;

    // 1. 针对 FP8 设定两个清晰的容忍度门槛：
    // atol: 绝对容差，FP8 乘完 Scale 后的数据底线（直接设为总 Scale 的 1~2 倍）
    // rtol: 相对容差，FP8 尾数只有 3bit，允许 10% 的相对精度偏差
    const float atol = scale_a * scale_b; 
    const float rtol = 0.10f;             

    for (size_t i = 0; i < num_C; ++i) {
        float diff = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > max_diff) max_diff = diff;
        
        float ref = std::abs(h_C_cpu[i]);

        // 2. 只要偏差超过了“绝对底线 + 相对允许偏差”，就断定算错
        if (diff > (atol + rtol * ref)) {
            pass = false;
            std::cout << "❌ Mismatch at index " << i 
                      << " | GPU: " << h_C_gpu[i] 
                      << " vs CPU: " << h_C_cpu[i] << std::endl;
            break;
        }
    }

    if (!pass) {
        std::cout << "❌ [FAIL] 未通过 CPU 校验！Max Diff: " << max_diff << std::endl;
        exit(1);
    }
    std::cout << "✅ [PASS] 成功通过绝对真值比对！Max Diff: " << max_diff << std::endl;

    std::cout << ">>> [2/2] 正在启动 Cold L2-Cache Benchmark..." << std::endl;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int bench_iters = 20;
    float total_ms = 0.0f;

    for (int i = 0; i < bench_iters; ++i) {
        // 1. 在打点前先刷写 L2 Cache（此时 start 还没开始记录）
        flusher.flush(stream);
        
        // 2. 仅在 GEMM 算子发射前后记录事件
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch_fused_fp8_tensor_core_gemm(d_A.get(), d_B.get(), d_C.get(), scale_a, scale_b, M, N, K, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        
        // 3. 阻塞等待当前这第 i 次迭代完成，并计算纯粹的算子耗时
        CUDA_CHECK(cudaEventSynchronize(stop));
        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start, stop));
        
        total_ms += iter_ms; // 累加纯 GEMM 的耗时
    }

    // 4. 计算纯粹 GEMM 的平均耗时
    float avg_ms = total_ms / bench_iters;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    std::cout << "Average Latency (Pure GEMM Cold Cache) : " << avg_ms << " ms" << std::endl;
    std::cout << "Performance                       : " << tflops << " TFLOPS" << std::endl;

    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

int main() {
    run_test_and_benchmark(512, 512, 512);
    run_test_and_benchmark(500, 500, 500);
    return 0;
}