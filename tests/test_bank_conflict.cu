#include "tiled_gemm_bank.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <memory>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

// =========================================================================
// 🌟 1. C++ RAII CUDA 资源管理 Wrapper
// =========================================================================
template <typename T>
struct CudaMemoryDeleter {
    void operator()(T* ptr) const {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
    }
};

template <typename T>
using UniqueCudaPtr = std::unique_ptr<T, CudaMemoryDeleter<T>>;

template <typename T>
UniqueCudaPtr<T> make_cuda_unique(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return UniqueCudaPtr<T>(ptr);
}

struct CudaStreamDeleter {
    void operator()(cudaStream_t stream) const {
        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    }
};
using UniqueCudaStream = std::unique_ptr<std::remove_pointer<cudaStream_t>::type, CudaStreamDeleter>;

UniqueCudaStream make_cuda_stream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return UniqueCudaStream(stream);
}

struct CudaEventDeleter {
    void operator()(cudaEvent_t event) const {
        if (event) CUDA_CHECK(cudaEventDestroy(event));
    }
};
using UniqueCudaEvent = std::unique_ptr<std::remove_pointer<cudaEvent_t>::type, CudaEventDeleter>;

UniqueCudaEvent make_cuda_event() {
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreate(&event));
    return UniqueCudaEvent(event);
}

// =========================================================================
// 🌟 2. 物理级 L2 Cache 刷新机制
// =========================================================================
class L2CacheFlusher {
private:
    void* d_flush_buffer_ = nullptr;
    size_t l2_size_bytes_ = 0;

public:
    L2CacheFlusher() {
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        
        l2_size_bytes_ = prop.l2CacheSize * 2;
        if (l2_size_bytes_ == 0) {
            l2_size_bytes_ = 64 * 1024 * 1024; // 默认 64MB
        }
        CUDA_CHECK(cudaMalloc(&d_flush_buffer_, l2_size_bytes_));
    }

    ~L2CacheFlusher() {
        if (d_flush_buffer_) cudaFree(d_flush_buffer_);
    }

    void flush(cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(d_flush_buffer_, 0, l2_size_bytes_, stream));
    }
};

// =========================================================================
// 🌟 3. CPU 绝对标量 Ground Truth
// =========================================================================
void cpu_gemm_fp16_ground_truth(const std::vector<__half>& A, const std::vector<__half>& B, 
                                std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// 朴素 Baseline Kernel
__global__ void naive_bank_gemm_fp16_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C, 
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

void launch_naive_bank_gemm(const __half* d_A, const __half* d_B, __half* d_C, 
                            int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_bank_gemm_fp16_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

// =========================================================================
// 4. 重构后的主程序入口
// =========================================================================
int main(int argc, char** argv) {
    const int warmup_iters = 5;
    const int bench_iters = 20;

    int M = 512;
    int N = 512;
    int K = 512;

    std::cout << "=================================================" << std::endl;
    std::cout << ">>> Robust & Industrial-grade Swizzle Test Harness" << std::endl;
    std::cout << ">>> Matrix Shape (M x N x K): " << M << " x " << N << " x " << K << std::endl;
    std::cout << "=================================================" << std::endl;

    size_t num_A = M * K;
    size_t num_B = K * N;
    size_t num_C = M * N;

    std::vector<__half> h_A(num_A);
    std::vector<__half> h_B(num_B);
    std::vector<__half> h_C_gpu_swizzle(num_C);
    std::vector<float>  h_C_cpu_truth(num_C);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < num_A; ++i) h_A[i] = __float2half(dist(rng));
    for (size_t i = 0; i < num_B; ++i) h_B[i] = __float2half(dist(rng));

    auto d_A = make_cuda_unique<__half>(num_A);
    auto d_B = make_cuda_unique<__half>(num_B);
    auto d_C = make_cuda_unique<__half>(num_C);

    CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), num_A * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.get(), h_B.data(), num_B * sizeof(__half), cudaMemcpyHostToDevice));

    auto stream = make_cuda_stream();
    L2CacheFlusher l2_flusher;

    // ------------------------------------------------------------------------
    // Step 1: 绝对真值 Correctness 校验
    // ------------------------------------------------------------------------
    std::cout << ">>> [1/3] 计算 CPU FP32 绝对 Ground Truth 校验基准..." << std::endl;
    cpu_gemm_fp16_ground_truth(h_A, h_B, h_C_cpu_truth, M, N, K);

    std::cout << ">>> 运行 Swizzle Bank-Free GEMM 算子..." << std::endl;
    smem_opt::launch_swizzle_bank_free_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K, stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    CUDA_CHECK(cudaMemcpy(h_C_gpu_swizzle.data(), d_C.get(), num_C * sizeof(__half), cudaMemcpyDeviceToHost));

    bool pass = true;
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int error_count = 0;

    for (size_t i = 0; i < num_C; ++i) {
        float gpu_val = __half2float(h_C_gpu_swizzle[i]);
        float cpu_val = h_C_cpu_truth[i];
        float abs_diff = std::abs(gpu_val - cpu_val);
        float rel_diff = abs_diff / (std::abs(cpu_val) + 1e-5f);

        if (abs_diff > max_diff) max_diff = abs_diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;

        // 🌟 动态双重门限，精准抵御 FP16 浮点加法顺序误差
        if (abs_diff > 0.25f && rel_diff > 0.02f) { 
            if (error_count < 5) {
                std::cerr << "❌ [Absolute Truth Mismatch] Index " << i 
                          << ": GPU Swizzle=" << gpu_val 
                          << ", CPU Ground Truth=" << cpu_val 
                          << " (Abs Diff: " << abs_diff << ", Rel Diff: " << rel_diff * 100 << "%)" << std::endl;
            }
            pass = false;
            error_count++;
        }
    }

    if (!pass) {
        std::cout << "❌ [FAIL] 算子逻辑错误！未能通过 CPU 绝对真值校验。" << std::endl;
        return 1; 
    }
    std::cout << "✅ [PASS] 成功通过 CPU 绝对真值比对！Max Abs Diff: " << max_diff << ", Max Rel Diff: " << max_rel_diff * 100 << "%" << std::endl;

    // ------------------------------------------------------------------------
    // Step 2: 精确排除了 L2 Flush 干扰的 Cold L2-Cache Benchmarking
    // ------------------------------------------------------------------------
    std::cout << ">>> [2/3] 正在启动真实 Cold-L2-Cache Performance Benchmark..." << std::endl;

    // 1. Benchmark Naive GEMM
    for (int i = 0; i < warmup_iters; ++i) {
        l2_flusher.flush(stream.get());
        launch_naive_bank_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K, stream.get());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    float total_naive_ms = 0.0f;
    for (int i = 0; i < bench_iters; ++i) {
        l2_flusher.flush(stream.get()); 
        
        auto start = make_cuda_event();
        auto stop  = make_cuda_event();
        
        CUDA_CHECK(cudaEventRecord(start.get(), stream.get()));
        launch_naive_bank_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K, stream.get());
        CUDA_CHECK(cudaEventRecord(stop.get(), stream.get()));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start.get(), stop.get()));
        total_naive_ms += iter_ms;
    }
    float naive_ms = total_naive_ms / bench_iters;

    // 2. Benchmark Swizzle Bank-Free GEMM
    for (int i = 0; i < warmup_iters; ++i) {
        l2_flusher.flush(stream.get());
        smem_opt::launch_swizzle_bank_free_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K, stream.get());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    float total_swizzle_ms = 0.0f;
    CUDA_CHECK(cudaProfilerStart());
    for (int i = 0; i < bench_iters; ++i) {
        l2_flusher.flush(stream.get()); 
        
        auto start = make_cuda_event();
        auto stop  = make_cuda_event();

        CUDA_CHECK(cudaEventRecord(start.get(), stream.get()));
        smem_opt::launch_swizzle_bank_free_gemm(d_A.get(), d_B.get(), d_C.get(), M, N, K, stream.get());
        CUDA_CHECK(cudaEventRecord(stop.get(), stream.get()));
        CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start.get(), stop.get()));
        total_swizzle_ms += iter_ms;
    }
    CUDA_CHECK(cudaProfilerStop());
    float swizzle_ms = total_swizzle_ms / bench_iters;

    // ------------------------------------------------------------------------
    // Step 3: 指标计算与格式化输出
    // ------------------------------------------------------------------------
    double total_flops = 2.0 * static_cast<double>(M) * N * K;
    double naive_tflops = (total_flops / (naive_ms / 1000.0)) / 1e12;
    double swizzle_tflops = (total_flops / (swizzle_ms / 1000.0)) / 1e12;
    float speedup = naive_ms / swizzle_ms;

    std::cout << "\n==================== BENCHMARK RESULTS (COLD L2 CACHE) ====================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Naive (Bank Conflict) Time : " << naive_ms << " ms (" << naive_tflops << " TFLOPS)" << std::endl;
    std::cout << "Swizzle (Bank-Free) Time   : " << swizzle_ms << " ms (" << swizzle_tflops << " TFLOPS)" << std::endl;
    std::cout << "Real DRAM Speedup          : " << speedup << " x" << std::endl;
    std::cout << "==========================================================================" << std::endl;

    return 0;
}