#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "async_pipeline.cuh"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at line " << __LINE__ << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t stat = call;                                            \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS Error code: " << stat                          \
                      << " at line " << __LINE__ << std::endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// -----------------------------------------------------------------------------
// 0. 对照组：Naive 同步 Block 搬运 GEMM Kernel (无 cp.async，显式同步)
// -----------------------------------------------------------------------------
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

__global__ void naive_sync_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    __shared__ float smem_A[TILE_M][TILE_K];
    __shared__ float smem_B[TILE_K][TILE_N];

    int tx = threadIdx.x; // 0..31
    int ty = threadIdx.y; // 0..31

    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_N + tx;

    float accum = 0.0f;
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // 同步显式读取：Global Memory -> Shared Memory
        int a_col = t * TILE_K + tx;
        int b_row = t * TILE_K + ty;

        smem_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        smem_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads(); // 必须同步以保证数据加载完成

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            accum += smem_A[ty][k] * smem_B[k][tx];
        }

        __syncthreads(); // 必须同步以防止覆盖下轮数据
    }

    if (row < M && col < N) {
        C[row * N + col] = accum;
    }
}

void launch_naive_sync_gemm(const float* d_A, const float* d_B, float* d_C,
                            int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    naive_sync_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

// -----------------------------------------------------------------------------
// 辅助函数
// -----------------------------------------------------------------------------
void init_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

void flush_l2_cache(void* dummy_ptr, size_t size, cudaStream_t stream) {
    CHECK_CUDA(cudaMemsetAsync(dummy_ptr, 0, size, stream));
}

bool verify_result(const float* host_C, const float* host_ref, int size, float tol = 1e-3f) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(host_C[i] - host_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (std::isnan(host_C[i]) || std::isinf(host_C[i])) {
            std::cout << "[Validation Failed] NaN or Inf detected at index " << i << "\n";
            return false;
        }
    }
    std::cout << "-> Max Absolute Error: " << max_diff;
    return max_diff < tol;
}

// -----------------------------------------------------------------------------
// 基准测试核心函数（三方对比）
// -----------------------------------------------------------------------------
void run_benchmark(int M, int N, int K, int warmup_iters, int bench_iters) {
    std::cout << "\n======================================================\n";
    std::cout << "Testing Matrix Size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "======================================================\n";

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_naive(M * N);
    std::vector<float> h_C_custom(M * N);
    std::vector<float> h_C_cublas(M * N);

    init_matrix(h_A.data(), M * K);
    init_matrix(h_B.data(), K * N);

    float *d_A, *d_B, *d_C_naive, *d_C_custom, *d_C_cublas;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C_naive, size_C));
    CHECK_CUDA(cudaMalloc(&d_C_custom, size_C));
    CHECK_CUDA(cudaMalloc(&d_C_cublas, size_C));

    size_t l2_flush_size = 64 * 1024 * 1024;
    void* d_dummy;
    CHECK_CUDA(cudaMalloc(&d_dummy, l2_flush_size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // -------------------------------------------------------------------------
    // 1. 正确性校验
    // -------------------------------------------------------------------------
    launch_naive_sync_gemm(d_A, d_B, d_C_naive, M, N, K, stream);
    launch_async_pipeline_gemm_stage(d_A, d_B, d_C_custom, M, N, K, stream);
    
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaMemcpy(h_C_naive.data(), d_C_naive, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_custom.data(), d_C_custom, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C_cublas, size_C, cudaMemcpyDeviceToHost));

    std::cout << "[Naive  Vs cuBLAS] ";
    verify_result(h_C_naive.data(), h_C_cublas.data(), M * N);
    std::cout << "\n[Async  Vs cuBLAS] ";
    verify_result(h_C_custom.data(), h_C_cublas.data(), M * N);
    std::cout << "\n";

    double flops = 2.0 * static_cast<double>(M) * N * K;

    // -------------------------------------------------------------------------
    // 2. 测试 Naive 同步 GEMM 性能
    // -------------------------------------------------------------------------
    for (int i = 0; i < warmup_iters; ++i) {
        launch_naive_sync_gemm(d_A, d_B, d_C_naive, M, N, K, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float total_ms_naive = 0.0f;
    for (int i = 0; i < bench_iters; ++i) {
        flush_l2_cache(d_dummy, l2_flush_size, stream);
        
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_naive_sync_gemm(d_A, d_B, d_C_naive, M, N, K, stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms_naive += iter_ms;
    }
    float avg_ms_naive = total_ms_naive / bench_iters;
    double tflops_naive = (flops / (avg_ms_naive / 1000.0)) / 1e12;

    // -------------------------------------------------------------------------
    // 3. 测试 Custom cp.async GEMM 性能
    // -------------------------------------------------------------------------
    for (int i = 0; i < warmup_iters; ++i) {
        launch_async_pipeline_gemm_stage(d_A, d_B, d_C_custom, M, N, K, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float total_ms_custom = 0.0f;
    for (int i = 0; i < bench_iters; ++i) {
        flush_l2_cache(d_dummy, l2_flush_size, stream);
        
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch_async_pipeline_gemm_stage(d_A, d_B, d_C_custom, M, N, K, stream);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms_custom += iter_ms;
    }
    float avg_ms_custom = total_ms_custom / bench_iters;
    double tflops_custom = (flops / (avg_ms_custom / 1000.0)) / 1e12;

    // -------------------------------------------------------------------------
    // 4. 测试 cuBLAS 性能
    // -------------------------------------------------------------------------
    for (int i = 0; i < warmup_iters; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float total_ms_cublas = 0.0f;
    for (int i = 0; i < bench_iters; ++i) {
        flush_l2_cache(d_dummy, l2_flush_size, stream);
        
        CHECK_CUDA(cudaEventRecord(start, stream));
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms_cublas += iter_ms;
    }
    float avg_ms_cublas = total_ms_cublas / bench_iters;
    double tflops_cublas = (flops / (avg_ms_cublas / 1000.0)) / 1e12;

    // -------------------------------------------------------------------------
    // 5. 打印对比结果
    // -------------------------------------------------------------------------
    std::cout << "--- Performance Breakdown ---\n";
    std::cout << "1. Naive Sync GEMM   : " << avg_ms_naive  << " ms | " << tflops_naive  << " TFLOPS\n";
    std::cout << "2. cp.async Pipeline : " << avg_ms_custom << " ms | " << tflops_custom << " TFLOPS\n";
    std::cout << "3. cuBLAS (Reference): " << avg_ms_cublas << " ms | " << tflops_cublas << " TFLOPS\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "Speedup (cp.async vs Naive) : " << (avg_ms_naive / avg_ms_custom) << "x\n";
    std::cout << "Performance Ratio vs cuBLAS : " << (tflops_custom / tflops_cublas) * 100.0f << " %\n";

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_custom);
    cudaFree(d_C_cublas);
    cudaFree(d_dummy);
}

int main() {
    int device_id = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "Device Name: " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ")\n";

    const int WARMUP = 10;
    const int BENCHMARK_ITERS = 100;

    std::vector<int> test_sizes = {512, 1024, 2048, 4096};

    for (int size : test_sizes) {
        run_benchmark(size, size, size, WARMUP, BENCHMARK_ITERS);
    }

    return 0;
}