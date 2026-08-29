cmake_minimum_required(VERSION 3.22)
project(TensorCore_FP8_GEMM LANGUAGES CXX CUDA)

# 1. C++ 与 CUDA 标准设置
# CXX 标准设为 20，CUDA 标准设为 17（解决 CMake 不识别 CUDA20 dialect 的问题）
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. 针对 RTX 50 系列 (Blackwell 架构 sm_120) 的现代化架构配置
set(CMAKE_CUDA_ARCHITECTURES 120-real 120-virtual)

# 3. CUDA 优化与编译选项设置
# --std=c++20 强行让 NVCC 在 C++20 模式下编译 CUDA 源文件
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

# 编译底层静态库
add_library(gemm_kernels STATIC
    src/async_pipeline.cu
    src/tiled_gemm_bank.cu
    src/fused_fp8_gemm.cu
)

# 使用 target_include_directories 自动向依赖项传递 include 路径
target_include_directories(gemm_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gemm_kernels PRIVATE CUDA::cudart)

# 编译板块 A 的测试程序
add_executable(test_async_copy tests/test_async_copy.cu)
target_link_libraries(test_async_copy 
    PRIVATE 
        gemm_kernels 
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
)

# 编译 Bank Conflict 测试程序
add_executable(test_bank_conflict tests/test_bank_conflict.cu)
target_link_libraries(test_bank_conflict 
    PRIVATE 
        gemm_kernels 
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
)

# 编译正确性验证程序
add_executable(test_gemm_correctness tests/test_gemm_correctness.cu)
target_link_libraries(test_gemm_correctness 
    PRIVATE 
        gemm_kernels 
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
)

# 编译 FP8 GEMM 三路对比测试 (Naive / cuBLASLt / Custom) + Benchmark
add_executable(test_fp8_gemm_comparison tests/test_fp8_gemm_comparison.cu)
target_link_libraries(test_fp8_gemm_comparison
    PRIVATE
        gemm_kernels
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
)
