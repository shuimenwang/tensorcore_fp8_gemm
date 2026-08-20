cmake_minimum_required(VERSION 3.22)
project(TensorCore_FP8_GEMM LANGUAGES CXX CUDA)

# 1. C++ 与 CUDA 标准设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. 彻底禁用默认的架构推导，防止产生 compute_75
set(CMAKE_CUDA_ARCHITECTURES OFF)

# 3. 🌟【关键修复】：使用 -gencode 强制只编译 compute_80 / sm_80，彻底屏蔽 compute_75
# 针对 RTX 50 系列 (Blackwell 架构) 指定 sm_100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_120,code=[sm_120,compute_120] -O3 --use_fast_math -lineinfo -Xptxas -v --expt-relaxed-constexpr")

include_directories(include)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

# 编译底层静态库
add_library(gemm_kernels STATIC
    src/async_pipeline.cu
    src/tiled_gemm_bank.cu
    src/fused_fp8_gemm.cu
)
target_link_libraries(gemm_kernels PRIVATE CUDA::cudart)

# 编译板块 A 的测试程序
add_executable(test_async_copy tests/test_async_copy.cu)
target_link_libraries(test_async_copy 
    PRIVATE 
        gemm_kernels 
        CUDA::cudart
)

# 编译 Benchmark 压测程序
#add_executable(bench_gemm benchmarks/bench_gemm_performance.cu)
#target_link_libraries(bench_gemm 
#    PRIVATE 
#        gemm_kernels 
#        CUDA::cublas 
#        CUDA::cudart
#)

add_executable(test_bank_conflict tests/test_bank_conflict.cu)
target_link_libraries(test_bank_conflict 
    PRIVATE 
        gemm_kernels 
        CUDA::cudart
)

add_executable(test_gemm_correctness tests/test_gemm_correctness.cu)
target_link_libraries(test_gemm_correctness 
    PRIVATE 
        gemm_kernels 
        CUDA::cudart
)




