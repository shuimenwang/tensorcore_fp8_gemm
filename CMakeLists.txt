cmake_minimum_required(VERSION 3.22)
project(TensorCore_FP8_GEMM LANGUAGES CXX CUDA)

# 强制开启 C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 针对当前 GPU 架构自动匹配最高的 NVCC 编译算子架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# CUDA 编译参数：最高级别优化 + 保留源码行信息（用于 Nsight 抓取）+ 开启快速浮点计算
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo -Xptxas -v --expt-relaxed-constexpr")

# 包含头文件路径
include_directories(include)

# 寻找 GoogleTest 库（用于 correctness 测试）
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# 编译主动态链接库 / 静态库
add_library(gemm_kernels STATIC
    src/async_pipeline.cu
    src/tiled_gemm_bank.cu
    src/fused_fp8_gemm.cu
)

# 编译算子正确性测试程序
add_executable(test_gemm_correctness tests/test_gemm_correctness.cu)
target_link_libraries(test_gemm_correctness gemm_kernels ${GTEST_LIBRARIES} pthread)

# 编译性能 Benchmark 压测程序
add_executable(bench_gemm benchmarks/bench_gemm_performance.cu)
target_link_libraries(bench_gemm gemm_kernels CUDA::cublas)