#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA 运行时 API 错误检查宏
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA Error at %s:%d - %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// 检查 Kernel 异步启动错误
#define CUDA_CHECK_LAST_ERROR()                                                \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n",           \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#endif // COMMON_H