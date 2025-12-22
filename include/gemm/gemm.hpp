
#pragma once
#include <cstddef>

namespace gemm {

struct GemmShape {
  int m, n, k;
};

using GemmFn = void (*)(const float *A, const float *B, float *C, GemmShape s);

// Solver CUDA
void gemm_cuda_naive(const float *A, const float *B, float *C, GemmShape s);
// solver alpaka

template <typename TQueue>
void gemm_alpaka_naive(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape);
// void gemm_cuda_tiled(const float* A, const float* B, float* C, GemmShape s);
// void gemm_alpaka_naive(const float* A, const float* B, float* C, GemmShape
// s);

} // namespace gemm
