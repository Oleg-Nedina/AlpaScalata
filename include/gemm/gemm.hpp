#pragma once

#include <alpaka/alpaka.hpp>

namespace gemm {

struct GemmShape {
  int m, n, k;
};

using GemmFn = void (*)(const float *A, const float *B, float *C, GemmShape s);

// --------------------------------------------------------------------------
// Solver CUDA (Implementato in gemm_cuda_naive.cu)
// --------------------------------------------------------------------------
void gemm_cuda_naive(const float *A, const float *B, float *C, GemmShape s);

// --------------------------------------------------------------------------
// Solver Alpaka (Implementato in gemm_alpaka_naive.cpp)
// --------------------------------------------------------------------------
template <typename TQueue>
void gemm_alpaka_naive(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape);

} // namespace gemm
