#pragma once
#include <cstddef>

#ifdef GEMM_ENABLE_ALPAKA
#include <alpaka/alpaka.hpp>
#endif

namespace gemm {

struct GemmShape {
  int m, n, k;
};

using GemmFn = void (*)(const float *A, const float *B, float *C, GemmShape s);

// Solver CUDA
void gemm_cuda_naive(const float *A, const float *B, float *C, GemmShape s);
void gemm_cuda_full_options(const float *A, const float *B, float *C, GemmShape s);
void gemm_cuda_mpi(const float *A, const float *B, float *C, GemmShape s);

// Solver Alpaka
#ifdef GEMM_ENABLE_ALPAKA
template <typename TQueue>
void gemm_alpaka_naive(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape);

template <typename TQueue>
void gemm_alpaka_full_options(TQueue &queue, float const *A, float const *B,
                              float *C, GemmShape shape);

template <typename TQueue>
void gemm_alpaka_tiled(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape);
#endif

} // namespace gemm
