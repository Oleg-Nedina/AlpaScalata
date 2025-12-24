#define GEMM_ENABLE_ALPAKA

#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>

/**
 * @file gemm_alpaka_naive.cpp
 * @brief Baseline / Naive Alpaka implementation for GEMM.
 *
 * This file contains the simplest possible implementation of Matrix
 * Multiplication using Alpaka. It maps one thread to one output pixel
 * (C[row][col]) and performs the dot product by reading directly from Global
 * Memory.
 *
 * **Performance Characteristics:**
 * - **Memory Bound:** Heavily limited by Global Memory bandwidth.
 * - **No Caching:** Does not use Shared Memory or Register Tiling.
 * - **Usage:** Primarily used for correctness verification and establishing a
 * minimum performance baseline.
 */
namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;

using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCudaRt;

/**
 * @brief Functor representing the Naive GEMM Kernel.
 *
 * Implements the standard matrix multiplication algorithm:
 * \f$ C_{row,col} = \sum_{k=0}^{K-1} A_{row,k} \times B_{k,col} \f$
 *
 * @details
 * - **Grid Mapping:** 1 Thread = 1 Output Element (Pixel) in C.
 * - **Memory Access:** Reads A and B directly from Global Memory for every step
 * of the accumulation loop. This results in high memory latency.
 */
struct GemmNaiveKernel {

  /**
   * @brief The kernel entry point executed on the device.
   *
   * @tparam TAcc The Alpaka Accelerator type (provides thread indexing).
   * @param acc The accelerator context.
   * @param A Input Matrix A (Row-Major).
   * @param B Input Matrix B (Row-Major).
   * @param C Output Matrix C (Row-Major).
   * @param M Number of rows of A and C.
   * @param N Number of columns of B and C.
   * @param K Shared dimension (Cols of A, Rows of B).
   */
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K) const {
    auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    int row = (int)idx[0];
    int col = (int)idx[1];

    if (row < M && col < N) {
      float sum = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        sum += A[row * K + kk] * B[kk * N + col];
      }
      C[row * N + col] = sum;
    }
  }
};

/**
 * @brief Host-side dispatcher for the Naive Alpaka Kernel.
 *
 * Configures the execution grid and launches the `GemmNaiveKernel`.
 *
 * @details
 * - **Block Size:** Fixed at 16x16 (256 threads). This is a safe default
 * supported by almost all GPUs.
 * - **Grid Size:** Calculated to cover the dimensions M and N (rounding up).
 * - **Execution:** Synchronous (Blocking) for simplicity in benchmarking.
 *
 * @tparam TQueue The Alpaka queue type.
 * @param queue The queue where the kernel will be enqueued.
 * @param A Pointer to matrix A (device memory).
 * @param B Pointer to matrix B (device memory).
 * @param C Pointer to matrix C (device memory).
 * @param shape Structure containing dimensions M, N, K.
 */
template <typename TQueue>
void gemm_alpaka_naive(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape) {
  constexpr Idx TX = 16;
  constexpr Idx TY = 16;

  Idx blocksY = (Idx)((shape.m + (int)TY - 1) / (int)TY);
  Idx blocksX = (Idx)((shape.n + (int)TX - 1) / (int)TX);

  auto const gridThreadExtent = alpaka::Vec<Dim2, Idx>{blocksY, blocksX};
  auto const blockThreadExtent = alpaka::Vec<Dim2, Idx>{TY, TX};
  auto const elemExtent = alpaka::Vec<Dim2, Idx>{1u, 1u};

  alpaka::WorkDivMembers<Dim2, Idx> workDiv(gridThreadExtent, blockThreadExtent,
                                            elemExtent);
  GemmNaiveKernel kernel;

  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, shape.m, shape.n, shape.k);
  alpaka::wait(queue);
}

using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;
template void gemm_alpaka_naive<QueueType>(QueueType &queue, float const *A,
                                           float const *B, float *C,
                                           GemmShape shape);

} // namespace gemm
