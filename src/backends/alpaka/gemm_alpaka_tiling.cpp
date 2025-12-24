#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <algorithm>
#include <alpaka/alpaka.hpp>

/**
 * @file gemm_alpaka_tiling.cpp
 * @brief Tiled Alpaka implementation for GEMM using Shared Memory.
 *
 * This implementation introduces the concept of **Tiling** (Blocking) to
 * optimize memory access. Instead of reading from Global Memory for every
 * multiplication (like the Naive version), threads cooperatively load a square
 * block (Tile) of data into fast On-Chip **Shared Memory** and perform
 * computations there.
 *
 * **Performance Characteristics:**
 * - **Shared Memory Limited:** Performance depends on how fast we can fill and
 * use the shared memory.
 * - **Reduced Global Bandwidth:** Accesses global memory (M/TILE_SIZE +
 * N/TILE_SIZE) times less often.
 * - **Dynamic Tuning:** Automatically selects between 16x16 or 32x32 tiles
 * based on hardware capabilities.
 */
namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;

/**
 * @brief Functor representing the Tiled GEMM Kernel.
 *
 * Implements the tiled matrix multiplication algorithm.
 *
 * @details
 * **Algorithm:**
 * 1. Threads are grouped into blocks of size `TILE_SIZE x TILE_SIZE`.
 * 2. Each block iterates over the K dimension in steps of `TILE_SIZE`.
 * 3. **Load Phase:** Threads cooperatively load a sub-matrix of A and B into
 * Shared Memory (`As` and `Bs`).
 * 4. **Sync:** `alpaka::syncBlockThreads` ensures all data is loaded.
 * 5. **Compute:** Threads compute the partial dot product using data from
 * Shared Memory.
 * 6. **Sync:** Barrier before loading the next tile to prevent overwriting data
 * still in use.
 *
 * @tparam TILE_SIZE The width/height of the square tile (e.g., 32 or 16).
 * Determined at compile time for loop unrolling and shared memory allocation
 * optimization.
 */
template <int TILE_SIZE> struct GemmTiledKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K) const {

    auto const globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto const localIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
    int localRow = localIdx[0];
    int localCol = localIdx[1];

    float (&As)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);
    float (&Bs)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);

    float accVal = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
      int tiledK = t * TILE_SIZE;

      if (globalIdx[0] < M && (tiledK + localCol) < K) {
        As[localRow][localCol] = A[globalIdx[0] * K + (tiledK + localCol)];
      } else {
        As[localRow][localCol] = 0.0f;
      }

      if ((tiledK + localRow) < K && globalIdx[1] < N) {
        Bs[localRow][localCol] = B[(tiledK + localRow) * N + globalIdx[1]];
      } else {
        Bs[localRow][localCol] = 0.0f;
      }

      alpaka::syncBlockThreads(acc);

      for (int k = 0; k < TILE_SIZE; ++k) {
        accVal += As[localRow][k] * Bs[k][localCol];
      }

      alpaka::syncBlockThreads(acc);
    }

    if (globalIdx[0] < M && globalIdx[1] < N) {
      C[globalIdx[0] * N + globalIdx[1]] = accVal;
    }
  }
};

/**
 * @brief Helper function to launch the Tiled Kernel with a specific template
 * parameter.
 *
 * Calculates the grid dimensions based on the compile-time `TILE_SIZE`.
 *
 * @tparam TILE_SIZE The tile size to use (must match the kernel's template).
 * @tparam TQueue The Alpaka queue type.
 * @param queue The queue where the kernel will be enqueued.
 * @param A Pointer to matrix A (device memory).
 * @param B Pointer to matrix B (device memory).
 * @param C Pointer to matrix C (device memory).
 * @param shape Structure containing dimensions M, N, K.
 */
template <int TILE_SIZE, typename TQueue>
void exec_tiled_template(TQueue &queue, float const *A, float const *B,
                         float *C, GemmShape shape) {
  Idx blocksY = (shape.m + TILE_SIZE - 1) / TILE_SIZE;
  Idx blocksX = (shape.n + TILE_SIZE - 1) / TILE_SIZE;

  auto workDiv = alpaka::WorkDivMembers<Dim2, Idx>{
      alpaka::Vec<Dim2, Idx>{blocksY, blocksX},
      alpaka::Vec<Dim2, Idx>{(Idx)TILE_SIZE, (Idx)TILE_SIZE},
      alpaka::Vec<Dim2, Idx>{1u, 1u}};

  GemmTiledKernel<TILE_SIZE> kernel;
  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, shape.m, shape.n, shape.k);
  alpaka::wait(queue);
}
/**
 * @brief Determines the optimal tile size based on Device Properties.
 *
 * Queries the GPU hardware to decide the best block size.
 *
 * @details
 * - **Target:** 32x32 tiles (1024 threads). This maximizes occupancy and data
 * reuse on modern GPUs.
 * - **Fallback:** 16x16 tiles (256 threads). Used for older GPUs or
 * architectures with limited registers/shared memory per block.
 *
 * @tparam TAcc The Alpaka Accelerator type.
 * @param platform The Alpaka platform instance.
 * @param devIdx The device index.
 * @return The optimal tile width (32 or 16).
 */
template <typename TAcc>
int get_optimal_tile_width_alpaka(alpaka::PlatformCudaRt const &platform,
                                  int devIdx) {
  auto dev = alpaka::getDevByIdx(platform, devIdx);
  auto props = alpaka::getAccDevProps<TAcc>(dev);

  if (props.m_blockThreadCountMax >= 1024) {
    return 32;
  }

  return 16;
}

/**
 * @brief Main Entry Point for the Tiled Alpaka Backend.
 *
 * Performs runtime dispatching to select the correct template instantiation
 * based on the hardware query.
 *
 * @details
 * Uses `get_optimal_tile_width_alpaka` to choose between
 * `exec_tiled_template<32>` and `exec_tiled_template<16>`. This allows the code
 * to be compiled once but run efficiently on different GPUs.
 *
 * @tparam TQueue The Alpaka queue type.
 * @param queue The queue provided by the application.
 * @param A Matrix A (Input).
 * @param B Matrix B (Input).
 * @param C Matrix C (Output).
 * @param shape Structure containing dimensions M, N, K.
 */
template <typename TQueue>
void gemm_alpaka_tiled(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape) {

  int optimal_tile =
      get_optimal_tile_width_alpaka<Acc>(alpaka::PlatformCudaRt{}, 0);

  if (optimal_tile >= 32) {
    exec_tiled_template<32>(queue, A, B, C, shape);
  } else {
    exec_tiled_template<16>(queue, A, B, C, shape);
  }
}

using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;
template void gemm_alpaka_tiled<QueueType>(QueueType &queue, float const *A,
                                           float const *B, float *C,
                                           GemmShape shape);

} // namespace gemm
