#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>

namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;

// 1. IL KERNEL (Templatizzato su TILE_SIZE)
template <int TILE_SIZE> struct GemmTiledKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K) const {

    auto const globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto const localIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
    int localRow = localIdx[0];
    int localCol = localIdx[1];

    // Shared Memory STATICA
    float (&As)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);
    float (&Bs)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);

    float accVal = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
      int tiledK = t * TILE_SIZE;

      // Caricamento A
      if (globalIdx[0] < M && (tiledK + localCol) < K) {
        As[localRow][localCol] = A[globalIdx[0] * K + (tiledK + localCol)];
      } else {
        As[localRow][localCol] = 0.0f;
      }

      // Caricamento B
      if ((tiledK + localRow) < K && globalIdx[1] < N) {
        Bs[localRow][localCol] = B[(tiledK + localRow) * N + globalIdx[1]];
      } else {
        Bs[localRow][localCol] = 0.0f;
      }

      alpaka::syncBlockThreads(acc);

      // Calcolo
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

// 2. HELPER PER LANCIARE IL KERNEL (Definito PRIMA dell'uso)
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

// 3. CALCOLO OTTIMALE (Host)
template <typename TAcc>
int get_optimal_tile_width_alpaka(alpaka::PlatformCudaRt const &platform,
                                  int devIdx) {
  auto dev = alpaka::getDevByIdx(platform, devIdx);
  auto props = alpaka::getAccDevProps<TAcc>(dev);

  size_t shared_mem_per_block_target = props.m_sharedMemPerMultiprocessor / 2;
  int max_elements = (int)(shared_mem_per_block_target / (2 * sizeof(float)));
  int side_shared = (int)std::sqrt((double)max_elements);
  int side_threads = (int)std::sqrt((double)props.m_blockThreadCountMax);

  int tile_width = std::min(side_shared, side_threads);

  if (tile_width >= 32)
    return 32;
  return 16;
}

// 4. MAIN ENTRY POINT (Rinominato in gemm_alpaka_tiled per compatibilità)
template <typename TQueue>
void gemm_alpaka_tiled(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape) {

  // Scegliamo dinamicamente la dimensione migliore
  int optimal_tile =
      get_optimal_tile_width_alpaka<Acc>(alpaka::PlatformCudaRt{}, 0);

  // Dispatcher: sceglie quale versione compilata lanciare
  if (optimal_tile >= 32) {
    exec_tiled_template<32>(queue, A, B, C, shape);
  } else {
    exec_tiled_template<16>(queue, A, B, C, shape);
  }
}

// 5. ISTANZIAZIONE ESPLICITA
using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;
template void gemm_alpaka_tiled<QueueType>(QueueType &queue, float const *A,
                                           float const *B, float *C,
                                           GemmShape shape);

} // namespace gemm
