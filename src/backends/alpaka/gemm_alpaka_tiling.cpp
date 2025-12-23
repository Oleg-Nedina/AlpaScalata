#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>

namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;

// Definiamo la dimensione del TILE
// Deve corrispondere alla dimensione del blocco di thread per questo algoritmo
// "semplice"
constexpr int TILE_SIZE = 16;

struct GemmTiledKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K) const {

    // 1. Indici Globali (della matrice finale C)
    auto const globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    int globalRow = globalIdx[0];
    int globalCol = globalIdx[1];

    // 2. Indici Locali (all'interno del blocco/tile)
    auto const localIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
    int localRow = localIdx[0]; // threadIdx.y
    int localCol = localIdx[1]; // threadIdx.x

    // 3. Dichiarazione Shared Memory (Static)
    // In CUDA: __shared__ float As[TILE_SIZE][TILE_SIZE];
    float (&As)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);
    float (&Bs)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);

    float accVal = 0.0f;

    // 4. Loop sui Tiles
    // Scorriamo le sottomatrici di A e B lungo la dimensione K a passi di
    // TILE_SIZE
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {

      // --- FASE DI CARICAMENTO (Cooperative Loading) ---

      // Indice colonna della tile corrente di A / riga della tile corrente di B
      int tiledK = t * TILE_SIZE;

      // Carichiamo A nel buffer condiviso As
      // Controllo bounds: siamo dentro la matrice A?
      if (globalRow < M && (tiledK + localCol) < K) {
        As[localRow][localCol] = A[globalRow * K + (tiledK + localCol)];
      } else {
        As[localRow][localCol] = 0.0f;
      }

      // Carichiamo B nel buffer condiviso Bs
      // Controllo bounds: siamo dentro la matrice B?
      if ((tiledK + localRow) < K && globalCol < N) {
        Bs[localRow][localCol] = B[(tiledK + localRow) * N + globalCol];
      } else {
        Bs[localRow][localCol] = 0.0f;
      }

      // --- SINCRONIZZAZIONE (Wait for Load) ---
      // Assicuriamo che tutti i thread abbiano caricato i dati prima di
      // calcolare
      alpaka::syncBlockThreads(acc);

      // --- FASE DI CALCOLO (Compute) ---
      for (int k = 0; k < TILE_SIZE; ++k) {
        accVal += As[localRow][k] * Bs[k][localCol];
      }

      // --- SINCRONIZZAZIONE (Wait for Compute) ---
      // Assicuriamo che tutti abbiano finito di usare As/Bs prima di
      // sovrascriverle nella prossima iterazione
      alpaka::syncBlockThreads(acc);
    }

    // 5. Scrittura Risultato
    if (globalRow < M && globalCol < N) {
      C[globalRow * N + globalCol] = accVal;
    }
  }
};

template <typename TQueue>
void gemm_alpaka_tiled(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape) {

  // Dimensione del blocco fissa a 16x16 per matchare il TILE_SIZE
  constexpr Idx TX = TILE_SIZE;
  constexpr Idx TY = TILE_SIZE;

  Idx blocksY = (Idx)((shape.m + TY - 1) / TY);
  Idx blocksX = (Idx)((shape.n + TX - 1) / TX);

  auto const gridThreadExtent = alpaka::Vec<Dim2, Idx>{blocksY, blocksX};
  auto const blockThreadExtent = alpaka::Vec<Dim2, Idx>{TY, TX};
  auto const elemExtent = alpaka::Vec<Dim2, Idx>{1u, 1u};

  alpaka::WorkDivMembers<Dim2, Idx> workDiv(gridThreadExtent, blockThreadExtent,
                                            elemExtent);
  GemmTiledKernel kernel;

  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, shape.m, shape.n, shape.k);
  alpaka::wait(queue);
}

// Istanziazione Esplicita
using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;
template void gemm_alpaka_tiled<QueueType>(QueueType &queue, float const *A,
                                           float const *B, float *C,
                                           GemmShape shape);

} // namespace gemm
