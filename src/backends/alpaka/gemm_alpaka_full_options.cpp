#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cmath>
#include <iostream>
#include <vector>

namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;
using Dim1 = alpaka::DimInt<1>;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;

// =============================================================================
// 0. HELPER: Crea View 2D da puntatore raw (Solo per Batching Path)
// =============================================================================
template <typename TDev, typename TPtr>
auto as_view_2d(TDev const &dev, TPtr *ptr, Idx rows, Idx cols,
                Idx pitch_elems) {
  auto ext = alpaka::Vec<Dim2, Idx>{rows, cols};
  auto strides = alpaka::Vec<Dim2, Idx>{pitch_elems, 1u};
  return alpaka::createView(dev, ptr, ext, strides);
}

// Versione per buffer contigui (Pitch == Cols)
template <typename TDev, typename TPtr>
auto as_view_2d_contiguous(TDev const &dev, TPtr *ptr, Idx rows, Idx cols) {
  return as_view_2d(dev, ptr, rows, cols, cols);
}

// =============================================================================
// 1. KERNEL GEMM (Invariato)
// =============================================================================
template <int TILE_SIZE> struct GemmFullKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K,
                                bool accumulate) const {

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
      Idx idx = globalIdx[0] * N + globalIdx[1];
      if (accumulate) {
        C[idx] += accVal;
      } else {
        C[idx] = accVal;
      }
    }
  }
};

// =============================================================================
// 2. UPLOAD/DOWNLOAD TILES (Per Batching)
// =============================================================================
template <typename TQueue, typename TDevHost, typename TDevAcc>
void upload_tile_alpaka(TQueue &queue, TDevHost const &devHost,
                        TDevAcc const &devAcc, float *d_dst, const float *h_src,
                        int big_N, int r_off, int c_off, int rows, int cols) {

  float const *pSrcStart = h_src + (Idx)r_off * big_N + c_off;
  auto viewHost =
      as_view_2d(devHost, pSrcStart, (Idx)rows, (Idx)cols, (Idx)big_N);
  auto viewDev = as_view_2d_contiguous(devAcc, d_dst, (Idx)rows, (Idx)cols);
  alpaka::memcpy(queue, viewDev, viewHost);
}

template <typename TQueue, typename TDevHost, typename TDevAcc>
void download_tile_alpaka(TQueue &queue, TDevHost const &devHost,
                          TDevAcc const &devAcc, float *h_dst,
                          const float *d_src, int big_N, int r_off, int c_off,
                          int rows, int cols) {

  float *pDstStart = h_dst + (Idx)r_off * big_N + c_off;
  auto viewHost =
      as_view_2d(devHost, pDstStart, (Idx)rows, (Idx)cols, (Idx)big_N);
  auto viewDev = as_view_2d_contiguous(devAcc, d_src, (Idx)rows, (Idx)cols);
  alpaka::memcpy(queue, viewHost, viewDev);
}

// =============================================================================
// 3. LOGICA BATCHING (Out-of-Core)
// =============================================================================
template <typename TQueue>
void gemm_out_of_core_alpaka(TQueue &queue, float const *h_A, float const *h_B,
                             float *h_C, int M, int N, int K) {

  const int CHUNK_SIZE = 4096;

  auto devAcc = alpaka::getDev(queue);
  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

  // Buffer 2D
  auto extBuf = alpaka::Vec<Dim2, Idx>{(Idx)CHUNK_SIZE, (Idx)CHUNK_SIZE};
  auto bufA = alpaka::allocBuf<float, Idx>(devAcc, extBuf);
  auto bufB = alpaka::allocBuf<float, Idx>(devAcc, extBuf);
  auto bufC = alpaka::allocBuf<float, Idx>(devAcc, extBuf);

  float *d_A = alpaka::getPtrNative(bufA);
  float *d_B = alpaka::getPtrNative(bufB);
  float *d_C = alpaka::getPtrNative(bufC);

  for (int i = 0; i < M; i += CHUNK_SIZE) {
    for (int j = 0; j < N; j += CHUNK_SIZE) {

      int m_curr = std::min(CHUNK_SIZE, M - i);
      int n_curr = std::min(CHUNK_SIZE, N - j);

      for (int l = 0; l < K; l += CHUNK_SIZE) {
        int k_curr = std::min(CHUNK_SIZE, K - l);

        upload_tile_alpaka(queue, devHost, devAcc, d_A, h_A, K, i, l, m_curr,
                           k_curr);
        upload_tile_alpaka(queue, devHost, devAcc, d_B, h_B, N, l, j, k_curr,
                           n_curr);

        bool accumulate = (l > 0);

        int TILE_SIZE = 32;
        Idx blocksY = (m_curr + TILE_SIZE - 1) / TILE_SIZE;
        Idx blocksX = (n_curr + TILE_SIZE - 1) / TILE_SIZE;

        auto workDiv = alpaka::WorkDivMembers<Dim2, Idx>{
            alpaka::Vec<Dim2, Idx>{blocksY, blocksX},
            alpaka::Vec<Dim2, Idx>{(Idx)TILE_SIZE, (Idx)TILE_SIZE},
            alpaka::Vec<Dim2, Idx>{1u, 1u}};

        GemmFullKernel<32> kernel;
        alpaka::exec<Acc>(queue, workDiv, kernel, d_A, d_B, d_C, m_curr, n_curr,
                          k_curr, accumulate);
      }
      download_tile_alpaka(queue, devHost, devAcc, h_C, d_C, N, i, j, m_curr,
                           n_curr);
    }
  }
  alpaka::wait(queue);
}

// =============================================================================
// 4. MAIN ENTRY POINT (Ottimizzato)
// =============================================================================
template <typename TQueue>
void gemm_alpaka_full_options(TQueue &queue, float const *A, float const *B,
                              float *C, GemmShape s) {

  size_t required = (size_t)s.m * s.k + (size_t)s.k * s.n + (size_t)s.m * s.n;
  required *= sizeof(float);

  const size_t SAFETY_MARGIN = 500ULL * 1024 * 1024;
  const size_t ESTIMATED_FREE_VRAM = 4ULL * 1024 * 1024 * 1024;

  // Se stiamo sotto il limite, usiamo il "Standard Path" ZERO-COPY.
  // Assumiamo che A, B, C siano già puntatori DEVICE (come fa il benchmark
  // attuale).
  bool use_batching = (required + SAFETY_MARGIN) >= ESTIMATED_FREE_VRAM;

  if (!use_batching) {
    // --- STANDARD PATH (ZERO-COPY) ---
    // Usiamo direttamente A, B, C senza allocare nulla di nuovo.
    // Questo elimina l'overhead di allocazione e copia che rallentava il
    // benchmark.

    int TILE_SIZE = 32;
    Idx blocksY = (s.m + TILE_SIZE - 1) / TILE_SIZE;
    Idx blocksX = (s.n + TILE_SIZE - 1) / TILE_SIZE;

    auto workDiv = alpaka::WorkDivMembers<Dim2, Idx>{
        alpaka::Vec<Dim2, Idx>{blocksY, blocksX},
        alpaka::Vec<Dim2, Idx>{(Idx)TILE_SIZE, (Idx)TILE_SIZE},
        alpaka::Vec<Dim2, Idx>{1u, 1u}};

    // Passiamo i puntatori originali
    GemmFullKernel<32> kernel;
    alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, s.m, s.n, s.k, false);
    alpaka::wait(queue);

  } else {
    // --- BATCHING PATH ---
    // Attenzione: questo path si aspetta puntatori HOST per fare l'upload.
    // Se il benchmark passa puntatori DEVICE, questo path potrebbe fallire o
    // dare risultati errati. Ma per testare il batching vero (matrici giganti),
    // dovrai modificare il benchmark per non allocare tutto su GPU all'inizio.
    gemm_out_of_core_alpaka(queue, A, B, C, s.m, s.n, s.k);
  }
}

// Istanziazione
template void gemm_alpaka_full_options<QueueType>(QueueType &queue,
                                                  float const *A,
                                                  float const *B, float *C,
                                                  GemmShape shape);

} // namespace gemm
