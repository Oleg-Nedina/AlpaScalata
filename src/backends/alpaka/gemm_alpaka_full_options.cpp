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
// 0. HELPERS PER VIEW (Safety & Compatibility)
// =============================================================================
// Crea View 2D da puntatore raw (Fondamentale per Out-of-Core)
template <typename TDev, typename TPtr>
auto as_view_2d(TDev const &dev, TPtr *ptr, Idx rows, Idx cols,
                Idx pitch_elems) {
  auto ext = alpaka::Vec<Dim2, Idx>{rows, cols};
  auto strides = alpaka::Vec<Dim2, Idx>{pitch_elems, 1u};
  return alpaka::createView(dev, ptr, ext, strides);
}

// Crea View 2D Contigua (Pitch == Cols)
template <typename TDev, typename TPtr>
auto as_view_2d_contiguous(TDev const &dev, TPtr *ptr, Idx rows, Idx cols) {
  return as_view_2d(dev, ptr, rows, cols, cols);
}

// Crea View 1D (Per Standard Path veloce)
template <typename TDev, typename TPtr>
auto as_view_1d(TDev const &dev, TPtr *ptr, Idx elems) {
  auto ext = alpaka::Vec<Dim1, Idx>{elems};
  return alpaka::createView(dev, ptr, ext);
}

// =============================================================================
// 1. KERNEL GEMM (Tiled + Accumulate)
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

    // Shared Memory Statica
    float (&As)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);
    float (&Bs)[TILE_SIZE][TILE_SIZE] =
        alpaka::declareSharedVar<float[TILE_SIZE][TILE_SIZE], __COUNTER__>(acc);

    float accVal = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
      int tiledK = t * TILE_SIZE;

      // Load A
      if (globalIdx[0] < M && (tiledK + localCol) < K) {
        As[localRow][localCol] = A[globalIdx[0] * K + (tiledK + localCol)];
      } else {
        As[localRow][localCol] = 0.0f;
      }

      // Load B
      if ((tiledK + localRow) < K && globalIdx[1] < N) {
        Bs[localRow][localCol] = B[(tiledK + localRow) * N + globalIdx[1]];
      } else {
        Bs[localRow][localCol] = 0.0f;
      }

      alpaka::syncBlockThreads(acc);

      // Compute
      for (int k = 0; k < TILE_SIZE; ++k) {
        accVal += As[localRow][k] * Bs[k][localCol];
      }
      alpaka::syncBlockThreads(acc);
    }

    // Store / Accumulate
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
// 2. DISPATCHER KERNEL (Helper per lanciare 16 o 32)
// =============================================================================
template <int TILE_SIZE, typename TQueue, typename TPtrA, typename TPtrB,
          typename TPtrC>
void launch_kernel_template(TQueue &queue, TPtrA const A, TPtrB const B,
                            TPtrC C, int m, int n, int k, bool accumulate) {

  Idx blocksY = (m + TILE_SIZE - 1) / TILE_SIZE;
  Idx blocksX = (n + TILE_SIZE - 1) / TILE_SIZE;

  auto workDiv = alpaka::WorkDivMembers<Dim2, Idx>{
      alpaka::Vec<Dim2, Idx>{blocksY, blocksX},
      alpaka::Vec<Dim2, Idx>{(Idx)TILE_SIZE, (Idx)TILE_SIZE},
      alpaka::Vec<Dim2, Idx>{1u, 1u}};

  GemmFullKernel<TILE_SIZE> kernel;
  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, m, n, k, accumulate);
}

// =============================================================================
// 3. COPY UTILS (Per Batching)
// =============================================================================
template <typename TQueue, typename TDevHost, typename TDevAcc>
void upload_tile_alpaka(TQueue &queue, TDevHost const &devHost,
                        TDevAcc const &devAcc, float *d_dst, const float *h_src,
                        int big_N, int r_off, int c_off, int rows, int cols) {
  float const *pSrcStart = h_src + (Idx)r_off * big_N + c_off;
  auto viewHost =
      as_view_2d(devHost, pSrcStart, (Idx)rows, (Idx)cols, (Idx)big_N);
  auto ext = alpaka::Vec<Dim2, Idx>{(Idx)rows, (Idx)cols};
  auto viewDev = alpaka::createView(devAcc, d_dst, ext);
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
  auto ext = alpaka::Vec<Dim2, Idx>{(Idx)rows, (Idx)cols};
  auto viewDev = alpaka::createView(devAcc, d_src, ext);
  alpaka::memcpy(queue, viewHost, viewDev);
}

// =============================================================================
// 4. LOGICA BATCHING (Out-of-Core)
// =============================================================================
template <typename TQueue>
void gemm_out_of_core_alpaka(TQueue &queue, float const *h_A, float const *h_B,
                             float *h_C, int M, int N, int K,
                             int optimal_tile) {

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

        // Dispatcher Dinamico anche nel Batching
        if (optimal_tile >= 32) {
          launch_kernel_template<32>(queue, d_A, d_B, d_C, m_curr, n_curr,
                                     k_curr, accumulate);
        } else {
          launch_kernel_template<16>(queue, d_A, d_B, d_C, m_curr, n_curr,
                                     k_curr, accumulate);
        }
      }
      download_tile_alpaka(queue, devHost, devAcc, h_C, d_C, N, i, j, m_curr,
                           n_curr);
    }
  }
  alpaka::wait(queue);
}

// =============================================================================
// 5. SELECTOR TILE SIZE (Portabile)
// =============================================================================
template <typename TAcc>
int get_optimal_tile_width_alpaka(alpaka::PlatformCudaRt const &platform,
                                  int devIdx) {
  auto dev = alpaka::getDevByIdx(platform, devIdx);
  auto props = alpaka::getAccDevProps<TAcc>(dev);
  if (props.m_blockThreadCountMax >= 1024)
    return 32;
  return 16;
}

// =============================================================================
// 6. MAIN ENTRY POINT
// =============================================================================
template <typename TQueue>
void gemm_alpaka_full_options(TQueue &queue, float const *A, float const *B,
                              float *C, GemmShape s) {

  // 1. Calcolo Memoria
  size_t required = (size_t)s.m * s.k + (size_t)s.k * s.n + (size_t)s.m * s.n;
  required *= sizeof(float);

  const size_t SAFETY_MARGIN = 500ULL * 1024 * 1024;
  const size_t ESTIMATED_FREE_VRAM = 4ULL * 1024 * 1024 * 1024;

  // 2. Decisione Batching
  bool use_batching = (required + SAFETY_MARGIN) >= ESTIMATED_FREE_VRAM;

  // 3. Decisione Tiling (Dinamico)
  int optimal_tile =
      get_optimal_tile_width_alpaka<Acc>(alpaka::PlatformCudaRt{}, 0);

  if (!use_batching) {
    // --- STANDARD PATH (In VRAM, Zero-Copy) ---
    // Usiamo direttamente i puntatori forniti (assumendo siano già su GPU)

    if (optimal_tile >= 32) {
      launch_kernel_template<32>(queue, A, B, C, s.m, s.n, s.k, false);
    } else {
      launch_kernel_template<16>(queue, A, B, C, s.m, s.n, s.k, false);
    }
    alpaka::wait(queue);

  } else {
    // --- BATCHING PATH (Out-of-Core) ---
    // Richiede puntatori Host.
    gemm_out_of_core_alpaka(queue, A, B, C, s.m, s.n, s.k, optimal_tile);
  }
}

// Istanziazione
template void gemm_alpaka_full_options<QueueType>(QueueType &queue,
                                                  float const *A,
                                                  float const *B, float *C,
                                                  GemmShape shape);

} // namespace gemm
