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

template <typename TDev, typename TPtr>
auto as_view_2d(TDev const &dev, TPtr *ptr, Idx rows, Idx cols,
                Idx pitch_elems) {
  auto ext = alpaka::Vec<Dim2, Idx>{rows, cols};
  auto strides = alpaka::Vec<Dim2, Idx>{pitch_elems, 1u};
  return alpaka::createView(dev, ptr, ext, strides);
}

template <typename TDev, typename TPtr>
auto as_view_2d_contiguous(TDev const &dev, TPtr *ptr, Idx rows, Idx cols) {
  return as_view_2d(dev, ptr, rows, cols, cols);
}

template <typename TDev, typename TPtr>
auto as_view_1d(TDev const &dev, TPtr *ptr, Idx elems) {
  auto ext = alpaka::Vec<Dim1, Idx>{elems};
  return alpaka::createView(dev, ptr, ext);
}

// TM: Righe del Tile C
// TN: Colonne del Tile C
// TK: Profondità di accumulo
template <int TM, int TN, int TK> struct GemmRectKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K,
                                bool accumulate) const {

    // Indici
    auto const globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto const localIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);

    int localRow = localIdx[0]; // Range 0..TM-1
    int localCol = localIdx[1]; // Range 0..TN-1

    // Linearizzazione thread ID (per caricamento collaborativo)
    int threadId = localRow * TN + localCol;
    int blockSize = TM * TN;

    // Shared Memory (Rettangolare)
    float (&As)[TM][TK] =
        alpaka::declareSharedVar<float[TM][TK], __COUNTER__>(acc);
    float (&Bs)[TK][TN] =
        alpaka::declareSharedVar<float[TK][TN], __COUNTER__>(acc);

    float accVal = 0.0f;
    int numTiles = (K + TK - 1) / TK;

    for (int t = 0; t < numTiles; ++t) {
      int tiledK = t * TK;

      for (int i = threadId; i < TM * TK; i += blockSize) {
        int r = i / TK;
        int c = i % TK;
        int globalR = globalIdx[0] - localRow + r;
        int globalC = tiledK + c;

        if (globalR < M && globalC < K) {
          As[r][c] = A[globalR * K + globalC];
        } else {
          As[r][c] = 0.0f;
        }
      }

      for (int i = threadId; i < TK * TN; i += blockSize) {
        int r = i / TN;
        int c = i % TN;
        int globalR = tiledK + r;
        int globalC = globalIdx[1] - localCol + c;

        if (globalR < K && globalC < N) {
          Bs[r][c] = B[globalR * N + globalC];
        } else {
          Bs[r][c] = 0.0f;
        }
      }

      alpaka::syncBlockThreads(acc);

      for (int k = 0; k < TK; ++k) {
        accVal += As[localRow][k] * Bs[k][localCol];
      }
      alpaka::syncBlockThreads(acc);
    }

    if (globalIdx[0] < M && globalIdx[1] < N) {
      Idx idx = globalIdx[0] * N + globalIdx[1];
      if (accumulate)
        C[idx] += accVal;
      else
        C[idx] = accVal;
    }
  }
};

template <int TM, int TN, int TK, typename TQueue, typename TPtrA,
          typename TPtrB, typename TPtrC>
void launch_rect_kernel(TQueue &queue, TPtrA const A, TPtrB const B, TPtrC C,
                        int m, int n, int k, bool accumulate) {

  Idx blocksY = (m + TM - 1) / TM;
  Idx blocksX = (n + TN - 1) / TN;

  auto workDiv = alpaka::WorkDivMembers<Dim2, Idx>{
      alpaka::Vec<Dim2, Idx>{blocksY, blocksX},
      alpaka::Vec<Dim2, Idx>{(Idx)TM, (Idx)TN}, alpaka::Vec<Dim2, Idx>{1u, 1u}};

  GemmRectKernel<TM, TN, TK> kernel;
  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, m, n, k, accumulate);
}

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

enum class TileConfig { Square32, Rect16x32, Square16 };

template <typename TQueue>
void gemm_out_of_core_alpaka(TQueue &queue, float const *h_A, float const *h_B,
                             float *h_C, int M, int N, int K,
                             TileConfig config) {

  const int CHUNK_SIZE = 4096;
  auto devAcc = alpaka::getDev(queue);
  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

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

        // Dispatcher Dinamico Interno
        if (config == TileConfig::Square32)
          launch_rect_kernel<32, 32, 32>(queue, d_A, d_B, d_C, m_curr, n_curr,
                                         k_curr, accumulate);
        else if (config == TileConfig::Rect16x32)
          launch_rect_kernel<16, 32, 32>(queue, d_A, d_B, d_C, m_curr, n_curr,
                                         k_curr, accumulate);
        else
          launch_rect_kernel<16, 16, 16>(queue, d_A, d_B, d_C, m_curr, n_curr,
                                         k_curr, accumulate);
      }
      download_tile_alpaka(queue, devHost, devAcc, h_C, d_C, N, i, j, m_curr,
                           n_curr);
    }
  }
  alpaka::wait(queue);
}

template <typename TAcc>
TileConfig get_optimal_tile_config(alpaka::PlatformCudaRt const &platform,
                                   int devIdx) {
  auto dev = alpaka::getDevByIdx(platform, devIdx);
  auto props = alpaka::getAccDevProps<TAcc>(dev);

  if (props.m_blockThreadCountMax >= 1024) {
    return TileConfig::Square32; // Tenta la massima potenza
  } else if (props.m_blockThreadCountMax >= 512) {
    return TileConfig::Rect16x32;
  }
  return TileConfig::Square16;
}

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

  // 3. Decisione Tiling Dinamico
  auto config = get_optimal_tile_config<Acc>(alpaka::PlatformCudaRt{}, 0);

  if (!use_batching) {
    // --- STANDARD PATH (ZERO-COPY) ---
    // Puntatori Device passati direttamente

    if (config == TileConfig::Square32) {
      launch_rect_kernel<32, 32, 32>(queue, A, B, C, s.m, s.n, s.k, false);
    } else if (config == TileConfig::Rect16x32) {
      launch_rect_kernel<16, 32, 32>(queue, A, B, C, s.m, s.n, s.k, false);
    } else {
      launch_rect_kernel<16, 16, 16>(queue, A, B, C, s.m, s.n, s.k, false);
    }
    alpaka::wait(queue);

  } else {
    // --- BATCHING PATH ---
    // Puntatori Host attesi
    gemm_out_of_core_alpaka(queue, A, B, C, s.m, s.n, s.k, config);
  }
}

// Istanziazione
template void gemm_alpaka_full_options<QueueType>(QueueType &queue,
                                                  float const *A,
                                                  float const *B, float *C,
                                                  GemmShape shape);

} // namespace gemm
