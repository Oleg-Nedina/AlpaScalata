#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>

// Header necessari per buffer CPU e GPU
#include <alpaka/mem/buf/cpu/BufCpu.hpp>
// Se servisse esplicitamente per GPU, di solito è incluso in alpaka.hpp,
// ma se da ancora errori sui traits aggiungi:
// <alpaka/mem/buf/cuda/BufCudaRt.hpp> (o simile a seconda della versione)

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory> // Per std::unique_ptr
#include <vector>

namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;
using Dim1 = alpaka::DimInt<1>;
// Definiamo i tipi base
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
// Questo è il tipo che mancava: Il Device corrispondente all'Acceleratore
using DevAcc = alpaka::Dev<Acc>;
using Platform = alpaka::Platform<DevAcc>;

// Coda BLOCCANTE per le operazioni standard (sincrone)
using QueueBlocking = alpaka::Queue<Acc, alpaka::Blocking>;
// Coda NON BLOCCANTE per i flussi asincroni (Async Streams)
using QueueAsync = alpaka::Queue<Acc, alpaka::NonBlocking>;

// =============================================================================
// 0. HELPERS PER VIEW
// =============================================================================
template <typename TDev, typename TPtr>
auto as_view_2d(TDev const &dev, TPtr *ptr, Idx rows, Idx cols,
                Idx pitch_elems) {
  auto ext = alpaka::Vec<Dim2, Idx>{rows, cols};
  auto strides = alpaka::Vec<Dim2, Idx>{pitch_elems, 1u};
  return alpaka::createView(dev, ptr, ext, strides);
}

// =============================================================================
// 1. KERNEL GEMM (INVARIATO)
// =============================================================================
template <int TM, int TN, int TK> struct GemmRectKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K,
                                bool accumulate) const {

    auto const globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto const localIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);

    int localRow = localIdx[0];
    int localCol = localIdx[1];
    int threadId = localRow * TN + localCol;
    int blockSize = TM * TN;

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
        if (globalR < M && globalC < K)
          As[r][c] = A[globalR * K + globalC];
        else
          As[r][c] = 0.0f;
      }

      for (int i = threadId; i < TK * TN; i += blockSize) {
        int r = i / TN;
        int c = i % TN;
        int globalR = tiledK + r;
        int globalC = globalIdx[1] - localCol + c;
        if (globalR < K && globalC < N)
          Bs[r][c] = B[globalR * N + globalC];
        else
          Bs[r][c] = 0.0f;
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

// =============================================================================
// 2. DISPATCHER KERNEL
// =============================================================================
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

// =============================================================================
// 3. LOGICA BATCHING ASINCRONA (PIPELINE) [CORRETTA]
// =============================================================================
enum class TileConfig { Square32, Rect16x32, Square16, Rect16x64 };

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

// STRUTTURA CONTESTO ASINCRONO
struct StreamContext {
  // Usiamo QueueAsync (NonBlocking) altrimenti non c'è parallelismo!
  QueueAsync queue;

  alpaka::Buf<DevAcc, float, Dim2, Idx> bufA;
  alpaka::Buf<DevAcc, float, Dim2, Idx> bufB;
  alpaka::Buf<DevAcc, float, Dim2, Idx> bufC;

  float *d_A;
  float *d_B;
  float *d_C;

  // CORREZIONE QUI: Il costruttore prende "DevAcc const& dev", NON "Acc"
  StreamContext(DevAcc const &dev, Idx chunkSize)
      : queue(dev) // La Queue vuole un Device
        ,
        bufA(alpaka::allocBuf<float, Idx>(
            dev, alpaka::Vec<Dim2, Idx>{chunkSize,
                                        chunkSize})) // allocBuf vuole un Device
        ,
        bufB(alpaka::allocBuf<float, Idx>(
            dev, alpaka::Vec<Dim2, Idx>{chunkSize, chunkSize})),
        bufC(alpaka::allocBuf<float, Idx>(
            dev, alpaka::Vec<Dim2, Idx>{chunkSize, chunkSize})) {
    d_A = alpaka::getPtrNative(bufA);
    d_B = alpaka::getPtrNative(bufB);
    d_C = alpaka::getPtrNative(bufC);
  }
};

template <typename TQueue>
void gemm_out_of_core_alpaka(TQueue &original_queue, float const *h_A,
                             float const *h_B, float *h_C, int M, int N, int K,
                             TileConfig config) {

  const int CHUNK_SIZE = 4096;

  // Otteniamo il DEVICE dalla coda originale
  auto devAcc = alpaka::getDev(original_queue);
  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

  // Creiamo 3 Stream Indipendenti
  const int NUM_STREAMS = 3;
  std::vector<std::unique_ptr<StreamContext>> streams;

  for (int i = 0; i < NUM_STREAMS; ++i) {
    // Passiamo devAcc che è di tipo "DevAcc" (Device), non "Acc"
    streams.push_back(std::make_unique<StreamContext>(devAcc, CHUNK_SIZE));
  }

  int stream_idx = 0;

  for (int i = 0; i < M; i += CHUNK_SIZE) {
    for (int j = 0; j < N; j += CHUNK_SIZE) {

      auto &ctx = *streams[stream_idx];
      int m_curr = std::min(CHUNK_SIZE, M - i);
      int n_curr = std::min(CHUNK_SIZE, N - j);

      for (int l = 0; l < K; l += CHUNK_SIZE) {
        int k_curr = std::min(CHUNK_SIZE, K - l);

        upload_tile_alpaka(ctx.queue, devHost, devAcc, ctx.d_A, h_A, K, i, l,
                           m_curr, k_curr);
        upload_tile_alpaka(ctx.queue, devHost, devAcc, ctx.d_B, h_B, N, l, j,
                           k_curr, n_curr);

        bool accumulate = (l > 0);

        if (config == TileConfig::Square32)
          launch_rect_kernel<32, 32, 32>(ctx.queue, ctx.d_A, ctx.d_B, ctx.d_C,
                                         m_curr, n_curr, k_curr, accumulate);
        else if (config == TileConfig::Rect16x32)
          launch_rect_kernel<16, 32, 32>(ctx.queue, ctx.d_A, ctx.d_B, ctx.d_C,
                                         m_curr, n_curr, k_curr, accumulate);
        else if (config == TileConfig::Rect16x64)
          launch_rect_kernel<16, 64, 32>(ctx.queue, ctx.d_A, ctx.d_B, ctx.d_C,
                                         m_curr, n_curr, k_curr, accumulate);
        else
          launch_rect_kernel<16, 16, 16>(ctx.queue, ctx.d_A, ctx.d_B, ctx.d_C,
                                         m_curr, n_curr, k_curr, accumulate);
      }

      download_tile_alpaka(ctx.queue, devHost, devAcc, h_C, ctx.d_C, N, i, j,
                           m_curr, n_curr);
      stream_idx = (stream_idx + 1) % NUM_STREAMS;
    }
  }

  // Barriera Finale
  for (auto &ctx : streams) {
    alpaka::wait(ctx->queue);
  }
}

// =============================================================================
// 4. INTELLIGENZA DINAMICA
// =============================================================================

template <typename TAcc>
std::size_t estimate_free_memory(alpaka::PlatformCudaRt const &platform,
                                 int devIdx) {
  auto dev = alpaka::getDevByIdx(platform, devIdx);
  auto props = alpaka::getAccDevProps<TAcc>(dev);
  std::size_t total_vram = props.m_globalMemSizeBytes;
  std::size_t system_reserve = 1024ULL * 1024 * 1024;
  if (total_vram > system_reserve)
    return total_vram - system_reserve;
  else
    return total_vram / 2;
}

template <typename TAcc>
TileConfig get_optimal_tile_config(alpaka::PlatformCudaRt const &platform,
                                   int devIdx) {
  // Configurazione "aggressiva" 16x64 per sfruttare la banda
  return TileConfig::Rect16x64;
}

// =============================================================================
// 5. MAIN ENTRY POINT
// =============================================================================
template <typename TQueue>
void gemm_alpaka_full_options(TQueue &queue, float const *A, float const *B,
                              float *C, GemmShape s) {

  size_t required = (size_t)s.m * s.k + (size_t)s.k * s.n + (size_t)s.m * s.n;
  required *= sizeof(float);

  size_t estimated_free_vram =
      estimate_free_memory<Acc>(alpaka::PlatformCudaRt{}, 0);
  const size_t SAFETY_MARGIN = 100ULL * 1024 * 1024;

  // Attiva batching se la memoria richiesta supera quella disponibile
  bool use_batching = (required + SAFETY_MARGIN) >= estimated_free_vram;

  auto config = get_optimal_tile_config<Acc>(alpaka::PlatformCudaRt{}, 0);

  if (!use_batching) {
    // --- STANDARD PATH (IN-CORE) ---
    if (config == TileConfig::Square32) {
      launch_rect_kernel<32, 32, 32>(queue, A, B, C, s.m, s.n, s.k, false);
    } else if (config == TileConfig::Rect16x32) {
      launch_rect_kernel<16, 32, 32>(queue, A, B, C, s.m, s.n, s.k, false);
    } else if (config == TileConfig::Rect16x64) {
      launch_rect_kernel<16, 64, 32>(queue, A, B, C, s.m, s.n, s.k, false);
    } else {
      launch_rect_kernel<16, 16, 16>(queue, A, B, C, s.m, s.n, s.k, false);
    }
    alpaka::wait(queue);

  } else {
    // --- BATCHING PATH (PIPELINED) ---
    gemm_out_of_core_alpaka(queue, A, B, C, s.m, s.n, s.k, config);
  }
}

// Istanziazione Esplicita
// Nota: Qui usiamo QueueBlocking per l'interfaccia esterna, ma internamente
// usiamo QueueAsync
template void gemm_alpaka_full_options<QueueBlocking>(QueueBlocking &queue,
                                                      float const *A,
                                                      float const *B, float *C,
                                                      GemmShape shape);

} // namespace gemm
