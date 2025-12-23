#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cmath>
#include <vector>

namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;

// =============================================================================
// 1. KERNEL AGGIORNATO (Supporta Accumulo +=)
// =============================================================================
template <int TILE_SIZE> struct GemmFullKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K,
                                bool accumulate) const { // <--- NUOVO PARAMETRO

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

      for (int k = 0; k < TILE_SIZE; ++k) {
        accVal += As[localRow][k] * Bs[k][localCol];
      }
      alpaka::syncBlockThreads(acc);
    }

    // Write Result con logica di accumulo
    if (globalIdx[0] < M && globalIdx[1] < N) {
      Idx idx = globalIdx[0] * N + globalIdx[1];
      if (accumulate) {
        C[idx] += accVal; // Somma al parziale esistente
      } else {
        C[idx] = accVal; // Sovrascrivi (primo chunk)
      }
    }
  }
};

// =============================================================================
// 2. HELPER COPY 2D (Sostituisce cudaMemcpy2D)
// =============================================================================
// Copia un rettangolo (Tile) dalla matrice Host gigante al buffer Device
// piccolo
template <typename TQueue, typename TDevHost, typename TDevAcc>
void upload_tile_alpaka(TQueue &queue, TDevHost const &devHost,
                        TDevAcc const &devAcc, float *d_dst, const float *h_src,
                        int big_M, int big_N, // Dimensioni totali Host
                        int r_off,
                        int c_off, // Offset (angolo in alto a sinistra)
                        int rows,
                        int cols) { // Dimensioni del rettangolo da copiare

  // 1. Definiamo l'estensione del rettangolo da copiare
  auto ext = alpaka::Vec<Dim2, Idx>{(Idx)rows, (Idx)cols};

  // 2. View HOST (Strided): La memoria host è larga big_N, noi ne prendiamo
  // solo un pezzo. Il puntatore parte dall'offset corretto.
  float const *pSrcStart = h_src + (Idx)r_off * big_N + c_off;

  // Stride [Y, X]: Per scendere di una riga in Host, devo saltare big_N
  // elementi. Per muovermi di una colonna, salto 1 elemento.
  auto stridesHost = alpaka::Vec<Dim2, Idx>{(Idx)big_N, 1u};

  // Creiamo la view Host "virtuale" usando ViewPlainPtr
  auto viewHost = alpaka::createView(devHost, pSrcStart, ext, stridesHost);

  // 3. View DEVICE (Packed): Il buffer device è piccolo e contiguo (pitch =
  // cols)
  auto viewDev = alpaka::createView(devAcc, d_dst, ext);

  // 4. Copia 2D
  alpaka::memcpy(queue, viewDev, viewHost);
}

// Viceversa: Download dal buffer Device alla matrice Host
template <typename TQueue, typename TDevHost, typename TDevAcc>
void download_tile_alpaka(TQueue &queue, TDevHost const &devHost,
                          TDevAcc const &devAcc, float *h_dst,
                          const float *d_src, int big_M, int big_N, int r_off,
                          int c_off, int rows, int cols) {

  auto ext = alpaka::Vec<Dim2, Idx>{(Idx)rows, (Idx)cols};

  float *pDstStart = h_dst + (Idx)r_off * big_N + c_off;
  auto stridesHost = alpaka::Vec<Dim2, Idx>{(Idx)big_N, 1u};

  auto viewHost = alpaka::createView(devHost, pDstStart, ext, stridesHost);
  auto viewDev = alpaka::createView(devAcc, d_src, ext);

  alpaka::memcpy(queue, viewHost, viewDev);
}

// =============================================================================
// 3. LOGICA BATCHING (Out-of-Core)
// =============================================================================
template <typename TQueue>
void gemm_out_of_core_alpaka(TQueue &queue, float const *h_A, float const *h_B,
                             float *h_C, int M, int N, int K) {

  // Dimensione del blocco in memoria (es. 4096 x 4096 float = ~64MB a buffer)
  const int CHUNK_SIZE = 4096;

  // Device e Host
  auto devAcc = alpaka::getDev(queue);
  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);

  // Allocazione Buffer GPU (Riutilizzabili)
  auto extBuf = alpaka::Vec<Dim2, Idx>{(Idx)CHUNK_SIZE, (Idx)CHUNK_SIZE};
  Idx bufElems = CHUNK_SIZE * CHUNK_SIZE;

  auto bufA = alpaka::allocBuf<float, Idx>(devAcc, extBuf);
  auto bufB = alpaka::allocBuf<float, Idx>(devAcc, extBuf);
  auto bufC = alpaka::allocBuf<float, Idx>(devAcc, extBuf);

  float *d_A = alpaka::getPtrNative(bufA);
  float *d_B = alpaka::getPtrNative(bufB);
  float *d_C = alpaka::getPtrNative(bufC);

  // Loop su blocchi del risultato C (M x N)
  for (int i = 0; i < M; i += CHUNK_SIZE) {
    for (int j = 0; j < N; j += CHUNK_SIZE) {

      int m_curr = std::min(CHUNK_SIZE, M - i);
      int n_curr = std::min(CHUNK_SIZE, N - j);

      // Loop sulla dimensione di accumulo K
      for (int l = 0; l < K; l += CHUNK_SIZE) {
        int k_curr = std::min(CHUNK_SIZE, K - l);

        // 1. Upload Tiles (Host -> Device)
        upload_tile_alpaka(queue, devHost, devAcc, d_A, h_A, M, K, i, l, m_curr,
                           k_curr);
        upload_tile_alpaka(queue, devHost, devAcc, d_B, h_B, K, N, l, j, k_curr,
                           n_curr);

        // Opzionale: Wait copy? No, Alpaka serializza sulla coda di default.

        // 2. Esecuzione Kernel
        bool accumulate = (l > 0); // Se non è il primo pezzo, somma!

        // Usiamo il Tiling 32 fisso per semplicità nel batching (o puoi
        // richiamare il dinamico) Qui hardcodiamo 32 per brevità, ma puoi
        // mettere l'if dinamico
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

      // 3. Download Result (Device -> Host)
      download_tile_alpaka(queue, devHost, devAcc, h_C, d_C, M, N, i, j, m_curr,
                           n_curr);
    }
  }

  alpaka::wait(queue);
}

// =============================================================================
// 4. MAIN ENTRY POINT (Smart Dispatch)
// =============================================================================
template <typename TQueue>
void gemm_alpaka_full_options(TQueue &queue, float const *A, float const *B,
                              float *C, GemmShape s) {

  // 1. Controllo Memoria Disponibile (Alpaka API)
  auto dev = alpaka::getDev(queue);
  size_t free_bytes = 0, total_bytes = 0;

  // Nota: getMemBytes potrebbe non essere supportato da tutti i backend, ma su
  // CUDA sì. Se fallisce, usiamo un fallback o assumiamo batching.
  try {
    alpaka::getMemBytes(dev, free_bytes, total_bytes);
  } catch (...) {
    free_bytes = 1024 * 1024 * 1024; // Fallback: fingiamo 1GB libero
  }

  size_t required = (size_t)s.m * s.k + (size_t)s.k * s.n + (size_t)s.m * s.n;
  required *= sizeof(float);
  size_t margin = 500 * 1024 * 1024; // 500MB margine sicurezza

  // 2. Decisione: Standard o Batching?
  if (required + margin < free_bytes) {
    // --- STANDARD (Tutto in memoria) ---
    // (Usa la tua implementazione tiled ottimizzata esistente, ma qui la ripeto
    // inline per completezza) Allocazione totale...
    Idx n_elems_A = s.m * s.k;
    Idx n_elems_B = s.k * s.n;
    Idx n_elems_C = s.m * s.n;

    auto bufA = alpaka::allocBuf<float, Idx>(
        dev, alpaka::Vec<Dim2, Idx>{(Idx)s.m, (Idx)s.k}); // O 1D estesa
    auto bufB = alpaka::allocBuf<float, Idx>(
        dev, alpaka::Vec<Dim2, Idx>{(Idx)s.k, (Idx)s.n});
    auto bufC = alpaka::allocBuf<float, Idx>(
        dev, alpaka::Vec<Dim2, Idx>{(Idx)s.m, (Idx)s.n});

    // Copia 1D classica (rapida)
    alpaka::memcpy(queue, bufA, A, n_elems_A);
    alpaka::memcpy(queue, bufB, B, n_elems_B);

    // Kernel Launch (Standard 32x32 tiled)
    int TILE_SIZE = 32;
    Idx blocksY = (s.m + TILE_SIZE - 1) / TILE_SIZE;
    Idx blocksX = (s.n + TILE_SIZE - 1) / TILE_SIZE;
    auto workDiv = alpaka::WorkDivMembers<Dim2, Idx>{
        alpaka::Vec<Dim2, Idx>{blocksY, blocksX},
        alpaka::Vec<Dim2, Idx>{(Idx)TILE_SIZE, (Idx)TILE_SIZE},
        alpaka::Vec<Dim2, Idx>{1u, 1u}};
    GemmFullKernel<32> kernel;
    alpaka::exec<Acc>(queue, workDiv, kernel, alpaka::getPtrNative(bufA),
                      alpaka::getPtrNative(bufB), alpaka::getPtrNative(bufC),
                      s.m, s.n, s.k, false);

    alpaka::memcpy(queue, C, bufC, n_elems_C);
    alpaka::wait(queue);

  } else {
    // --- BATCHING (Out-of-Core) ---
    // Le matrici rimangono su Host, passiamo i puntatori raw host
    gemm_out_of_core_alpaka(queue, A, B, C, s.m, s.n, s.k);
  }
}

// Istanziazione
template void gemm_alpaka_full_options<QueueType>(QueueType &queue,
                                                  float const *A,
                                                  float const *B, float *C,
                                                  GemmShape shape);

} // namespace gemm
