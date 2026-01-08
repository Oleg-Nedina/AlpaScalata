#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <alpaka/mem/buf/cpu/BufCpu.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

/**
 *  * @file gemm_alpaka_full_options.cpp
 * @brief High-Performance Alpaka Backend implementation for GEMM.
 *
 * This file contains the fully optimized implementation of Matrix
 * Multiplication using Alpaka. It features:
 * - **2D Register Tiling (Thread Coarsening):** Increases arithmetic intensity.
 * - **Vectorized Global Loads (float4):** Maximizes memory bandwidth.
 * - **Async Pipeline (Double/Triple Buffering):** Hides PCIe latency for large
 * matrices.
 * - **Hardware Agnostic Design:** Compiles for CUDA, HIP, or CPU via Alpaka.
 *
 * @note This implementation assumes that input matrices have been padded
 * on the host side to ensure dimensions (M, N, K) are multiples of 4.
 */
namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;
using Dim1 = alpaka::DimInt<1>;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using DevAcc = alpaka::Dev<Acc>;
using Platform = alpaka::Platform<DevAcc>;
using QueueBlocking = alpaka::Queue<Acc, alpaka::Blocking>;
using QueueAsync = alpaka::Queue<Acc, alpaka::NonBlocking>;

/**
 * @brief Creates a 2D Alpaka View from a raw pointer.
 *
 * Helper function to wrap a raw pointer into an Alpaka view with 2D strides.
 * This is used to simplify memory copies between Host and Device.
 *
 * @tparam TDev The device type (Host or Accelerator).
 * @tparam TPtr The pointer type.
 * @param dev The device instance associated with the pointer.
 * @param ptr Raw pointer to the data buffer.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param pitch_elems The pitch (stride) in elements between consecutive rows.
 * @return An alpaka::View representing the 2D memory region.
 */
template <typename TDev, typename TPtr>
auto as_view_2d(TDev const &dev, TPtr *ptr, Idx rows, Idx cols,
                Idx pitch_elems) {
  auto ext = alpaka::Vec<Dim2, Idx>{rows, cols};
  auto strides = alpaka::Vec<Dim2, Idx>{pitch_elems, 1u};
  return alpaka::createView(dev, ptr, ext, strides);
}

/**
 * @brief "Ultimate" Optimized GEMM Kernel using 2D Register Tiling and
 * Vectorization.
 *
 * This functor implements the core matrix multiplication logic on the GPU.
 * It uses a hierarchical tiling strategy to optimize memory access:
 * 1. **Global Memory -> Shared Memory:** Loads data in macro-tiles (BM x BK, BK
 * x BN) using vectorized 128-bit loads (`float4`) to saturate global memory
 * bandwidth.
 * 2. **Shared Memory -> Registers:** Loads data into micro-tiles (TM x TN) to
 * compute the outer product.
 *
 * @details
 * **Thread Coarsening:**
 * Each thread computes a grid of `TM x TN` pixels (e.g., 4x4 = 16 elements).
 * This reduces redundant shared memory accesses and increases instruction-level
 * parallelism.
 *
 * **Vectorization Safety:**
 * The kernel uses `reinterpret_cast<float4*>` for global loads. This requires
 * the host code to guarantee that M, N, K and pointers are aligned to 128-bit
 * boundaries (handled via padding in `gemm_mpi.cpp`).
 *
 * @tparam BM Macro-Tile M dimension (Shared Memory block height).
 * @tparam BN Macro-Tile N dimension (Shared Memory block width).
 * @tparam BK Macro-Tile K dimension (Depth of the tile).
 * @tparam TM Micro-Tile M dimension (Work per thread in Y).
 * @tparam TN Micro-Tile N dimension (Work per thread in X).
 */
template <int BM, int BN, int BK, int TM, int TN> struct GemmCoarsenedKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int M, int N, int K,
                                bool accumulate) const {

    using float4 = ::float4;

    auto const localIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
    auto const blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);

    int ty = localIdx[0];
    int tx = localIdx[1];
    int tid = ty * (BN / TN) + tx;
    int threadsPerBlock = (BM / TM) * (BN / TN);

    int rowStart = blockIdx[0] * BM;
    int colStart = blockIdx[1] * BN;

    float (&As)[BM][BK] =
        alpaka::declareSharedVar<float[BM][BK], __COUNTER__>(acc);
    float (&Bs)[BK][BN] =
        alpaka::declareSharedVar<float[BK][BN], __COUNTER__>(acc);

    float accum[TM][TN];

#pragma unroll
    for (int i = 0; i < TM; ++i)
      for (int j = 0; j < TN; ++j)
        accum[i][j] = 0.0f;

    float regA[TM];
    float regB[TN];

    int numTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numTiles; ++t) {
      int tiledK = t * BK;

      int totalVecsA = (BM * BK) / 4;
      for (int i = tid; i < totalVecsA; i += threadsPerBlock) {
        int vecRow = i / (BK / 4);
        int vecCol = i % (BK / 4);
        int col = vecCol * 4;

        int globalRow = rowStart + vecRow;
        int globalCol = tiledK + col;

        if (globalRow < M && globalCol + 3 < K) {
          float4 loaded =
              *reinterpret_cast<const float4 *>(&A[globalRow * K + globalCol]);

          *reinterpret_cast<float4 *>(&As[vecRow][col]) = loaded;
        } else {
          if (globalRow < M) {
            size_t idxA = (size_t)globalRow * K + globalCol;
            As[vecRow][col] = (globalCol + 0 < K) ? A[idxA + 0] : 0.0f;
            As[vecRow][col + 1] = (globalCol + 1 < K) ? A[idxA + 1] : 0.0f;
            As[vecRow][col + 2] = (globalCol + 2 < K) ? A[idxA + 2] : 0.0f;
            As[vecRow][col + 3] = (globalCol + 3 < K) ? A[idxA + 3] : 0.0f;
          } else {
            As[vecRow][col] = 0.0f;
            As[vecRow][col + 1] = 0.0f;
            As[vecRow][col + 2] = 0.0f;
            As[vecRow][col + 3] = 0.0f;
          }
        }
      }

      int totalVecsB = (BK * BN) / 4;
      for (int i = tid; i < totalVecsB; i += threadsPerBlock) {
        int vecRow = i / (BN / 4);
        int vecCol = i % (BN / 4);
        int col = vecCol * 4;

        int globalRow = tiledK + vecRow;
        int globalCol = colStart + col;

        if (globalRow < K && globalCol + 3 < N) {
          float4 loaded =
              *reinterpret_cast<const float4 *>(&B[globalRow * N + globalCol]);
          *reinterpret_cast<float4 *>(&Bs[vecRow][col]) = loaded;
        } else {
          if (globalRow < K) {
            size_t idxB = (size_t)globalRow * N + globalCol;
            Bs[vecRow][col] = (globalCol + 0 < N) ? B[idxB + 0] : 0.0f;
            Bs[vecRow][col + 1] = (globalCol + 1 < N) ? B[idxB + 1] : 0.0f;
            Bs[vecRow][col + 2] = (globalCol + 2 < N) ? B[idxB + 2] : 0.0f;
            Bs[vecRow][col + 3] = (globalCol + 3 < N) ? B[idxB + 3] : 0.0f;
          } else {
            Bs[vecRow][col] = 0.0f;
            Bs[vecRow][col + 1] = 0.0f;
            Bs[vecRow][col + 2] = 0.0f;
            Bs[vecRow][col + 3] = 0.0f;
          }
        }
      }

      alpaka::syncBlockThreads(acc);

#pragma unroll
      for (int k = 0; k < BK; ++k) {

#pragma unroll
        for (int i = 0; i < TM; ++i) {
          regA[i] = As[ty * TM + i][k];
        }

#pragma unroll
        for (int j = 0; j < TN; ++j) {
          regB[j] = Bs[k][tx * TN + j];
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            accum[i][j] += regA[i] * regB[j];
          }
        }
      }
      alpaka::syncBlockThreads(acc);
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int globalRow = rowStart + ty * TM + i;
        int globalCol = colStart + tx * TN + j;

        if (globalRow < M && globalCol < N) {
          Idx idx = (Idx)globalRow * N + globalCol;
          if (accumulate)
            C[idx] += accum[i][j];
          else
            C[idx] = accum[i][j];
        }
      }
    }
  }
};

/**
 * @brief Launches the Coarsened GEMM Kernel with optimal tuning parameters.
 *
 * Configures the grid and block dimensions based on the compile-time constants.
 * Currently tuned for NVIDIA L4 / Ampere architectures.
 *
 * **Configuration:**
 * - **Block Size (Shared):** 128x128 (BM x BN)
 * - **Thread Work:** 4x4 (TM x TN) per thread.
 * - **Threads per Block:** 1024 (Maximum occupancy).
 *
 * @tparam TQueue The Alpaka queue type.
 * @param queue The queue where the kernel will be enqueued.
 * @param A Pointer to matrix A (device memory).
 * @param B Pointer to matrix B (device memory).
 * @param C Pointer to matrix C (device memory).
 * @param m Number of rows of A and C.
 * @param n Number of columns of B and C.
 * @param k Shared dimension of A and B.
 * @param accumulate If true, performs `C += A*B`; otherwise `C = A*B`.
 */
template <typename TQueue, typename TPtrA, typename TPtrB, typename TPtrC>
void launch_coarsened_kernel(TQueue &queue, TPtrA const A, TPtrB const B,
                             TPtrC C, int m, int n, int k, bool accumulate) {

  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 4;
  const int TN = 4;

  Idx blocksY = (m + BM - 1) / BM;
  Idx blocksX = (n + BN - 1) / BN;

  Idx threadsY = BM / TM;
  Idx threadsX = BN / TN;

  auto workDiv = alpaka::WorkDivMembers<Dim2, Idx>{
      alpaka::Vec<Dim2, Idx>{blocksY, blocksX},
      alpaka::Vec<Dim2, Idx>{threadsY, threadsX},
      alpaka::Vec<Dim2, Idx>{1u, 1u}};

  GemmCoarsenedKernel<BM, BN, BK, TM, TN> kernel;
  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, m, n, k, accumulate);
}

enum class TileConfig { Square32, Rect16x32, Square16, Rect16x64 };

/**
 * @brief Asynchronously uploads a matrix tile from Host to Device.
 *
 * Uses `alpaka::memcpy` to transfer a sub-region of the matrix.
 *
 * @param queue The async queue (stream) to use.
 * @param devHost The host device instance.
 * @param devAcc The accelerator device instance.
 * @param d_dst Destination pointer (Device).
 * @param h_src Source pointer (Host).
 * @param big_N The stride/leading dimension of the full host matrix.
 * @param r_off Row offset in the full matrix.
 * @param c_off Column offset in the full matrix.
 * @param rows Number of rows to copy.
 * @param cols Number of columns to copy.
 */
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

/**
 * @brief Asynchronously downloads a matrix tile from Device to Host.
 *
 * Inverse operation of `upload_tile_alpaka`.
 * @see upload_tile_alpaka
 */
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

/**
 * @brief Manages resources for a single asynchronous execution stream.
 *
 * Encapsulates the Alpaka Queue and the associated Device Buffers required
 * for the Out-of-Core pipeline.
 *
 * @details
 * Each stream needs its own independent buffers (`bufA`, `bufB`, `bufC`) to
 * avoid race conditions when multiple streams operate in parallel (e.g., one
 *          * calculating while another uploads).
 */
struct StreamContext {
  QueueAsync queue;
  alpaka::Buf<DevAcc, float, Dim2, Idx> bufA, bufB, bufC;
  float *d_A, *d_B, *d_C;

  StreamContext(DevAcc const &dev, Idx chunkSize)
      : queue(dev), bufA(alpaka::allocBuf<float, Idx>(
                        dev, alpaka::Vec<Dim2, Idx>{chunkSize, chunkSize})),
        bufB(alpaka::allocBuf<float, Idx>(
            dev, alpaka::Vec<Dim2, Idx>{chunkSize, chunkSize})),
        bufC(alpaka::allocBuf<float, Idx>(
            dev, alpaka::Vec<Dim2, Idx>{chunkSize, chunkSize})) {
    d_A = alpaka::getPtrNative(bufA);
    d_B = alpaka::getPtrNative(bufB);
    d_C = alpaka::getPtrNative(bufC);
  }
};

/**
 * @brief Executes GEMM using an Out-of-Core Pipelined strategy.
 *
 * Handles matrices larger than GPU memory by breaking the operation into chunks
 * and overlapping computation with memory transfers.
 *
 * **Pipeline Strategy:**
 * Uses 3 independent streams to implement a software pipeline:
 * 1. **Compute:** Executes kernel on Tile[i].
 * 2. **Upload:** Transfers Tile[i+1] from Host to Device.
 * 3. **Download:** Transfers Tile[i-1] from Device to Host.
 *
 * @param original_queue The main queue (used to derive device context).
 * @param h_A Host pointer to Matrix A.
 * @param h_B Host pointer to Matrix B.
 * @param h_C Host pointer to Matrix C.
 * @param M Row dimension.
 * @param N Col dimension.
 * @param K Inner dimension.
 * @param config Tile configuration (passed for compatibility).
 */
template <typename TQueue>
void gemm_out_of_core_alpaka(TQueue &original_queue, float const *h_A,
                             float const *h_B, float *h_C, int M, int N, int K,
                             TileConfig config) {
  const int CHUNK_SIZE = 4096;
  auto devAcc = alpaka::getDev(original_queue);
  auto devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
  const int NUM_STREAMS = 3;
  std::vector<std::unique_ptr<StreamContext>> streams;
  for (int i = 0; i < NUM_STREAMS; ++i)
    streams.push_back(std::make_unique<StreamContext>(devAcc, CHUNK_SIZE));

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

        launch_coarsened_kernel(ctx.queue, ctx.d_A, ctx.d_B, ctx.d_C, m_curr,
                                n_curr, k_curr, accumulate);
      }
      download_tile_alpaka(ctx.queue, devHost, devAcc, h_C, ctx.d_C, N, i, j,
                           m_curr, n_curr);
      stream_idx = (stream_idx + 1) % NUM_STREAMS;
    }
  }
  for (auto &ctx : streams)
    alpaka::wait(ctx->queue);
}

template <typename TAcc>
std::size_t estimate_free_memory(alpaka::PlatformCudaRt const &platform,
                                 int devIdx) {
  auto dev = alpaka::getDevByIdx(platform, devIdx);
  auto props = alpaka::getAccDevProps<TAcc>(dev);
  std::size_t total_vram = props.m_globalMemSizeBytes;
  std::size_t system_reserve = 1024ULL * 1024 * 1024;
  return (total_vram > system_reserve) ? total_vram - system_reserve
                                       : total_vram / 2;
}

template <typename TAcc>
TileConfig get_optimal_tile_config(alpaka::PlatformCudaRt const &platform,
                                   int devIdx) {
  return TileConfig::Rect16x64;
}

/**
 * @brief Main Entry Point for the Alpaka GEMM Backend.
 *
 * Decides the execution strategy based on available VRAM and problem size.
 *
 * @details
 * 1. Estimates available VRAM.
 * 2. Calculates required memory for the full matrices.
 * 3. **Branch:**
 * - If (Required < Free VRAM): Launches `launch_coarsened_kernel` directly
 * (In-Core).
 * - If (Required > Free VRAM): Launches `gemm_out_of_core_alpaka`
 * (Out-of-Core/Pipelined).
 *
 * @param queue The Alpaka queue provided by the MPI layer.
 * @param A Matrix A (Input).
 * @param B Matrix B (Input).
 * @param C Matrix C (Output).
 * @param s Structure containing dimensions M, N, K.
 */
template <typename TQueue>
void gemm_alpaka_full_options(TQueue &queue, float const *A, float const *B,
                              float *C, GemmShape s) {

  size_t required = (size_t)s.m * s.k + (size_t)s.k * s.n + (size_t)s.m * s.n;
  required *= sizeof(float);
  size_t estimated_free_vram =
      estimate_free_memory<Acc>(alpaka::PlatformCudaRt{}, 0);
  const size_t SAFETY_MARGIN = 100ULL * 1024 * 1024;
  bool use_batching = (required + SAFETY_MARGIN) >= estimated_free_vram;

  if (!use_batching) {
    launch_coarsened_kernel(queue, A, B, C, s.m, s.n, s.k, false);
    alpaka::wait(queue);
  } else {
    gemm_out_of_core_alpaka(queue, A, B, C, s.m, s.n, s.k,
                            TileConfig::Rect16x64);
  }
}

template void gemm_alpaka_full_options<QueueBlocking>(QueueBlocking &queue,
                                                      float const *A,
                                                      float const *B, float *C,
                                                      GemmShape shape);
} // namespace gemm
