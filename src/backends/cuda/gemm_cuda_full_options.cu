#include "gemm/gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define cudaCheck(err, msg)                                                    \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace gemm {
// Tuning constants
constexpr int BK = 16; // K-dimension unroll factor
constexpr int TM = 8;  // Rows per thread
constexpr int TN = 8;  // Cols per thread

inline int align_stride(int n) { return (n + 3) & ~3; }

void safe_host_register(const void *ptr, size_t size) {
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  if (err == cudaSuccess && attributes.type == cudaMemoryTypeHost) {
    return;
  }
  cudaGetLastError();
  err = cudaHostRegister((void *)ptr, size, cudaHostRegisterDefault);
  if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
    cudaCheck(err, "Host Register");
  }
}

__global__ void gemm_full_options_kernel(const float *__restrict__ A,
                                         const float *__restrict__ B,
                                         float *__restrict__ C, int M, int N,
                                         int K, int lda, int ldb, int ldc,
                                         bool accumulation_flag) {

  // Dynamic tile dim
  int BM = blockDim.y * TM;
  int BN = blockDim.x * TN;

  extern __shared__ float shared_mem[];
  float *As = shared_mem;
  float *Bs = shared_mem + (BM * BK);

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  int threadsPerBlock = blockDim.x * blockDim.y;

  int rowStart = by * BM;
  int colStart = bx * BN;

  float acc[TM][TN] = {0.0f};

  float regA[TM];
  float regB[TN];
  int numTiles = (K + BK - 1) / BK;

  for (int t = 0; t < numTiles; ++t) {

    int tiledK = t * BK;

    // LOAD PHASE
    int totalVecA = (BM * BK) / 4;
    for (int i = tid; i < totalVecA; i += threadsPerBlock) {
      int vecRow = i / (BK / 4);
      int vecCol = i % (BK / 4);
      int col = vecCol * 4;

      int globalRow = rowStart + vecRow;
      int globalCol = tiledK + col;

      if (globalRow < M && globalCol < K) {
        size_t idxA = (size_t)globalRow * lda + globalCol;
        float4 loaded = *reinterpret_cast<const float4 *>(&A[idxA]);
        *reinterpret_cast<float4 *>(&As[vecRow * BK + col]) = loaded;
      } else {
        float *ptr = &As[vecRow * BK + col];
        ptr[0] = 0.0f;
        ptr[1] = 0.0f;
        ptr[2] = 0.0f;
        ptr[3] = 0.0f;
      }
    }

    int totalVecB = (BK * BN) / 4;
    for (int i = tid; i < totalVecB; i += threadsPerBlock) {
      int vecRow = i / (BN / 4);
      int vecCol = i % (BN / 4);
      int col = vecCol * 4;

      int globalRow = tiledK + vecRow;
      int globalCol = colStart + col;

      if (globalRow < K && globalCol < N) {
        size_t idxB = (size_t)globalRow * ldb + globalCol;
        float4 loaded = *reinterpret_cast<const float4 *>(&B[idxB]);
        *reinterpret_cast<float4 *>(&Bs[vecRow * BN + col]) = loaded;
      } else {
        float *ptr = &Bs[vecRow * BN + col];
        ptr[0] = 0.0f;
        ptr[1] = 0.0f;
        ptr[2] = 0.0f;
        ptr[3] = 0.0f;
      }
    }

    __syncthreads();

// COMPUTE PHASE
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        regA[i] = As[(ty * TM + i) * BK + k];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        regB[j] = Bs[k * BN + (tx * TN + j)];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += regA[i] * regB[j];
        }
      }
    }
    __syncthreads();
  }

  // STORE PHASE (CORRETTA)
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int globalRow = rowStart + ty * TM + i;

    // FIX: Se la riga è fuori dalla matrice, saltiamo tutto immediatamente!
    if (globalRow >= M)
      continue;

    for (int j = 0; j < TN; j += 4) {
      int globalCol = colStart + tx * TN + j;

      // Check per scrittura vettorizzata (float4)
      if (globalCol + 3 < N) {
        float4 vecC;
        vecC.x = acc[i][j + 0];
        vecC.y = acc[i][j + 1];
        vecC.z = acc[i][j + 2];
        vecC.w = acc[i][j + 3];

        size_t idx = (size_t)globalRow * ldc + globalCol;

        if (accumulation_flag) {
          float4 loaded = *reinterpret_cast<float4 *>(&C[idx]);
          vecC.x += loaded.x;
          vecC.y += loaded.y;
          vecC.z += loaded.z;
          vecC.w += loaded.w;
        }

        *reinterpret_cast<float4 *>(&C[idx]) = vecC;
      } else {
        // Fallback scalare sicuro
        for (int k = 0; k < 4; ++k) {
          if (globalCol + k < N) {
            float val = acc[i][j + k];
            if (accumulation_flag)
              val += C[(size_t)globalRow * ldc + globalCol + k];
            C[(size_t)globalRow * ldc + globalCol + k] = val;
          }
        }
      }
    }
  }
}

void launch_gemm_kernel(const float *A, const float *B, float *C, int m, int n,
                        int k, int lda, int ldb, int ldc,
                        bool accumulation_flag, cudaStream_t stream) {

  // Block 16x16 with TM=8, TN=8 covers 128x128 elements per block
  dim3 block(16, 16, 1);

  int BM = block.y * TM;
  int BN = block.x * TN;
  dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, 1);

  size_t shared_mem_size_bytes = (BM * BK + BK * BN) * sizeof(float);

  gemm_full_options_kernel<<<grid, block, shared_mem_size_bytes, stream>>>(
      A, B, C, m, n, k, lda, ldb, ldc, accumulation_flag);
  cudaCheck(cudaGetLastError(), "kernel launch");
}

void upload_tile(float *d_dst, const float *h_src, int host_stride_elements,
                 int device_stride_elements, int r_offset, int c_offset,
                 int tile_rows, int tile_cols, cudaStream_t stream) {
  size_t dpitch = device_stride_elements * sizeof(float);
  size_t spitch = host_stride_elements * sizeof(float);
  const float *src_ptr =
      h_src + ((size_t)r_offset * host_stride_elements) + c_offset;
  cudaCheck(cudaMemcpy2DAsync(d_dst, dpitch, src_ptr, spitch,
                              tile_cols * sizeof(float), tile_rows,
                              cudaMemcpyHostToDevice, stream),
            "Tile Upload");
}

void download_tile(float *h_dst, const float *d_src, int host_stride_elements,
                   int device_stride_elements, int r_offset, int c_offset,
                   int tile_rows, int tile_cols, cudaStream_t stream) {
  size_t dpitch = host_stride_elements * sizeof(float);
  size_t spitch = device_stride_elements * sizeof(float);
  float *dst_ptr = h_dst + ((size_t)r_offset * host_stride_elements) + c_offset;
  cudaCheck(cudaMemcpy2DAsync(dst_ptr, dpitch, d_src, spitch,
                              tile_cols * sizeof(float), tile_rows,
                              cudaMemcpyDeviceToHost, stream),
            "Tile Download");
}

void gemm_out_of_core(const float *h_A, const float *h_B, float *h_C, int M,
                      int N, int K) {
  const int CHUNK_SIZE = 4096;
  const int N_STREAMS = 3;

  safe_host_register(h_A, (size_t)M * K * sizeof(float));
  safe_host_register(h_B, (size_t)K * N * sizeof(float));
  safe_host_register(h_C, (size_t)M * N * sizeof(float));

  cudaStream_t streams[N_STREAMS];
  float *d_A[N_STREAMS], *d_B[N_STREAMS], *d_C[N_STREAMS];
  size_t buf_size = (size_t)CHUNK_SIZE * CHUNK_SIZE * sizeof(float);
  for (int s = 0; s < N_STREAMS; ++s) {
    cudaCheck(cudaStreamCreate(&streams[s]), "Stream Create");
    cudaCheck(cudaMalloc(&d_A[s], buf_size), "Malloc A");
    cudaCheck(cudaMalloc(&d_B[s], buf_size), "Malloc B");
    cudaCheck(cudaMalloc(&d_C[s], buf_size), "Malloc C");
  }
  int stream_idx = 0;

  for (int i = 0; i < M; i += CHUNK_SIZE) {
    for (int j = 0; j < N; j += CHUNK_SIZE) {
      int s = stream_idx % N_STREAMS;
      cudaCheck(cudaStreamSynchronize(streams[s]), "Stream Sync");
      int m_curr = std::min(CHUNK_SIZE, M - i);
      int n_curr = std::min(CHUNK_SIZE, N - j);

      for (int l = 0; l < K; l += CHUNK_SIZE) {
        int k_curr = std::min(CHUNK_SIZE, K - l);
        bool accumulation_flag = (l > 0);
        upload_tile(d_A[s], h_A, K, CHUNK_SIZE, i, l, m_curr, k_curr,
                    streams[s]);
        upload_tile(d_B[s], h_B, N, CHUNK_SIZE, l, j, k_curr, n_curr,
                    streams[s]);
        launch_gemm_kernel(d_A[s], d_B[s], d_C[s], m_curr, n_curr, k_curr,
                           CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE,
                           accumulation_flag, streams[s]);
      }
      download_tile(h_C, d_C[s], N, CHUNK_SIZE, i, j, m_curr, n_curr,
                    streams[s]);
      stream_idx++;
    }
  }
  cudaDeviceSynchronize();
  for (int s = 0; s < N_STREAMS; ++s) {
    cudaFree(d_A[s]);
    cudaFree(d_B[s]);
    cudaFree(d_C[s]);
    cudaStreamDestroy(streams[s]);
  }
}

void gemm_cuda_full_options(const float *A, const float *B, float *C,
                            GemmShape s) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, A);

  if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
    launch_gemm_kernel(A, B, C, s.m, s.n, s.k, s.k, s.n, s.n, false, 0);
    cudaCheck(cudaDeviceSynchronize(), "device sync direct");
    return;
  }

  cudaGetLastError();
  int deviceId;
  cudaGetDevice(&deviceId);
  size_t free_byte, total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);

  int lda = align_stride(s.k);
  int ldb = align_stride(s.n);
  int ldc = align_stride(s.n);

  size_t required = (size_t)s.m * lda + (size_t)s.k * ldb + (size_t)s.m * ldc;
  required *= sizeof(float);
  size_t margin = 500 * 1024 * 1024;

  if (required + margin < free_byte) {
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, (size_t)s.m * lda * sizeof(float)), "Malloc A");
    cudaCheck(cudaMalloc(&d_B, (size_t)s.k * ldb * sizeof(float)), "Malloc B");
    cudaCheck(cudaMalloc(&d_C, (size_t)s.m * ldc * sizeof(float)), "Malloc C");

    cudaCheck(cudaMemcpy2D(d_A, lda * sizeof(float), A, s.k * sizeof(float),
                           s.k * sizeof(float), s.m, cudaMemcpyHostToDevice),
              "Copy A");
    cudaCheck(cudaMemcpy2D(d_B, ldb * sizeof(float), B, s.n * sizeof(float),
                           s.n * sizeof(float), s.k, cudaMemcpyHostToDevice),
              "Copy B");

    launch_gemm_kernel(d_A, d_B, d_C, s.m, s.n, s.k, lda, ldb, ldc, false, 0);

    cudaCheck(cudaMemcpy2D(C, s.n * sizeof(float), d_C, ldc * sizeof(float),
                           s.n * sizeof(float), s.m, cudaMemcpyDeviceToHost),
              "Copy C");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  } else {
    gemm_out_of_core(A, B, C, s.m, s.n, s.k);
  }
  cudaCheck(cudaDeviceSynchronize(), "device sync");
}
} // namespace gemm
