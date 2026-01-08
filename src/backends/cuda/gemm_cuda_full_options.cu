#include "gemm/gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/**
 * @brief Macro for checking CUDA API call errors.
 *
 * Checks the result of a CUDA API call. If an error occurs, it prints the
 * error message and the provided description to stderr, then exits the program.
 *
 * @param err The error code returned by the CUDA function.
 * @param msg A custom message to display if an error occurs.
 */
#define cudaCheck(err, msg)                                                    \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace gemm {

// Tuning constants
/** @brief The K-dimension unroll factor (block size in the K dimension). */
    constexpr int BK = 16;
/** @brief Number of rows computed per thread (Register blocking factor M). */
    constexpr int TM = 8;
/** @brief Number of columns computed per thread (Register blocking factor N). */
    constexpr int TN = 8;

/**
 * @brief Aligns a dimension size to the nearest multiple of 4.
 * * Used to ensure that strides are compatible with vectorized float4 operations.
 *
 * @param n The input dimension size.
 * @return The size aligned to the next multiple of 4.
 */
    inline int align_stride(int n) { return (n + 3) & ~3; }

/**
 * @brief Safely registers host memory for pinned access if not already registered.
 *
 * Checks if the provided pointer is already registered with the CUDA runtime.
 * If not, it registers the memory to enable faster (pinned) transfers and
 * asynchronous copies.
 *
 * @param ptr Pointer to the host memory buffer.
 * @param size Size of the memory buffer in bytes.
 */
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

/**
 * @brief CUDA Kernel performing General Matrix Multiplication (GEMM).
 *
 * This kernel implements a tiled GEMM approach utilizing Shared Memory for
 * data reuse and Register Blocking (TM x TN) to increase instruction-level
 * parallelism. It supports vectorized loading (float4) and accumulation.
 *
 * @param A Pointer to matrix A data (device memory).
 * @param B Pointer to matrix B data (device memory).
 * @param C Pointer to matrix C data (device memory).
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in A and rows in B.
 * @param lda Leading dimension (stride) of matrix A.
 * @param ldb Leading dimension (stride) of matrix B.
 * @param ldc Leading dimension (stride) of matrix C.
 * @param accumulation_flag If true, results are added to the existing values in C.
 * If false, C is overwritten.
 */
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

                if (globalRow < M && globalCol + 3 < K) {
                    size_t idxA = (size_t)globalRow * lda + globalCol;
                    float4 loaded = *reinterpret_cast<const float4 *>(&A[idxA]);
                    *reinterpret_cast<float4 *>(&As[vecRow * BK + col]) = loaded;
                } else {
                    float *ptr = &As[vecRow * BK + col];
                    if (globalRow < M) {
                        size_t idxA = (size_t)globalRow * lda + globalCol;
                        ptr[0] = (globalCol + 0 < K) ? A[idxA + 0] : 0.0f;
                        ptr[1] = (globalCol + 1 < K) ? A[idxA + 1] : 0.0f;
                        ptr[2] = (globalCol + 2 < K) ? A[idxA + 2] : 0.0f;
                        ptr[3] = (globalCol + 3 < K) ? A[idxA + 3] : 0.0f;
                    } else {
                        ptr[0] = 0.0f;
                        ptr[1] = 0.0f;
                        ptr[2] = 0.0f;
                        ptr[3] = 0.0f;
                    }
                }
            }

            int totalVecB = (BK * BN) / 4;
            for (int i = tid; i < totalVecB; i += threadsPerBlock) {
                int vecRow = i / (BN / 4);
                int vecCol = i % (BN / 4);
                int col = vecCol * 4;

                int globalRow = tiledK + vecRow;
                int globalCol = colStart + col;

                if (globalRow < K && globalCol + 3 < N) {
                    size_t idxB = (size_t)globalRow * ldb + globalCol;
                    float4 loaded = *reinterpret_cast<const float4 *>(&B[idxB]);
                    *reinterpret_cast<float4 *>(&Bs[vecRow * BN + col]) = loaded;
                } else {
                    float *ptr = &Bs[vecRow * BN + col];
                    if (globalRow < K) {
                        size_t idxB = (size_t)globalRow * ldb + globalCol;
                        ptr[0] = (globalCol + 0 < N) ? B[idxB + 0] : 0.0f;
                        ptr[1] = (globalCol + 1 < N) ? B[idxB + 1] : 0.0f;
                        ptr[2] = (globalCol + 2 < N) ? B[idxB + 2] : 0.0f;
                        ptr[3] = (globalCol + 3 < N) ? B[idxB + 3] : 0.0f;
                    } else {
                        ptr[0] = 0.0f;
                        ptr[1] = 0.0f;
                        ptr[2] = 0.0f;
                        ptr[3] = 0.0f;
                    }
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

        // STORE PHASE
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            int globalRow = rowStart + ty * TM + i;

            // If the row is out of matrix bounds, skip
            if (globalRow >= M)
                continue;

            for (int j = 0; j < TN; j += 4) {
                int globalCol = colStart + tx * TN + j;

                // Check for vectorized writing (float4)
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
                    // Scalar fallback
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

/**
 * @brief Helper to configure grid dimensions and launch the GEMM kernel.
 *
 * Configures the CUDA grid and block dimensions based on the compile-time
 * constants TM and TN. Allocates necessary shared memory and launches the kernel.
 *
 * @param A Pointer to matrix A.
 * @param B Pointer to matrix B.
 * @param C Pointer to matrix C.
 * @param m Matrix M dimension.
 * @param n Matrix N dimension.
 * @param k Matrix K dimension.
 * @param lda Leading dimension of A.
 * @param ldb Leading dimension of B.
 * @param ldc Leading dimension of C.
 * @param accumulation_flag Whether to accumulate into C.
 * @param stream The CUDA stream to execute the kernel in.
 */
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

/**
 * @brief Asynchronously uploads a 2D matrix tile from Host to Device.
 *
 * Performs a cudaMemcpy2DAsync to copy a sub-region (tile) of a matrix
 * from host memory to a specific device buffer.
 *
 * @param d_dst Destination pointer on the device.
 * @param h_src Source pointer on the host.
 * @param host_stride_elements Stride (in elements) of the host matrix row.
 * @param device_stride_elements Stride (in elements) of the device buffer.
 * @param r_offset Row offset in the source matrix.
 * @param c_offset Column offset in the source matrix.
 * @param tile_rows Number of rows to copy.
 * @param tile_cols Number of columns to copy.
 * @param stream CUDA stream for async operation.
 */
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

/**
 * @brief Asynchronously downloads a 2D matrix tile from Device to Host.
 *
 * Performs a cudaMemcpy2DAsync to copy results from a device buffer back
 * to the corresponding sub-region of the host matrix.
 *
 * @param h_dst Destination pointer on the host.
 * @param d_src Source pointer on the device.
 * @param host_stride_elements Stride (in elements) of the host matrix row.
 * @param device_stride_elements Stride (in elements) of the device buffer.
 * @param r_offset Row offset in the destination matrix.
 * @param c_offset Column offset in the destination matrix.
 * @param tile_rows Number of rows to copy.
 * @param tile_cols Number of columns to copy.
 * @param stream CUDA stream for async operation.
 */
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

/**
 * @brief Performs GEMM for matrices larger than GPU memory (Out-of-Core).
 *
 * Implements a streaming out-of-core GEMM algorithm. It breaks the large
 * host matrices into smaller chunks (tiles) that fit into GPU memory.
 * Multiple CUDA streams are used to overlap data transfer (uploading operands,
 * downloading results) with kernel execution.
 *
 * @param h_A Pointer to matrix A on Host.
 * @param h_B Pointer to matrix B on Host.
 * @param h_C Pointer to matrix C on Host.
 * @param M Number of rows.
 * @param N Number of columns.
 * @param K Inner dimension.
 */
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

/**
 * @brief Main entry point for the optimized CUDA GEMM.
 *
 * This function determines the optimal execution strategy based on the
 * memory location of the input pointers and available GPU memory.
 * * Strategy:
 * 1. If inputs are on the device: Run the kernel directly.
 * 2. If inputs are on the host and fit in GPU RAM: Allocate device buffers,
 * copy everything, compute, and copy back.
 * 3. If inputs are on the host and are too large: Fallback to Out-of-Core execution.
 *
 * @param A Pointer to matrix A (Host or Device).
 * @param B Pointer to matrix B (Host or Device).
 * @param C Pointer to matrix C (Host or Device).
 * @param s Shape structure containing dimensions (m, n, k).
 */
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
