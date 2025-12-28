
#include "gemm/gemm.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace gemm {
    // Tuning constants
    constexpr int BK = 8; // K-dimension unroll factor
    constexpr int TM = 4; // Rows per thread
    constexpr int TN = 4; // Cols per thread

    inline int align_stride(int n) {
        return (n + 3) & ~3;
    }

    void safe_host_register(const void* ptr, size_t size) {
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

        // If pointer is already host memory and managed/pinned, skip registration
        if (err == cudaSuccess && attributes.type == cudaMemoryTypeHost) {
            return;
        }

        // Reset error state if cudaPointerGetAttributes failed
        cudaGetLastError();

        // Try registering
        err = cudaHostRegister((void*)ptr, size, cudaHostRegisterDefault);

        // Ignore if already registered
        if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
            cudaCheck(err, "Host Register");
        }
    }

    static inline void cudaCheck(cudaError_t e, const char *msg) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }

    __global__ void gemm_full_options_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                             int M, int N, int K,
                                             int lda, int ldb, int ldc,
                                             bool accumulation_flag) {

        // Dynamic tile dim
        int BM = blockDim.y * TM;
        int BN = blockDim.x * TN;

        extern __shared__ float shared_mem[]; // Dynamic shared memory declaration
        float* As = shared_mem;
        float* Bs = shared_mem + (BM * BK);

        int bx = blockIdx.x; // Thread x
        int by = blockIdx.y; // Thread y
        int tx = threadIdx.x; // Block x
        int ty = threadIdx.y; // Block y
        int tid = ty * blockDim.x + tx; // Thread id
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

            // Load A
            int totalVecA = (BM * BK) / 4;
            for (int i = tid; i < totalVecA; i += threadsPerBlock) {
                int vecRow = i / (BK / 4);
                int vecCol = i % (BK / 4);
                int col = vecCol * 4;

                int globalRow = rowStart + vecRow;
                int globalCol = tiledK + col;

                if (globalRow < M && globalCol < K) {
                    float4 loaded = *reinterpret_cast<const float4*>(&A[globalRow * lda + globalCol]);
                    // Manual pointer arithmetic for 2D array in 1D dynamic shared memory
                    *reinterpret_cast<float4*>(&As[vecRow * BK + col]) = loaded;
                } else {
                    // Padding with zeros
                    float* ptr = &As[vecRow * BK + col];
                    ptr[0] = 0.0f; ptr[1] = 0.0f; ptr[2] = 0.0f; ptr[3] = 0.0f;
                }
            }

            // Load B
            int totalVecB = (BK * BN) / 4;
            for (int i = tid; i < totalVecB; i += threadsPerBlock) {
                int vecRow = i / (BN / 4);
                int vecCol = i % (BN / 4);
                int col = vecCol * 4;

                int globalRow = tiledK + vecRow;
                int globalCol = colStart + col;

                if (globalRow < K && globalCol < N) {
                    float4 loaded = *reinterpret_cast<const float4*>(&B[globalRow * ldb + globalCol]);
                    *reinterpret_cast<float4*>(&Bs[vecRow * BN + col]) = loaded;
                } else {
                    float* ptr = &Bs[vecRow * BN + col];
                    ptr[0] = 0.0f; ptr[1] = 0.0f; ptr[2] = 0.0f; ptr[3] = 0.0f;
                }
            }

            // SYNC (Wait for load)
            __syncthreads();

            // COMPUTE PHASE (Shared -> Registers -> ALU)
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

            // SYNC (Wait for compute before next load)
            __syncthreads();
        }

        // STORE PHASE (Registers -> Global)
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int globalRow = rowStart + ty * TM + i;
                int globalCol = colStart + tx * TN + j;

                if (globalRow < M && globalCol < N) {
                    size_t idx = (size_t)globalRow * ldc + globalCol;
                    if (accumulation_flag)
                        C[idx] += acc[i][j];
                    else
                        C[idx] = acc[i][j];
                }
            }
        }
    }

    dim3 get_optimal_block_dim(int deviceId) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);

        // Max threads per block is usually 1024, so side is capped at 32
        int max_threads_side = (int)std::sqrt((double)prop.maxThreadsPerBlock);
        //target to hide mem latency: 2 (or 4: to test)
        int target_blocks_per_SM = 2;
        // Calculate shared memory per block to hit that target
        size_t shared_mem_per_block_target = prop.sharedMemPerSM / target_blocks_per_SM;
        // Convert shared memory bytes to max elements for 2 (4) tiles (A and B)
        int max_elements_per_block = (int)(shared_mem_per_block_target / (size_t)256);

        // Final selection
        int TileWidth = std::min(max_threads_side, max_elements_per_block);

        // Alignment to Warp size (32)
        if (TileWidth >= 32) TileWidth = 32;
        else if (TileWidth >= 16) TileWidth = 16;
        else TileWidth = 8;

        return dim3(TileWidth, TileWidth, 1);
    }

    void launch_gemm_kernel(const float *A, const float *B, float *C,
                            int m, int n, int k,
                            int lda, int ldb, int ldc,
                            bool accumulation_flag, cudaStream_t stream) {

        int deviceId;
        cudaGetDevice(&deviceId);

        dim3 block = get_optimal_block_dim(deviceId);
        int BM = block.y * TM;
        int BN = block.x * TN;
        dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, 1);

        //total elements in shared memory
        size_t shared_mem_size_bytes = (BM * BK + BK * BN) * sizeof(float);

        gemm_full_options_kernel<<<grid, block, shared_mem_size_bytes, stream>>>(
                A, B, C, m, n, k, lda, ldb, ldc, accumulation_flag
        );
        cudaCheck(cudaGetLastError(), "kernel launch");
    }

    void upload_tile(float* d_dst, const float* h_src,
                     int host_stride_elements,
                     int device_stride_elements,
                     int r_offset, int c_offset, // Top-left corner of the tile
                     int tile_rows, int tile_cols, cudaStream_t stream) {

        // Device is tightly packed
        size_t dpitch = device_stride_elements * sizeof(float);
        // Host is strided by full width
        size_t spitch = host_stride_elements * sizeof(float);

        const float* src_ptr = h_src + ((size_t)r_offset * host_stride_elements) + c_offset;

        cudaCheck(cudaMemcpy2DAsync(d_dst, dpitch, src_ptr, spitch,
                               tile_cols * sizeof(float), tile_rows,
                               cudaMemcpyHostToDevice, stream), "Tile Upload");
    }

    void download_tile(float* h_dst, const float* d_src,
                       int host_stride_elements,
                       int device_stride_elements,
                       int r_offset, int c_offset,
                       int tile_rows, int tile_cols, cudaStream_t stream) {

        size_t dpitch = host_stride_elements * sizeof(float);
        size_t spitch = device_stride_elements * sizeof(float);

        float* dst_ptr = h_dst + ((size_t)r_offset * host_stride_elements) + c_offset;

        cudaCheck(cudaMemcpy2DAsync(dst_ptr, dpitch, d_src, spitch,
                               tile_cols * sizeof(float), tile_rows,
                               cudaMemcpyDeviceToHost, stream), "Tile Download");
    }

    void gemm_out_of_core(const float* h_A, const float* h_B, float* h_C,
                          int M, int N, int K) {

        // Define chunk size
        const int CHUNK_SIZE = 4096;
        const int N_STREAMS = 3; // Triple buffering

        // Pin host memory
        safe_host_register(h_A, (size_t)M * K * sizeof(float));
        safe_host_register(h_B, (size_t)K * N * sizeof(float));
        safe_host_register(h_C, (size_t)M * N * sizeof(float));

        // Per-stream allocation of buffers on GPU
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

        // Loop over result blocks (M x N)
        for (int i = 0; i < M; i += CHUNK_SIZE) {
            for (int j = 0; j < N; j += CHUNK_SIZE) {

                int s = stream_idx % N_STREAMS;
                cudaCheck(cudaStreamSynchronize(streams[s]), "Stream Sync");
                int m_curr = std::min(CHUNK_SIZE, M - i);
                int n_curr = std::min(CHUNK_SIZE, N - j);

                // Loop over accumulation dimension (K)
                for (int l = 0; l < K; l += CHUNK_SIZE) {
                    int k_curr = std::min(CHUNK_SIZE, K - l);
                    bool accumulation_flag = (l > 0);

                    // Upload chunks
                    upload_tile(d_A[s], h_A, K, CHUNK_SIZE, i, l, m_curr, k_curr, streams[s]);
                    upload_tile(d_B[s], h_B, N, CHUNK_SIZE, l, j, k_curr, n_curr, streams[s]);

                    // Compute
                    launch_gemm_kernel(d_A[s], d_B[s], d_C[s],
                                       m_curr, n_curr, k_curr,
                                       CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE,
                                       accumulation_flag, streams[s]);
                }

                // Download result
                download_tile(h_C, d_C[s], N, CHUNK_SIZE, i, j, m_curr, n_curr, streams[s]);
                stream_idx++;
            }
        }
        cudaDeviceSynchronize();
        for(int s = 0; s < N_STREAMS; ++s) {
            cudaFree(d_A[s]);
            cudaFree(d_B[s]);
            cudaFree(d_C[s]);
            cudaStreamDestroy(streams[s]);
        }
    }

    void gemm_cuda_full_options(const float *A, const float *B, float *C, GemmShape s) {
        int deviceId;
        cudaGetDevice(&deviceId);
        // Check memory requirements
        size_t free_byte, total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);

        // Calculate padded requirements
        int lda = align_stride(s.k);
        int ldb = align_stride(s.n);
        int ldc = align_stride(s.n);

        size_t required = (size_t)s.m * lda + (size_t)s.k * ldb + (size_t)s.m * ldc;
        required *= sizeof(float);

        // Safety: Leave arbitrary 500MB for system/overhead
        size_t margin = 500 * 1024 * 1024;
        if(required + margin < free_byte) {
            // No banching
            float *d_A, *d_B, *d_C;
            cudaCheck(cudaMalloc(&d_A, (size_t)s.m * lda * sizeof(float)), "Malloc A");
            cudaCheck(cudaMalloc(&d_B, (size_t)s.k * ldb * sizeof(float)), "Malloc B");
            cudaCheck(cudaMalloc(&d_C, (size_t)s.m * ldc * sizeof(float)), "Malloc C");

            cudaCheck(cudaMemset(d_A, 0, (size_t)s.m * lda * sizeof(float)), "Zero A");
            cudaCheck(cudaMemset(d_B, 0, (size_t)s.k * ldb * sizeof(float)), "Zero B");

            cudaCheck(cudaMemcpy2D(d_A, lda*sizeof(float), A, s.k*sizeof(float), s.k*sizeof(float), s.m, cudaMemcpyHostToDevice), "Copy A");

            cudaCheck(cudaMemcpy2D(d_B, ldb*sizeof(float), B, s.n*sizeof(float), s.n*sizeof(float), s.k, cudaMemcpyHostToDevice), "Copy B");

            launch_gemm_kernel(d_A, d_B, d_C, s.m, s.n, s.k, lda, ldb, ldc, false, 0);

            cudaCheck(cudaMemcpy2D(C, s.n*sizeof(float), d_C, ldc*sizeof(float), s.n*sizeof(float), s.m, cudaMemcpyDeviceToHost), "Copy C");

            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        }
        else {
            // Batching
            gemm_out_of_core(A, B, C, s.m, s.n, s.k);
        }

        cudaCheck(cudaDeviceSynchronize(), "device sync");
    }

} // namespace gemm