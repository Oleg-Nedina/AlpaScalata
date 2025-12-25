
#include "gemm/gemm.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace gemm {

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
                                             int m, int n, int k, int Ns_offset, int TileWidth, bool accumulation_flag) {

        int row = blockIdx.y * TileWidth + threadIdx.y; // global thread row position
        int col = blockIdx.x * TileWidth + threadIdx.x; // global thread col position

        extern __shared__ float Ms_Ns[]; //dynamic shared memory declaration

        float *M_s = (float *)Ms_Ns; //first part
        float *N_s = (float *)Ms_Ns + Ns_offset; //second part

        float acc = 0.0f;

        for (int t = 0; t < (k + TileWidth - 1) / TileWidth; ++t) {

            // LOAD PHASE

            // Load A into M_s
            if(row >= m || (t * TileWidth + threadIdx.x) >= k) { //boundary check
                M_s[threadIdx.y * TileWidth + threadIdx.x] = 0.0f;
            }
            else{
                M_s[threadIdx.y * TileWidth + threadIdx.x] = A[(size_t)row * k + t * TileWidth + threadIdx.x];
            }

            // Load B into N_s
            if ((t * TileWidth + threadIdx.y) >= k || col >= n) { //boundary check
                N_s[threadIdx.y * TileWidth + threadIdx.x] = 0.0f;
            } else {
                N_s[threadIdx.y * TileWidth + threadIdx.x] = B[(size_t)(t * TileWidth + threadIdx.y) * n + col];
            }

            // SYNC (Wait for load)
            __syncthreads();

            // ACCUMULATION PHASE
            for (int i = 0; i < TileWidth; ++i) {
                acc = fmaf(M_s[threadIdx.y * TileWidth + i], N_s[i * TileWidth + threadIdx.x], acc);
            }

            // SYNC (Wait for compute before next load)
            __syncthreads();
        }

        // Write Result
        if (row < m && col < n) {
            if(accumulation_flag) {
                C[(size_t)row * n + col] += acc;
            }
            else {
                C[(size_t)row * n + col] = acc;
            }
        }
    }

    int get_optimal_tile_width(int deviceId) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);

        //target to hide mem latency: 2 (or 4: to test)
        int target_blocks_per_SM = 2;
        // Calculate Shared Memory per block to hit that target
        size_t shared_mem_per_block_target = prop.sharedMemPerSM / target_blocks_per_SM;
        // Convert shared memory bytes to max elements for 2 (4) tiles (A and B)
        int max_elements_per_block = (int)(shared_mem_per_block_target / (2 * sizeof(float)));
        int side_from_shared_mem = (int)std::sqrt((double)max_elements_per_block);
        // Max threads per block is usually 1024, so side is capped at 32
        int side_from_threads = (int)std::sqrt((double)prop.maxThreadsPerBlock);

        // Final TileWidth Selection
        int TileWidth = std::min(side_from_shared_mem, side_from_threads);

        // Alignment to Warp Size (32)
        if (TileWidth >= 32) TileWidth = 32;
        else if (TileWidth >= 16) TileWidth = 16;
        else TileWidth = 8;

        return TileWidth;
    }

    void launch_gemm_kernel(const float *A, const float *B, float *C,
                            int m, int n, int k,
                            int TileWidth, bool accumulation_flag, cudaStream_t stream) {

        int tile_elements = TileWidth * TileWidth;
        //total elements in shared memory
        size_t shared_mem_size_bytes = 2 * tile_elements * sizeof(float);
        //offset for second part of dynamic shared memory
        int Ns_offset = tile_elements;

        dim3 block(TileWidth, TileWidth, 1);
        dim3 grid((n + TileWidth - 1) / TileWidth, (m + TileWidth - 1) / TileWidth, 1);

        gemm_full_options_kernel<<<grid, block, shared_mem_size_bytes, stream>>>(
                A, B, C, m, n, k, Ns_offset, TileWidth, accumulation_flag
        );
        cudaCheck(cudaGetLastError(), "kernel launch");
    }

    void upload_tile(float* d_dst, const float* h_src,
                     int big_M, int big_N, // Dimensions of the full host matrix
                     int r_offset, int c_offset, // Top-left corner of the tile
                     int tile_rows, int tile_cols, cudaStream_t stream) {

        // Device is tightly packed
        size_t dpitch = tile_cols * sizeof(float);
        // Host is strided by full width
        size_t spitch = big_N * sizeof(float);

        const float* src_ptr = h_src + ((size_t)r_offset * big_N) + c_offset;

        cudaCheck(cudaMemcpy2DAsync(d_dst, dpitch, src_ptr, spitch,
                               tile_cols * sizeof(float), tile_rows,
                               cudaMemcpyHostToDevice, stream), "Tile Upload");
    }

    void download_tile(float* h_dst, const float* d_src,
                       int big_M, int big_N,
                       int r_offset, int c_offset,
                       int tile_rows, int tile_cols, cudaStream_t stream) {

        size_t dpitch = big_N * sizeof(float);      // Host is strided by full width
        size_t spitch = tile_cols * sizeof(float);  // Device is tightly packed

        float* dst_ptr = h_dst + ((size_t)r_offset * big_N) + c_offset;

        cudaCheck(cudaMemcpy2DAsync(dst_ptr, dpitch, d_src, spitch,
                               tile_cols * sizeof(float), tile_rows,
                               cudaMemcpyDeviceToHost, stream), "Tile Download");
    }

    void gemm_out_of_core(const float* h_A, const float* h_B, float* h_C,
                          int M, int N, int K, int TileWidth) {

        // Define Chunk Size
        const int CHUNK_SIZE = 4096;
        const int N_STREAMS = 2; // Double buffering

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
                int m_curr = std::min(CHUNK_SIZE, M - i);
                int n_curr = std::min(CHUNK_SIZE, N - j);

                // Loop over accumulation dimension (K)
                for (int l = 0; l < K; l += CHUNK_SIZE) {
                    int k_curr = std::min(CHUNK_SIZE, K - l);
                    bool accumulation_flag = (l > 0);

                    // Upload Chunks
                    upload_tile(d_A[s], h_A, M, K, i, l, m_curr, k_curr, streams[s]);
                    upload_tile(d_B[s], h_B, K, N, l, j, k_curr, n_curr, streams[s]);

                    // Compute
                    launch_gemm_kernel(d_A[s], d_B[s], d_C[s],
                                       m_curr, n_curr, k_curr,
                                       TileWidth, accumulation_flag, streams[s]);
                }

                // Download Result
                download_tile(h_C, d_C[s], M, N, i, j, m_curr, n_curr, streams[s]);
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
        //get optimal TileWidth
        int TileWidth = get_optimal_tile_width(deviceId);
        //check memory requirements
        size_t free_byte, total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);

        size_t required = (size_t)s.m * s.k + (size_t)s.k * s.n + (size_t)s.m * s.n;
        required *= sizeof(float);

        // Safety: Leave arbitrary 500MB for system/overhead
        size_t margin = 500 * 1024 * 1024;
        if(required + margin < free_byte) {
            //no banching
            float *d_A, *d_B, *d_C;
            cudaCheck(cudaMalloc(&d_A, s.m * s.k * sizeof(float)), "Malloc A");
            cudaCheck(cudaMalloc(&d_B, s.k * s.n * sizeof(float)), "Malloc B");
            cudaCheck(cudaMalloc(&d_C, s.m * s.n * sizeof(float)), "Malloc C");

            cudaCheck(cudaMemcpy(d_A, A, s.m * s.k * sizeof(float), cudaMemcpyHostToDevice), "Copy A");
            cudaCheck(cudaMemcpy(d_B, B, s.k * s.n * sizeof(float), cudaMemcpyHostToDevice), "Copy B");

            launch_gemm_kernel(d_A, d_B, d_C, s.m, s.n, s.k, TileWidth, false, 0);

            cudaCheck(cudaMemcpy(C, d_C, s.m * s.n * sizeof(float), cudaMemcpyDeviceToHost), "Copy C");

            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        }
        else {
            //batching
            gemm_out_of_core(A, B, C, s.m, s.n, s.k, TileWidth);
        }

        cudaCheck(cudaDeviceSynchronize(), "device sync");
    }

} // namespace gemm
