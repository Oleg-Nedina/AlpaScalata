
#include "gemm/gemm.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace gemm {

    static inline void cudaCheck(cudaError_t e, const char *msg) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }

    __global__ void gemm_full_options_kernel(const float *A, const float *B, float *C,
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
                M_s[threadIdx.y * TileWidth + threadIdx.x] = A[row * k + t * TileWidth + threadIdx.x];
            }

            // Load B into N_s
            if ((t * TileWidth + threadIdx.y) >= k || col >= n) { //boundary check
                N_s[threadIdx.y * TileWidth + threadIdx.x] = 0.0f;
            } else {
                N_s[threadIdx.y * TileWidth + threadIdx.x] = B[(t * TileWidth + threadIdx.y) * n + col];
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
                C[row * n + col] = acc;
            }
            else {
                C[row * n + col] = acc;
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
                            int TileWidth, bool accumulation_flag) {

        int tile_elements = TileWidth * TileWidth;
        //total elements in shared memory
        size_t shared_mem_size_bytes = 2 * tile_elements * sizeof(float);
        //offset for second part of dynamic shared memory
        int Ns_offset = tile_elements;

        dim3 block(TileWidth, TileWidth, 1);
        dim3 grid((n + TileWidth - 1) / TileWidth, (m + TileWidth - 1) / TileWidth, 1);

        gemm_full_options_kernel<<<grid, block, shared_mem_size_bytes>>>(
                A, B, C, m, n, k, Ns_offset, TileWidth, accumulate
        );

        cudaCheck(cudaGetLastError(), "kernel launch");
    }

    void gemm_cuda_full_options(const float *A, const float *B, float *C, GemmShape s) {
        int deviceId;
        cudaGetDevice(&deviceId);
        int TileWidth = get_optimal_tile_width(deviceId);
        launch_gemm_kernel(A, B, C, s.m, s.n, s.k, TileWidth, false);
        cudaCheck(cudaDeviceSynchronize(), "device sync");
    }

} // namespace gemm
