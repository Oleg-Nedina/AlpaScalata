
#include "gemm/gemm.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace gemm {

    static inline void cudaCheck(cudaError_t e, const char *msg) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }

    __global__ void gemm_full_options_kernel(const float *A, const float *B, float *C,
                                             int m, int n, int k, int Ns_offset, int TileWidth) {

        int row = blockIdx.y * TileWidth + threadIdx.y; // global thread row position
        int col = blockIdx.x * TileWidth + threadIdx.x; // global thread col position

        extern __shared__ float Ms_Ns[]; //dynamic shared memory declaration

        float *M_s = (float *)Ms_Ns; //first part
        float *N_s = (float *)Ms_Ns + Ns_offset; //second part

        float acc = 0.0f;

        for (int t = 0; t < (k + TileWidth - 1) / TileWidth; ++t) {

            // LOAD PHASE

            // Load A into M_s
            if(row >= m || (t * TileWidth + threadIdx.x) >= k){ //boundary check
                M_s[threadIdx.y * TileWidth + threadIdx.x] = 0.0f;
            }
            else{
                M_s[threadIdx.y * TileWidth + threadIdx.x] = A[row * k + t * TileWidth + threadIdx.x];
            }

            // Load B into N_s
            if ((t * TileWidth + threadIdx.y) >= k || col >= n) { //boundary check
                Nds[threadIdx.y * TileWidth + threadIdx.x] = 0.0f;
            } else {
                Nds[threadIdx.y * TileWidth + threadIdx.x] = B[(t * TileWidth + threadIdx.y) * n + col];
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
            C[row * n + col] = acc;
        }
    }

    void gemm_cuda_full_options(const float *A, const float *B, float *C, GemmShape s) {
        // Determine optimal tile size based on hardware
        // For many GPUs, 32x32 is a standard starting point
        int TileWidth = 32;
        int tile_elements = TileWidth * TileWidth;
        size_t shared_mem_size_bytes = 2 * tile_elements * sizeof(float); //total elements in shared memory
        int Ns_offset = tile_elements;
        dim3 block(TileWidth, TileWidth, 1);
        dim3 grid((s.n + TileWidth - 1) / TileWidth, (s.m + TileWidth - 1) / TileWidth, 1);
        gemm_full_options_kernel<<<grid, block, shared_mem_size_bytes>>>(A, B, C, s.m, s.n, s.k, opt_size/2, TileWidth);
        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "device sync");
    }

} // namespace gemm
