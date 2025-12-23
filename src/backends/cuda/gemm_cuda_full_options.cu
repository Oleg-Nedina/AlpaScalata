
#include "gemm/gemm.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

// tile dim
#define BM 16
#define BN 32
#define BK 64
#define STRIPES (BK / BM)

namespace gemm {

    static inline void cudaCheck(cudaError_t e, const char *msg) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
    }

    __global__ void gemm_full_options_kernel(const float *A, const float *B, float *C,
                                      int m, int n, int k) {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // global thread row position
        int col = blockIdx.x * blockDim.x + threadIdx.x; // global thread col position

        if (row >= m || col >= n)
            return;

        __shared__ float M_s[BM][BK];
        __shared__ float N_s[BK][BN+1];

        float acc = 0.0f;
        int numTiles = (k+BK-1)/BK;

        for (int t = 0; t < numTiles; t++) {

            // LOAD PHASE
            int aColBase = t * BK + threadIdx.x;
            int bRowBase = t * BK + threadIdx.y;

            // Load M Tile
            #pragma unroll
            for(int i = 0; i < (BK/BN); i++){
                int aCol = aColBase + i * BN;
                int sx = threadIdx.x + i * BN;
                M_s[threadIdx.y][sx] = (row < m && aCol < k) ? M[row * k + aCol] : 0.0f;
            }

            // Load N Tile
            #pragma unroll
            for(int i = 0; i < STRIPES; i++){
                int bRow = bRowBase + i * BM;
                int sy = threadIdx.y + i * BM;
                N_s[sy][threadIdx.x] = (bRow < k && col < n) ? N[bRow * n + col] : 0.0f;
            }

            // SYNC (Wait for load)
            __syncthreads();

            // COMPUTE PHASE
            #pragma unroll 64
            for (int kk = 0; kk < BK; ++kk) {
                acc = fmaf(M_s[threadIdx.y][kk], N_s[kk][threadIdx.x], acc);
            }

            // SYNC (Wait for compute before next load)
            __syncthreads();
        }

        // Write Result
        if (row < m && col < n) {
            P[row * n + col] = acc;
        }
    }
    }

    void gemm_cuda_full_options(const float *A, const float *B, float *C, GemmShape s) {
        dim3 block(16, 16, 1);
        dim3 grid((s.n + block.x - 1) / block.x, (s.m + block.y - 1) / block.y, 1);
        gemm_naive_kernel<<<grid, block>>>(A, B, C, s.m, s.n, s.k);
        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "device sync");
    }

} // namespace gemm
