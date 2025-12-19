
#include "gemm/gemm.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace gemm {

static inline void cudaCheck(cudaError_t e, const char *msg) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}

__global__ void gemm_naive_kernel(const float *A, const float *B, float *C,
                                  int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n)
    return;

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc = fmaf(A[row * k + kk], B[kk * n + col], acc);
  }
  C[row * n + col] = acc;
}

void gemm_cuda_naive(const float *A, const float *B, float *C, GemmShape s) {
  dim3 block(16, 16, 1);
  dim3 grid((s.n + block.x - 1) / block.x, (s.m + block.y - 1) / block.y, 1);
  gemm_naive_kernel<<<grid, block>>>(A, B, C, s.m, s.n, s.k);
  cudaCheck(cudaGetLastError(), "kernel launch");
  cudaCheck(cudaDeviceSynchronize(), "device sync");
}

} // namespace gemm
