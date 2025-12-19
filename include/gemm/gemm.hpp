#pragma once
#include <cstddef>

namespace gemm {

struct GemmShape {
  int m, n, k;
};

// row-major: A[m,k], B[k,n], C[m,n]
void gemm_cuda_naive(const float *A, const float *B, float *C, GemmShape s);

} // namespace gemm
