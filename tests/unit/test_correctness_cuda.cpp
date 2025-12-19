
#include "gemm/gemm.hpp"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

static void ref_cpu(const float *A, const float *B, float *C, int m, int n,
                    int k) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk)
        acc = std::fmaf(A[i * k + kk], B[kk * n + j], acc);
      C[i * n + j] = acc;
    }
}

static void check(const float *got, const float *ref, int size) {
  const float atol = 1e-4f, rtol = 1e-4f;
  for (int i = 0; i < size; ++i) {
    float diff = std::fabs(got[i] - ref[i]);
    float tol = atol + rtol * std::fabs(ref[i]);
    if (diff > tol) {
      std::printf("Mismatch at %d: got=%f ref=%f diff=%e tol=%e\n", i, got[i],
                  ref[i], diff, tol);
      std::exit(1);
    }
  }
}

int main() {
  int m = 64, k = 128, n = 96;
  gemm::GemmShape s{m, n, k};

  std::vector<float> A(m * k), B(k * n), C(m * n), Cref(m * n);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &x : A)
    x = dist(rng);
  for (auto &x : B)
    x = dist(rng);

  float *Ad, *Bd, *Cd;
  cudaMalloc(&Ad, A.size() * sizeof(float));
  cudaMalloc(&Bd, B.size() * sizeof(float));
  cudaMalloc(&Cd, C.size() * sizeof(float));

  cudaMemcpy(Ad, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Bd, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);

  gemm::gemm_cuda_naive(Ad, Bd, Cd, s);

  cudaMemcpy(C.data(), Cd, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

  ref_cpu(A.data(), B.data(), Cref.data(), m, n, k);
  check(C.data(), Cref.data(), m * n);

  std::puts("OK");
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);
  return 0;
}
