// src/backends/alpaka/gemm_alpaka_naive.cpp
#include "gemm/gemm.hpp"
#include <external/alpaka/alpaka.hpp>
#include <stdexcept>

namespace gemm {

namespace {
using Dim = alpaka::DimInt<2>;
using Idx = std::size_t;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
#else
using Acc = alpaka::AccCpuSerial<Dim, Idx>;
#endif
} // namespace

struct GemmNaiveKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, const float *A, const float *B,
                                float *C, int m, int n, int k) const {
    auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    int col = static_cast<int>(idx[0]);
    int row = static_cast<int>(idx[1]);

    if (row >= m || col >= n)
      return;

    float accv = 0.0f;
    for (int kk = 0; kk < k; ++kk)
      accv = alpaka::math::fma(acc, A[row * k + kk], B[kk * n + col], accv);

    C[row * n + col] = accv;
  }
};

void gemm_alpaka_naive(const float *A, const float *B, float *C, GemmShape s) {
  if (!A || !B || !C)
    throw std::invalid_argument("gemm_alpaka_naive: null ptr");
  if (s.m <= 0 || s.n <= 0 || s.k <= 0)
    throw std::invalid_argument("gemm_alpaka_naive: bad shape");

  auto platform = alpaka::Platform<Acc>{};
  auto devAcc = alpaka::getDevByIdx(platform, 0u);
  auto queue = alpaka::Queue<decltype(devAcc), alpaka::Blocking>{devAcc};

  alpaka::Vec<Dim, Idx> threads{16u, 16u};
  alpaka::Vec<Dim, Idx> elems{static_cast<Idx>(s.n), static_cast<Idx>(s.m)};

  GemmNaiveKernel kernel{};
  alpaka::KernelCfg<Acc> cfg{elems, threads};

  // IMPORTANT: pass kernel args also to getValidWorkDiv (your Alpaka version
  // needs it)
  auto workDiv =
      alpaka::getValidWorkDiv(cfg, devAcc, kernel, A, B, C, s.m, s.n, s.k);

  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, s.m, s.n, s.k);
  alpaka::wait(queue);
}

} // namespace gemm
