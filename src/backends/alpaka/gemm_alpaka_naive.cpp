// src/backends/alpaka/gemm_alpaka_naive.cpp
#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>
#include <cstring>
#include <stdexcept>

namespace gemm {

gemm_alpaka_naive.cpp #if defined (ALPAKA_ACC_GPU_CUDA_ENABLED) &&
    defined(__CUDACC__) using TagAcc = alpaka::TagGpuCudaRt;
#else
using TagAcgemm_alpaka_naive.cppc = alpaka::TagCpuSerial;
#endif

using TagHost = alpaka::TagCpuSerial;

struct GemmNaiveKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, const float *A, const float *B,
                                float *C, int m, int n, int k) const {
    // 2D index: x = col, y = row
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

void gemm_alpaka_naive(const float *Ah, const float *Bh, float *Ch,
                       GemmShape s) {
  using Dim = alpaka::DimInt<2>;
  using Idx = std::size_t;

  // device scelto (GPU se disponibile in questa TU, altrimenti CPU)
  auto devAcc = alpaka::getDevByIdx<TagAcc>(0);
  auto queue = alpaka::Queue<TagAcc, alpaka::Blocking>{devAcc};

  // host device per buffer host alpaka
  auto devHost = alpaka::getDevByIdx<TagHost>(0);

  // extent 2D: (x = n, y = m)
  alpaka::Vec<Dim, Idx> extent{static_cast<Idx>(s.n), static_cast<Idx>(s.m)};

  // --- device buffers ---
  auto bufA_d = alpaka::allocBuf<float, Idx>(devAcc, extent);
  auto bufB_d = alpaka::allocBuf<float, Idx>(devAcc, extent);
  auto bufC_d = alpaka::allocBuf<float, Idx>(devAcc, extent);

  // --- host staging buffers (sempre host) ---
  auto bufA_h = alpaka::allocBuf<float, Idx>(devHost, extent);
  auto bufB_h = alpaka::allocBuf<float, Idx>(devHost, extent);
  auto bufC_h = alpaka::allocBuf<float, Idx>(devHost, extent);

  // riempi host staging (qui std::memcpy è OK)
  std::memcpy(alpaka::getPtrNative(bufA_h), Ah,
              sizeof(float) * (std::size_t)s.m * s.k);
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh,
              sizeof(float) * (std::size_t)s.k * s.n);

  // copia H->D (qui DEVI usare alpaka::memcpy)
  alpaka::memcpy(queue, bufA_d, bufA_h, extent);
  alpaka::memcpy(queue, bufB_d, bufB_h, extent);

  // workdiv naive: threads 16x16, griglia ceil(n/16) x ceil(m/16)
  alpaka::Vec<Dim, Idx> threads{16u, 16u};
  alpaka::Vec<Dim, Idx> elems{static_cast<Idx>(s.n), static_cast<Idx>(s.m)};
  alpaka::KernelCfg<alpaka::Acc<TagAcc, Dim, Idx>> cfg = {elems, threads};

  // NB: in alcuni Alpaka setup serve Acc esplicito; se ti dà rogne, dimmelo e
  // lo adapto
  using Acc = alpaka::Acc<TagAcc, Dim, Idx>;
  GemmNaiveKernel kernel;
  auto workDiv = alpaka::getValidWorkDiv(cfg, devAcc, kernel);

  alpaka::exec<TagAcc>(queue, workDiv, kernel, alpaka::getPtrNative(bufA_d),
                       alpaka::getPtrNative(bufB_d),
                       alpaka::getPtrNative(bufC_d), s.m, s.n, s.k);

  alpaka::wait(queue);

  // copia D->H staging
  alpaka::memcpy(queue, bufC_h, bufC_d, extent);
  alpaka::wait(queue);

  // copia in output host “vero”
  std::memcpy(Ch, alpaka::getPtrNative(bufC_h),
              sizeof(float) * (std::size_t)s.m * s.n);
}

} // namespace gemm
