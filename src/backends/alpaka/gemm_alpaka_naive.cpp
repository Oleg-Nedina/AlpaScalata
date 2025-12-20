#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>
#include <cstring>

namespace gemm {

struct GemmNaiveKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, float const *A, float const *B,
                                float *C, int m, int n, int k) const {
    // idx: [x, y]
    auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    int col = static_cast<int>(idx[0u]);
    int row = static_cast<int>(idx[1u]);
    if (row >= m || col >= n)
      return;

    float accv = 0.0f;
    for (int kk = 0; kk < k; ++kk)
      accv += A[row * k + kk] * B[kk * n + col];

    C[row * n + col] = accv;
  }
};

void gemm_alpaka_naive(const float *Ah, const float *Bh, float *Ch,
                       GemmShape s) {
  using Dim = alpaka::DimInt<2>;
  using Idx = std::size_t;

  // per ora CPU serial (locale)
  using Tag = alpaka::TagCpuSerial;
  using Acc = alpaka::TagToAcc<Tag, Dim, Idx>;
  using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

  auto const platformAcc = alpaka::Platform<Acc>{};
  auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
  Queue q(devAcc);

  // extent: [x, y] = [cols, rows]
  auto const extA =
      alpaka::Vec<Dim, Idx>{static_cast<Idx>(s.k), static_cast<Idx>(s.m)};
  auto const extB =
      alpaka::Vec<Dim, Idx>{static_cast<Idx>(s.n), static_cast<Idx>(s.k)};
  auto const extC =
      alpaka::Vec<Dim, Idx>{static_cast<Idx>(s.n), static_cast<Idx>(s.m)};

  auto bufA = alpaka::allocBuf<float, Idx>(devAcc, extA);
  auto bufB = alpaka::allocBuf<float, Idx>(devAcc, extB);
  auto bufC = alpaka::allocBuf<float, Idx>(devAcc, extC);

  std::memcpy(alpaka::getPtrNative(bufA), Ah,
              sizeof(float) * (std::size_t)s.m * s.k);
  std::memcpy(alpaka::getPtrNative(bufB), Bh,
              sizeof(float) * (std::size_t)s.k * s.n);

  // workdiv: grid threads = [n, m]
  auto const elementsPerThread = alpaka::Vec<Dim, Idx>::all(1u);
  auto const elementsPerGrid =
      alpaka::Vec<Dim, Idx>{static_cast<Idx>(s.n), static_cast<Idx>(s.m)};

  GemmNaiveKernel kernel;
  alpaka::KernelCfg<Acc> const cfg = {elementsPerGrid, elementsPerThread};
  auto const workDiv = alpaka::getValidWorkDiv(cfg, devAcc, kernel);

  alpaka::exec<Tag>(q, workDiv, kernel, alpaka::getPtrNative(bufA),
                    alpaka::getPtrNative(bufB), alpaka::getPtrNative(bufC), s.m,
                    s.n, s.k);
  alpaka::wait(q);

  std::memcpy(Ch, alpaka::getPtrNative(bufC),
              sizeof(float) * (std::size_t)s.m * s.n);
}

} // namespace gemm
