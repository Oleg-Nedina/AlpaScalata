#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>

namespace gemm {

using Idx  = std::size_t;
using Dim2 = alpaka::DimInt<2>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using Acc         = alpaka::AccGpuCudaRt<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCudaRt;
#else
using Acc         = alpaka::AccCpuSerial<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCpu;
#endif

// --- DEFINIZIONE DELLA STRUCT PRIMA DELLA FUNZIONE ---
struct GemmNaiveKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                float const* A,
                                float const* B,
                                float* C,
                                int M, int N, int K) const {
    auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

    int row = (int)idx[0];
    int col = (int)idx[1];

    if (row < M && col < N) {
      float sum = 0.0f;
      for (int kk = 0; kk < K; ++kk) {
        sum += A[row * K + kk] * B[kk * N + col];
      }
      C[row * N + col] = sum;
    }
  }
};
// -----------------------------------------------------

void gemm_alpaka_naive(float const* A, float const* B, float* C, GemmShape shape) {
  // Nota: Idealmente dovresti passare la queue come argomento qui, 
  // invece di ricrearla, come discusso nella risposta precedente.
  auto platform = PlatformAcc{};
  auto devAcc   = alpaka::getDevByIdx(platform, 0u); 

  alpaka::Queue<decltype(devAcc), alpaka::Blocking> queue(devAcc);

  constexpr Idx TX = 16;
  constexpr Idx TY = 16;

  Idx blocksY = (Idx)((shape.m + (int)TY - 1) / (int)TY);
  Idx blocksX = (Idx)((shape.n + (int)TX - 1) / (int)TX);

  auto const gridThreadExtent  = alpaka::Vec<Dim2, Idx>{blocksY, blocksX};
  auto const blockThreadExtent = alpaka::Vec<Dim2, Idx>{TY, TX};
  auto const elemExtent        = alpaka::Vec<Dim2, Idx>{1u, 1u};

  alpaka::WorkDivMembers<Dim2, Idx> workDiv(gridThreadExtent, blockThreadExtent, elemExtent);

  // ORA QUESTO FUNZIONERA' PERCHE' LA STRUCT E' DEFINITA SOPRA
  GemmNaiveKernel kernel;
  
  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, shape.m, shape.n, shape.k);
  alpaka::wait(queue);
}

} // namespace gemm
