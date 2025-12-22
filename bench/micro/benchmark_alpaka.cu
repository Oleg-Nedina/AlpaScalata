#define GEMM_ENABLE_ALPAKA 
#include "gemm/gemm.hpp" // Include l'header corretto
#include <alpaka/alpaka.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace alpaka_bench {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;

// Configurazione Acceleratore (Deve matchare con quella in
// gemm_alpaka_naive.cpp)
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCudaRt;
#else
using Acc = alpaka::AccCpuSerial<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCpu;
#endif

using PlatformHost = alpaka::PlatformCpu;

struct Ctx {
  decltype(alpaka::getDevByIdx(PlatformAcc{}, 0u)) devAcc;
  alpaka::Queue<decltype(devAcc), alpaka::Blocking> queue;
  decltype(alpaka::getDevByIdx(PlatformHost{}, 0u)) devHost;

  Ctx()
      : devAcc(alpaka::getDevByIdx(PlatformAcc{}, 0u)), queue(devAcc),
        devHost(alpaka::getDevByIdx(PlatformHost{}, 0u)) {}
};

// ... Helpers (fill_random, get_arg) ...
// (Copia qui le funzioni fill_random e get_arg dalle risposte precedenti)
static inline void fill_random(std::vector<float> &v, unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : v)
    x = dist(rng);
}

int get_arg(int argc, char **argv, std::string flag, int default_val) {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == flag) {
      return std::stoi(argv[i + 1]);
    }
  }
  return default_val;
}

void run_naive_device(Ctx &ctx, const float *Ah, const float *Bh, float *Ch,
                      int M, int N, int K, float *elapsed_ms_out) {
  // ... (Allocazione memoria identica a prima) ...
  using Dim1 = alpaka::DimInt<1>;
  const int elemsA = M * K;
  const int elemsB = K * N;
  const int elemsC = M * N;

  auto extentA = alpaka::Vec<Dim1, Idx>{(Idx)elemsA};
  auto extentB = alpaka::Vec<Dim1, Idx>{(Idx)elemsB};
  auto extentC = alpaka::Vec<Dim1, Idx>{(Idx)elemsC};

  auto bufA_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentA);
  auto bufB_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentB);
  auto bufC_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentC);
  auto bufA_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentA);
  auto bufB_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentB);
  auto bufC_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentC);

  std::memcpy(alpaka::getPtrNative(bufA_h), Ah, elemsA * sizeof(float));
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh, elemsB * sizeof(float));

  alpaka::memcpy(ctx.queue, bufA_d, bufA_h, extentA);
  alpaka::memcpy(ctx.queue, bufB_d, bufB_h, extentB);
  alpaka::wait(ctx.queue);

  auto *Ad = alpaka::getPtrNative(bufA_d);
  auto *Bd = alpaka::getPtrNative(bufB_d);
  auto *Cd = alpaka::getPtrNative(bufC_d);

  gemm::GemmShape shape{M, N, K};

  auto t0 = std::chrono::high_resolution_clock::now();

  // CHIAMATA ALLA FUNZIONE ESTERNA
  gemm::gemm_alpaka_naive(ctx.queue, Ad, Bd, Cd, shape);

  auto t1 = std::chrono::high_resolution_clock::now();
  if (elapsed_ms_out)
    *elapsed_ms_out =
        (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

  alpaka::memcpy(ctx.queue, bufC_h, bufC_d, extentC);
  alpaka::wait(ctx.queue);
  std::memcpy(Ch, alpaka::getPtrNative(bufC_h), elemsC * sizeof(float));
}

} // namespace alpaka_bench

int main(int argc, char **argv) {
  int size = alpaka_bench::get_arg(argc, argv, "--check", 64);
  const int M = size;
  const int N = size;
  const int K = size;

  std::cout << "Alpaka Naive (Separate Files) | Size: " << M << "x" << N << "x"
            << K << "\n";

  std::vector<float> A(M * K), B(K * N), C(M * N);
  alpaka_bench::fill_random(A, 1);
  alpaka_bench::fill_random(B, 2);

  alpaka_bench::Ctx ctx;
  float ms = 0.0f;
  alpaka_bench::run_naive_device(ctx, A.data(), B.data(), C.data(), M, N, K,
                                 &ms);

  std::cout << "Time: " << ms << " ms\nC[0]: " << C[0] << "\n";
  return 0;
}
