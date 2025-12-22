#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gemm/gemm.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka_bench {

// -------------------- Alpaka types --------------------
using Idx = std::size_t;
using Dim1 = alpaka::DimInt<1>;
using Dim2 = alpaka::DimInt<2>;

// Scegli acceleratore: se hai CUDA abilitato in Alpaka, usa GPU; altrimenti
// CPU.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCudaRt;
#else
using Acc = alpaka::AccCpuSerial<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCpu;
#endif

using PlatformHost = alpaka::PlatformCpu;

// -------------------- Context --------------------
struct Ctx {
  decltype(alpaka::getDevByIdx(PlatformAcc{}, 0u)) devAcc;
  alpaka::Queue<decltype(devAcc), alpaka::Blocking> queue;

  decltype(alpaka::getDevByIdx(PlatformHost{}, 0u)) devHost;

  Ctx()
      : devAcc(alpaka::getDevByIdx(PlatformAcc{}, 0u)), queue(devAcc),
        devHost(alpaka::getDevByIdx(PlatformHost{}, 0u)) {}
};

// -------------------- Helpers --------------------
static inline void fill_random(std::vector<float> &v, unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : v)
    x = dist(rng);
}

// -------------------- Run --------------------
void run_naive_device(Ctx &ctx, const float *Ah, const float *Bh, float *Ch,
                      int M, int N, int K, float *elapsed_ms_out) {
  const int elemsA = M * K;
  const int elemsB = K * N;
  const int elemsC = M * N;

  const std::size_t bytesA = (std::size_t)elemsA * sizeof(float);
  const std::size_t bytesB = (std::size_t)elemsB * sizeof(float);
  const std::size_t bytesC = (std::size_t)elemsC * sizeof(float);

  auto extentA = alpaka::Vec<Dim1, Idx>{(Idx)elemsA};
  auto extentB = alpaka::Vec<Dim1, Idx>{(Idx)elemsB};
  auto extentC = alpaka::Vec<Dim1, Idx>{(Idx)elemsC};

  // device buffers
  auto bufA_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentA);
  auto bufB_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentB);
  auto bufC_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentC);

  // host buffers (alpaka host device)
  auto bufA_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentA);
  auto bufB_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentB);
  auto bufC_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentC);

  std::memcpy(alpaka::getPtrNative(bufA_h), Ah, bytesA);
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh, bytesB);

  alpaka::memcpy(ctx.queue, bufA_d, bufA_h, extentA);
  alpaka::memcpy(ctx.queue, bufB_d, bufB_h, extentB);
  alpaka::wait(ctx.queue);

  auto *Ad = alpaka::getPtrNative(bufA_d);
  auto *Bd = alpaka::getPtrNative(bufB_d);
  auto *Cd = alpaka::getPtrNative(bufC_d);

  gemm::GemmShape shape{M, N, K};

  auto t0 = std::chrono::high_resolution_clock::now();
  gemm::gemm_alpaka_naive(Ad, Bd, Cd, shape);
  auto t1 = std::chrono::high_resolution_clock::now();

  // gemm_alpaka_naive fa già wait internamente, quindi Cd è pronto
  std::chrono::duration<double, std::milli> ms = t1 - t0;
  if (elapsed_ms_out)
    *elapsed_ms_out = (float)ms.count();

  alpaka::memcpy(ctx.queue, bufC_h, bufC_d, extentC);
  alpaka::wait(ctx.queue);
  std::memcpy(Ch, alpaka::getPtrNative(bufC_h), bytesC);
}

} // namespace alpaka_bench

// ... includes ...

// ... Helper Parsing ...
int get_arg(int argc, char **argv, std::string flag, int default_val) {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == flag) {
      return std::stoi(argv[i + 1]);
    }
  }
  return default_val;
}

// ... run_naive_device ...
void run_naive_device(Ctx &ctx, ..., int M, int N, int K, ...) {
  // ... allocazioni uguali a prima ...

  auto t0 = std::chrono::high_resolution_clock::now();

  // PASSAGGIO FONDAMENTALE: Passiamo la ctx.queue!
  gemm::gemm_alpaka_naive(ctx.queue, Ad, Bd, Cd, shape);

  auto t1 = std::chrono::high_resolution_clock::now();
  // ... resto uguale ...
}

int main(int argc, char **argv) {
  // Parsing basilare
  int size = get_arg(argc, argv, "--check", 64);

  // Se vuoi supportare M, N, K diversi, estendi la logica
  const int M = size;
  const int N = size;
  const int K = size;

  std::cout << "Running with size: " << M << "x" << N << "x" << K << "\n";

  std::vector<float> A((size_t)M * K);
  // ... e così via, usa M/N/K invece dei numeri fissi
