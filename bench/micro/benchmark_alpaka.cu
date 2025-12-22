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

// Scegli acceleratore: se hai CUDA abilitato in Alpaka, usa GPU; altrimenti CPU.
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
  // Device acceleratore (GPU o CPU seriale)
  decltype(alpaka::getDevByIdx(PlatformAcc{}, 0u)) devAcc;
  // Coda di esecuzione associata al device
  alpaka::Queue<decltype(devAcc), alpaka::Blocking> queue;
  
  // Device Host (CPU per allocazione memoria host)
  decltype(alpaka::getDevByIdx(PlatformHost{}, 0u)) devHost;

  Ctx()
      : devAcc(alpaka::getDevByIdx(PlatformAcc{}, 0u)),
        queue(devAcc),
        devHost(alpaka::getDevByIdx(PlatformHost{}, 0u)) {}
};

// -------------------- Helpers --------------------
static inline void fill_random(std::vector<float>& v, unsigned seed = 1) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(rng);
}

// Funzione helper per parsare gli argomenti da riga di comando
int get_arg(int argc, char** argv, std::string flag, int default_val) {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == flag) {
      return std::stoi(argv[i + 1]);
    }
  }
  return default_val;
}

// -------------------- Run --------------------
void run_naive_device(Ctx& ctx,
                      const float* Ah,
                      const float* Bh,
                      float* Ch,
                      int M, int N, int K,
                      float* elapsed_ms_out) {
  const int elemsA = M * K;
  const int elemsB = K * N;
  const int elemsC = M * N;

  const std::size_t bytesA = (std::size_t)elemsA * sizeof(float);
  const std::size_t bytesB = (std::size_t)elemsB * sizeof(float);
  const std::size_t bytesC = (std::size_t)elemsC * sizeof(float);

  auto extentA = alpaka::Vec<Dim1, Idx>{(Idx)elemsA};
  auto extentB = alpaka::Vec<Dim1, Idx>{(Idx)elemsB};
  auto extentC = alpaka::Vec<Dim1, Idx>{(Idx)elemsC};

  // 1. Allocazione buffer su device
  auto bufA_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentA);
  auto bufB_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentB);
  auto bufC_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extentC);

  // 2. Allocazione buffer su host (pinned memory di Alpaka)
  auto bufA_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentA);
  auto bufB_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentB);
  auto bufC_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extentC);

  // 3. Copia dati input nei buffer host di Alpaka
  std::memcpy(alpaka::getPtrNative(bufA_h), Ah, bytesA);
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh, bytesB);

  // 4. Copia Host -> Device
  alpaka::memcpy(ctx.queue, bufA_d, bufA_h, extentA);
  alpaka::memcpy(ctx.queue, bufB_d, bufB_h, extentB);
  // Aspettiamo che la copia finisca prima di lanciare il kernel
  alpaka::wait(ctx.queue);

  // Otteniamo i pointer raw per il kernel
  auto* Ad = alpaka::getPtrNative(bufA_d);
  auto* Bd = alpaka::getPtrNative(bufB_d);
  auto* Cd = alpaka::getPtrNative(bufC_d);

  gemm::GemmShape shape{M, N, K};

  // 5. Esecuzione e Timing
  auto t0 = std::chrono::high_resolution_clock::now();
  
  // *** FIX: Passiamo ctx.queue alla funzione! ***
  gemm::gemm_alpaka_naive(ctx.queue, Ad, Bd, Cd, shape);
  
  // Nota: gemm_alpaka_naive fa già alpaka::wait(queue) internamente
  auto t1 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> ms = t1 - t0;
  if (elapsed_ms_out) *elapsed_ms_out = (float)ms.count();

  // 6. Copia Device -> Host
  alpaka::memcpy(ctx.queue, bufC_h, bufC_d, extentC);
  alpaka::wait(ctx.queue);
  
  // Copia finale nell'array di output standard C++
  std::memcpy(Ch, alpaka::getPtrNative(bufC_h), bytesC);
}

} // namespace alpaka_bench

// -------------------- main --------------------
int main(int argc, char** argv) {
  // Lettura dimensione da riga di comando (default 64)
  int size = alpaka_bench::get_arg(argc, argv, "--check", 64);

  const int M = size;
  const int N = size;
  const int K = size;

  std::cout << "Running Alpaka Naive Benchmark with size: " 
            << M << "x" << N << "x" << K << "\n";

  std::vector<float> A((size_t)M * K);
  std::vector<float> B((size_t)K * N);
  std::vector<float> C((size_t)M * N, 0.0f);

  alpaka_bench::fill_random(A, 1);
  alpaka_bench::fill_random(B, 2);

  alpaka_bench::Ctx ctx;
  float ms = 0.0f;
  
  alpaka_bench::run_naive_device(ctx, A.data(), B.data(), C.data(), M, N, K, &ms);

  std::cout << "alpaka naive done in " << ms << " ms\n";
  std::cout << "C[0] = " << C[0] << "\n";
  
  return 0;
}
