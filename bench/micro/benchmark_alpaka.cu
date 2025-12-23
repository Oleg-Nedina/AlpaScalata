#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace alpaka_bench {

// -----------------------------------------------------------------------------
// 1. Configurazione Alpaka (GPU Forzata)
// -----------------------------------------------------------------------------
using Idx = std::size_t;
using Dim1 = alpaka::DimInt<1>;
using Dim2 = alpaka::DimInt<2>;

// GPU CUDA
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCudaRt;
using PlatformHost = alpaka::PlatformCpu;

struct Ctx {
  decltype(alpaka::getDevByIdx(PlatformAcc{}, 0u)) devAcc;
  alpaka::Queue<decltype(devAcc), alpaka::Blocking> queue;
  decltype(alpaka::getDevByIdx(PlatformHost{}, 0u)) devHost;

  Ctx()
      : devAcc(alpaka::getDevByIdx(PlatformAcc{}, 0u)), queue(devAcc),
        devHost(alpaka::getDevByIdx(PlatformHost{}, 0u)) {}
};

// -----------------------------------------------------------------------------
// 2. Helpers
// -----------------------------------------------------------------------------
struct RunCfg {
  int warmup = 5;
  int reps = 20;
  int minN = 256;
  int maxN = 4096;
  int step = 256;
  unsigned seed = 123;
  std::string solver = "alpaka_naive";
  std::string precision = "float";
};

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

static std::unordered_map<std::string, std::string>
parse_prm(const std::string &path) {
  std::unordered_map<std::string, std::string> m;
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "Cannot open config file %s\n", path.c_str());
    std::exit(4);
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    auto pos = line.find('=');
    if (pos == std::string::npos)
      continue;
    std::string key = lower(line.substr(0, pos));
    std::string val = line.substr(pos + 1);
    key.erase(key.find_last_not_of(" \t") + 1);
    val.erase(0, val.find_first_not_of(" \t"));
    val.erase(val.find_last_not_of(" \t") + 1);
    m[key] = val;
  }
  return m;
}

static inline void fill_random(std::vector<float> &v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &x : v)
    x = dist(rng);
}

// Implementazione GEMM semplice su CPU per riferimento
void cpu_gemm_ref(const float *A, const float *B, float *C, int M, int N,
                  int K) {
  // Azzera C
  std::fill(C, C + (M * N), 0.0f);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

static void check_close(const std::vector<float> &got,
                        const std::vector<float> &ref, float atol = 1e-3f,
                        float rtol = 1e-3f) {
  for (size_t i = 0; i < got.size(); ++i) {
    float diff = std::fabs(got[i] - ref[i]);
    float tol = atol + rtol * std::fabs(ref[i]);
    if (diff > tol) {
      std::fprintf(stderr, "Mismatch at %zu: got=%f ref=%f diff=%e tol=%e\n", i,
                   got[i], ref[i], diff, tol);
      std::exit(10); // Esci con errore se fallisce
    }
  }
}

void verify_correctness(Ctx &ctx, int N, const RunCfg &cfg) {
  std::cout << "Running correctness check: Alpaka(GPU) vs CPU_Ref (N=" << N
            << ")... ";
  std::flush(std::cout);

  int M = N;
  int K = N;
  size_t size_elems = (size_t)N * N;
  size_t size_bytes = size_elems * sizeof(float);

  // Host Memory
  std::vector<float> Ah(size_elems), Bh(size_elems);
  std::vector<float> Ch_alpaka(size_elems); // Risultato dalla GPU
  std::vector<float> Ch_ref(size_elems);    // Risultato dalla CPU

  fill_random(Ah, cfg.seed);
  fill_random(Bh, cfg.seed + 1);

  // 1. Calcolo Alpaka (GPU)
  auto extent = alpaka::Vec<Dim1, Idx>{(Idx)size_elems};

  // Allocazione Buffer
  auto bufA_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);
  auto bufB_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);
  auto bufC_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);

  auto bufA_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);
  auto bufB_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);
  auto bufC_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);

  // Prepare dati pinned
  std::memcpy(alpaka::getPtrNative(bufA_h), Ah.data(), size_bytes);
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh.data(), size_bytes);

  // H2D
  alpaka::memcpy(ctx.queue, bufA_d, bufA_h, extent);
  alpaka::memcpy(ctx.queue, bufB_d, bufB_h, extent);
  alpaka::wait(ctx.queue);

  // Esegui Kernel
  gemm::GemmShape shape{M, N, K};
  auto *pA = alpaka::getPtrNative(bufA_d);
  auto *pB = alpaka::getPtrNative(bufB_d);
  auto *pC = alpaka::getPtrNative(bufC_d);

  if (cfg.solver == "alpaka_naive") {
    gemm::gemm_alpaka_naive(ctx.queue, pA, pB, pC, shape);
  } else if (cfg.solver == "alpaka_tiled") {
    gemm::gemm_alpaka_tiled(ctx.queue, pA, pB, pC, shape);
  } else {
    std::cerr << "\nERROR: Unknown solver '" << cfg.solver << "'\n";
    std::exit(1);
  }
  // D2H
  alpaka::memcpy(ctx.queue, bufC_h, bufC_d, extent);
  alpaka::wait(ctx.queue);

  // Copia risultato in vector standard
  std::memcpy(Ch_alpaka.data(), alpaka::getPtrNative(bufC_h), size_bytes);

  // 2. Calcolo CPU (Gold Standard)
  cpu_gemm_ref(Ah.data(), Bh.data(), Ch_ref.data(), M, N, K);

  // 3. Confronto
  check_close(Ch_alpaka, Ch_ref);
  std::cout << "CHECK_OK\n";
}

// -----------------------------------------------------------------------------
// 4. Benchmark Function
// -----------------------------------------------------------------------------
static float bench_alpaka_once_ms(Ctx &ctx, int N, const RunCfg &cfg) {
  const int M = N;
  const int K = N;
  const int elems = N * N;

  std::vector<float> Ah(elems), Bh(elems);
  fill_random(Ah, cfg.seed);
  fill_random(Bh, cfg.seed + 1);

  auto extent = alpaka::Vec<Dim1, Idx>{(Idx)elems};
  auto bufA_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);
  auto bufB_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);
  auto bufC_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);
  auto bufA_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);
  auto bufB_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);

  std::memcpy(alpaka::getPtrNative(bufA_h), Ah.data(), elems * sizeof(float));
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh.data(), elems * sizeof(float));

  alpaka::memcpy(ctx.queue, bufA_d, bufA_h, extent);
  alpaka::memcpy(ctx.queue, bufB_d, bufB_h, extent);
  alpaka::wait(ctx.queue);

  auto *Ad = alpaka::getPtrNative(bufA_d);
  auto *Bd = alpaka::getPtrNative(bufB_d);
  auto *Cd = alpaka::getPtrNative(bufC_d);
  gemm::GemmShape shape{M, N, K};

  // Helper lambda per lanciare il solver corretto
  auto run_solver = [&]() {
    if (cfg.solver == "alpaka_naive") {
      gemm::gemm_alpaka_naive(ctx.queue, Ad, Bd, Cd, shape);
    } else if (cfg.solver == "alpaka_tiled") {
      gemm::gemm_alpaka_tiled(ctx.queue, Ad, Bd, Cd, shape);
    } else {
      std::cerr << "ERROR:
          Unknown solver '" << cfg.solver
                << "'\n";
      std::exit(1);
    }
  };

  //
  Warmup for (int i = 0; i < cfg.warmup; ++i) { run_solver(); }
  alpaka::wait(ctx.queue);

  // Timing
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < cfg.reps; ++i) {
    run_solver();
  }
  alpaka::wait(ctx.queue); // Assicura che la GPU abbia finito
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  return (float)(total_ms / cfg.reps);
}

// -----------------------------------------------------------------------------
// 5. Loop di Benchmark
// -----------------------------------------------------------------------------
static int tb_alpaka_loop(Ctx &ctx, const RunCfg &cfg) {
  std::cout << "Solver,Precision,Size,Time_ms,GFLOPs\n";

  for (int N = cfg.minN; N <= cfg.maxN; N += cfg.step) {
    float t_ms = bench_alpaka_once_ms(ctx, N, cfg);
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = flops / (t_ms * 1e-3) / 1e9;
    std::printf("%s,%s,%d,%.6f,%.6f\n", cfg.solver.c_str(),
                cfg.precision.c_str(), N, t_ms, gflops);
    std::fflush(stdout);
  }
  return 0;
}

} // namespace alpaka_bench

// -----------------------------------------------------------------------------
// 6. Main
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
  std::string config_file;
  int checkN = -1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *opt) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", opt);
        std::exit(3);
      }
    };
    if (a == "--config") {
      need("--config");
      config_file = argv[++i];
    } else if (a == "--check") {
      need("--check");
      checkN = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
      return 3;
    }
  }

  if (config_file.empty()) {
    std::fprintf(stderr, "Usage: %s --config <file.prm> [--check N]\n",
                 argv[0]);
    return 1;
  }

  auto prm = alpaka_bench::parse_prm(config_file);
  auto require_key = [&](const char *k) -> std::string {
    std::string key = alpaka_bench::lower(std::string(k));
    if (prm.find(key) == prm.end()) {
      std::fprintf(stderr, "ERROR: missing key '%s' in %s\n", k,
                   config_file.c_str());
      std::exit(4);
    }
    return prm[key];
  };

  alpaka_bench::RunCfg cfg;
  cfg.solver = "alpaka_" + alpaka_bench::lower(require_key("solver"));
  cfg.precision = alpaka_bench::lower(require_key("precision"));
  cfg.minN = std::stoi(require_key("min"));
  cfg.maxN = std::stoi(require_key("max"));
  cfg.step = std::stoi(require_key("step"));
  cfg.reps = std::stoi(require_key("reps"));
  cfg.warmup = std::stoi(require_key("warmup"));
  cfg.seed = (unsigned)std::stoul(require_key("seed"));

  alpaka_bench::Ctx ctx;
  std::cerr << "Accelerator: " << alpaka::getAccName<alpaka_bench::Acc>()
            << "\n";

  // MODALITA' CHECK: Verifica Numerica Reale
  if (checkN > 0) {
    alpaka_bench::verify_correctness(ctx, checkN, cfg);
    return 0;
  }

  return alpaka_bench::tb_alpaka_loop(ctx, cfg);
}
