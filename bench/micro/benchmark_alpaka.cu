#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <algorithm>
#include <alpaka/alpaka.hpp>
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

/**
 * @file benchmark_alpaka.cu
 * @brief Micro-benchmark harness for Alpaka GEMM implementations.
 *
 * This executable is designed to measure the performance (Time and GFLOPS)
 * of the various Alpaka GEMM solvers (Naive, Tiling, Full Options) on a single
 * GPU.
 *
 * **Key Features:**
 * - **Configurable:** Reads run parameters (size, repetitions, solver type)
 * from a `.prm` file.
 * - **Correctness Check:** Can verify GPU results against a CPU reference
 * implementation (`--check N`).
 * - **Performance Profiling:** Executes a loop of GEMM calls with warmup to
 * measure stable execution time.
 * - **CSV Output:** Prints results in CSV format for easy plotting.
 */
namespace alpaka_bench {

using Idx = std::size_t;
using Dim1 = alpaka::DimInt<1>;
using Dim2 = alpaka::DimInt<2>;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using PlatformAcc = alpaka::PlatformCudaRt;
using PlatformHost = alpaka::PlatformCpu;

/**
 * @brief Holds the Alpaka runtime context.
 *
 * Encapsulates the Device, Queue, and Host Platform needed to execute Alpaka
 * kernels. Initializing this once avoids overhead during the benchmark loop.
 */
struct Ctx {
  decltype(alpaka::getDevByIdx(PlatformAcc{}, 0u)) devAcc;
  alpaka::Queue<decltype(devAcc), alpaka::Blocking> queue;
  decltype(alpaka::getDevByIdx(PlatformHost{}, 0u)) devHost;

  Ctx()
      : devAcc(alpaka::getDevByIdx(PlatformAcc{}, 0u)), queue(devAcc),
        devHost(alpaka::getDevByIdx(PlatformHost{}, 0u)) {}
};

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

/**
 * @brief Reference GEMM implementation on CPU.
 *
 * A simple, single-threaded triple-loop matrix multiplication:
 * \f$ C_{i,j} = \sum_{k} A_{i,k} \times B_{k,j} \f$
 *
 * Used as the "Golden Standard" to verify the correctness of the GPU
 * implementations.
 *
 * @param A Input Matrix A.
 * @param B Input Matrix B.
 * @param C Output Matrix C (will be overwritten).
 * @param M Number of rows.
 * @param N Number of columns.
 * @param K Shared dimension.
 */
void cpu_gemm_ref(const float *A, const float *B, float *C, int M, int N,
                  int K) {
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
      std::exit(10);
    }
  }
}

/**
 * @brief Verifies the correctness of the selected Alpaka solver.
 *
 * 1. Generates random matrices A and B on the Host.
 * 2. Uploads them to the GPU.
 * 3. Executes the selected GPU solver (Naive, Tiling, or Full).
 * 4. Downloads the result C to the Host.
 * 5. Computes the reference result using `cpu_gemm_ref`.
 * 6. Compares the two results element-wise with a tolerance.
 *
 * Prints "CHECK_OK" if successful, or terminates with an error code if mismatch
 * found.
 *
 * @param ctx The Alpaka context.
 * @param N The matrix dimension (Square NxN).
 * @param cfg The run configuration (solver type, seed).
 */
void verify_correctness(Ctx &ctx, int N, const RunCfg &cfg) {
  std::cout << "Running correctness check: Alpaka(GPU) vs CPU_Ref (N=" << N
            << ")... ";
  std::flush(std::cout);

  int M = N;
  int K = N;
  size_t size_elems = (size_t)N * N;
  size_t size_bytes = size_elems * sizeof(float);

  std::vector<float> Ah(size_elems), Bh(size_elems);
  std::vector<float> Ch_alpaka(size_elems);
  std::vector<float> Ch_ref(size_elems);

  fill_random(Ah, cfg.seed);
  fill_random(Bh, cfg.seed + 1);

  auto extent = alpaka::Vec<Dim1, Idx>{(Idx)size_elems};

  auto bufA_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);
  auto bufB_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);
  auto bufC_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent);

  auto bufA_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);
  auto bufB_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);
  auto bufC_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent);

  std::memcpy(alpaka::getPtrNative(bufA_h), Ah.data(), size_bytes);
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh.data(), size_bytes);

  alpaka::memcpy(ctx.queue, bufA_d, bufA_h, extent);
  alpaka::memcpy(ctx.queue, bufB_d, bufB_h, extent);
  alpaka::wait(ctx.queue);

  gemm::GemmShape shape{M, N, K};
  auto *pA = alpaka::getPtrNative(bufA_d);
  auto *pB = alpaka::getPtrNative(bufB_d);
  auto *pC = alpaka::getPtrNative(bufC_d);

  if (cfg.solver == "alpaka_naive") {
    gemm::gemm_alpaka_naive(ctx.queue, pA, pB, pC, shape);
  } else if (cfg.solver == "alpaka_tiling") {
    gemm::gemm_alpaka_tiled(ctx.queue, pA, pB, pC, shape);
  } else if (cfg.solver == "alpaka_full") {
    gemm::gemm_alpaka_full_options(ctx.queue, pA, pB, pC, shape);
  } else {
    std::cerr << "\nERROR: Unknown solver '" << cfg.solver << "'\n";
    std::exit(1);
  }
  alpaka::memcpy(ctx.queue, bufC_h, bufC_d, extent);
  alpaka::wait(ctx.queue);

  std::memcpy(Ch_alpaka.data(), alpaka::getPtrNative(bufC_h), size_bytes);

  cpu_gemm_ref(Ah.data(), Bh.data(), Ch_ref.data(), M, N, K);

  check_close(Ch_alpaka, Ch_ref);
  std::cout << "CHECK_OK\n";
}

/**
 * @brief Executes a single benchmark run for a specific size N.
 *
 * 1. Allocates and initializes memory.
 * 2. Performs **Warmup** runs (results discarded) to wake up the GPU.
 * 3. Measures the execution time of `cfg.reps` repetitions using `std::chrono`.
 * 4. Ensures GPU synchronization (`alpaka::wait`) before and after the timer.
 *
 * @param ctx The Alpaka context.
 * @param N The matrix dimension.
 * @param cfg Configuration parameters (reps, warmup, solver).
 * @return Average execution time in milliseconds.
 */
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

  auto run_solver = [&]() {
    if (cfg.solver == "alpaka_naive") {
      gemm::gemm_alpaka_naive(ctx.queue, Ad, Bd, Cd, shape);
    } else if (cfg.solver == "alpaka_tiling") {
      gemm::gemm_alpaka_tiled(ctx.queue, Ad, Bd, Cd, shape);
    } else if (cfg.solver == "alpaka_full") {
      gemm::gemm_alpaka_full_options(ctx.queue, Ad, Bd, Cd, shape);
    } else {
      std::cerr << "ERROR: Unknown solver '" << cfg.solver << "'\n";
      std::exit(1);
    }
  };

  for (int i = 0; i < cfg.warmup; ++i) {
    run_solver();
  }
  alpaka::wait(ctx.queue);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < cfg.reps; ++i) {
    run_solver();
  }
  alpaka::wait(ctx.queue);
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  return (float)(total_ms / cfg.reps);
}

/**
 * @brief Main benchmark loop iterating over matrix sizes.
 *
 * Iterates from `minN` to `maxN` with a stride of `step`.
 * For each size, calls `bench_alpaka_once_ms` and computes the effective
 * GFLOPS.
 *
 * **GFLOPS Formula:**
 * \f$ \text{GFLOPS} = \frac{2 \times N^3}{\text{Time (s)} \times 10^9} \f$
 *
 * Prints the results to `stdout` in CSV format:
 * `Solver,Precision,Size,Time_ms,GFLOPs`
 *
 * @param ctx The Alpaka context.
 * @param cfg The parsed configuration.
 * @return 0 on success.
 */
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

/**
 * @brief Entry point for the Micro-Benchmark executable.
 *
 * **Usage:**
 * `./benchmark_alpaka --config <file.prm> [--check N]`
 *
 * - `--config`: Path to the parameter file defining the run (min/max size,
 * reps, solver).
 * - `--check`: If present, runs a correctness verification for size N instead
 * of the benchmark loop.
 *
 * Parses arguments, initializes the device, and dispatches either the
 * verification or the benchmark loop.
 */
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

  if (checkN > 0) {
    alpaka_bench::verify_correctness(ctx, checkN, cfg);
    return 0;
  }

  return alpaka_bench::tb_alpaka_loop(ctx, cfg);
}
