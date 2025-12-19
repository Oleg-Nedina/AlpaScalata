
// bench/micro/benchmark_test.cu
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

#include "gemm/gemm.hpp" gemm::gemm_cuda_naive + GemmShape

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

static void ck(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(2);
  }
}

struct RunCfg {
  int warmup = 5;
  int reps = 20;
  int batch = 1;

  int minN = 256;
  int maxN = 4096;
  int step = 256;

  unsigned seed = 123;
};

static float bench_naive_once_ms(int N, const RunCfg &cfg) {
  // GEMM NxN: A[N,N], B[N,N], C[N,N]
  const size_t bytesA = (size_t)N * N * sizeof(float);
  const size_t bytesB = (size_t)N * N * sizeof(float);
  const size_t bytesC = (size_t)N * N * sizeof(float);

  // Host init
  std::vector<float> A((size_t)N * N), B((size_t)N * N);

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &x : A)
    x = dist(rng);
  for (auto &x : B)
    x = dist(rng);

  // Device alloc
  float *Ad = nullptr, *Bd = nullptr, *Cd = nullptr;
  ck(cudaMalloc(&Ad, bytesA), "cudaMalloc A");
  ck(cudaMalloc(&Bd, bytesB), "cudaMalloc B");
  ck(cudaMalloc(&Cd, bytesC), "cudaMalloc C");

  ck(cudaMemcpy(Ad, A.data(), bytesA, cudaMemcpyHostToDevice), "H2D A");
  ck(cudaMemcpy(Bd, B.data(), bytesB, cudaMemcpyHostToDevice), "H2D B");

  // Warmup
  gemm::GemmShape s{N, N, N};
  for (int i = 0; i < cfg.warmup; i++) {
    gemm::gemm_cuda_naive(Ad, Bd, Cd, s);
  }
  ck(cudaDeviceSynchronize(), "sync after warmup");

  // Timing
  cudaEvent_t start, stop;
  ck(cudaEventCreate(&start), "event create start");
  ck(cudaEventCreate(&stop), "event create stop");

  ck(cudaEventRecord(start), "event record start");
  for (int i = 0; i < cfg.reps; i++) {
    gemm::gemm_cuda_naive(Ad, Bd, Cd, s);
  }
  ck(cudaEventRecord(stop), "event record stop");
  ck(cudaEventSynchronize(stop), "event sync stop");

  float total_ms = 0.0f;
  ck(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");

  ck(cudaEventDestroy(start), "destroy start");
  ck(cudaEventDestroy(stop), "destroy stop");

  // Cleanup
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);

  return total_ms / cfg.reps;
}

static int tb_naive_float(const RunCfg &cfg) {
  std::printf("Size,Time_ms,GFLOPs\n");
  for (int N = cfg.minN; N <= cfg.maxN; N += cfg.step) {
    float t_ms = bench_naive_once_ms(N, cfg);

    // FLOPs = 2*N^3 per GEMM (batch=1 qui; se vuoi batch, moltiplica)
    double flops = 2.0 * (double)N * (double)N * (double)N * (double)cfg.batch;
    double gflops = flops / (t_ms * 1e-3) / 1e9;

    std::printf("%d,%.6f,%.6f\n", N, t_ms, gflops);
    std::fflush(stdout);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "Usage: %s <solver> <precision> [--min N --max N --step N "
                 "--reps R --warmup W --seed S]\n",
                 argv[0]);
    std::fprintf(stderr,
                 "Example: %s naive float --min 256 --max 4096 --step 256 "
                 "--reps 30 --warmup 5\n",
                 argv[0]);
    return 1;
  }

  std::string solver = lower(argv[1]);
  std::string prec = lower(argv[2]);

  RunCfg cfg;

  // opzionale: parsing argomenti
  for (int i = 3; i < argc; i++) {
    auto need = [&](const char *opt) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", opt);
        std::exit(3);
      }
    };
    std::string a = argv[i];
    if (a == "--min") {
      need("--min");
      cfg.minN = std::atoi(argv[++i]);
    } else if (a == "--max") {
      need("--max");
      cfg.maxN = std::atoi(argv[++i]);
    } else if (a == "--step") {
      need("--step");
      cfg.step = std::atoi(argv[++i]);
    } else if (a == "--reps") {
      need("--reps");
      cfg.reps = std::atoi(argv[++i]);
    } else if (a == "--warmup") {
      need("--warmup");
      cfg.warmup = std::atoi(argv[++i]);
    } else if (a == "--seed") {
      need("--seed");
      cfg.seed = (unsigned)std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
      return 3;
    }
  }

  // dispatcher
  if (solver == "naive" && prec == "float") {
    return tb_naive_float(cfg);
  }

  std::fprintf(stderr,
               "Unknown solver/precision: %s %s (supported: naive float)\n",
               argv[1], argv[2]);
  return 2;
}
