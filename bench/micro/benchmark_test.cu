
// bench/micro/benchmark_test.cu
#include "gemm/gemm.hpp"  // gemm::gemm_cuda_naive + GemmShape
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

#include <fstream>
#include <sstream>
#include <unordered_map>



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

    m[key] = val;
  }
  return m;
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

static void check_close(const std::vector<float> &got,
                        const std::vector<float> &ref, float atol = 1e-4f,
                        float rtol = 1e-4f) {
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

static gemm::GemmFn solver_from_name(const std::string &solver) {
  if (solver == "naive")
    return gemm::gemm_cuda_naive;
  // if (solver == "tiled") return gemm::gemm_cuda_tiled;
  return nullptr;
}

static void check_solver_vs_naive(int N, const RunCfg &cfg,
                                  gemm::GemmFn solver_fn) {
  const size_t elems = (size_t)N * (size_t)N;
  const size_t bytesA = elems * sizeof(float);
  const size_t bytesB = elems * sizeof(float);
  const size_t bytesC = elems * sizeof(float);

  // Host init (stesso seed => deterministico)
  std::vector<float> A(elems), B(elems);
  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &x : A)
    x = dist(rng);
  for (auto &x : B)
    x = dist(rng);

  // Device alloc
  float *Ad = nullptr, *Bd = nullptr, *Cref_d = nullptr, *Ctest_d = nullptr;
  ck(cudaMalloc(&Ad, bytesA), "cudaMalloc A");
  ck(cudaMalloc(&Bd, bytesB), "cudaMalloc B");
  ck(cudaMalloc(&Cref_d, bytesC), "cudaMalloc Cref");
  ck(cudaMalloc(&Ctest_d, bytesC), "cudaMalloc Ctest");

  ck(cudaMemcpy(Ad, A.data(), bytesA, cudaMemcpyHostToDevice), "H2D A");
  ck(cudaMemcpy(Bd, B.data(), bytesB, cudaMemcpyHostToDevice), "H2D B");

  gemm::GemmShape s{N, N, N};

  // Reference: naive
  gemm::gemm_cuda_naive(Ad, Bd, Cref_d, s);
  ck(cudaGetLastError(), "naive launch");
  ck(cudaDeviceSynchronize(), "naive sync");

  // Test: solver under test
  solver_fn(Ad, Bd, Ctest_d, s);
  ck(cudaGetLastError(), "solver launch");
  ck(cudaDeviceSynchronize(), "solver sync");

  // Copy back
  std::vector<float> Cref(elems), Ctest(elems);
  ck(cudaMemcpy(Cref.data(), Cref_d, bytesC, cudaMemcpyDeviceToHost),
     "D2H Cref");
  ck(cudaMemcpy(Ctest.data(), Ctest_d, bytesC, cudaMemcpyDeviceToHost),
     "D2H Ctest");

  // Compare
  check_close(Ctest, Cref);

  // Cleanup
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cref_d);
  cudaFree(Ctest_d);
}

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

static int tb_naive_float(const RunCfg &cfg,
                          const std::string& solver,
                          const std::string& prec) {
  std::printf("Solver,Precision,Size,Time_ms,GFLOPs\n");
  for (int N = cfg.minN; N <= cfg.maxN; N += cfg.step) {
    float t_ms = bench_naive_once_ms(N, cfg);

    // FLOPs = 2*N^3 per GEMM (batch=1 qui; se vuoi batch, moltiplica)
    double flops = 2.0 * (double)N * (double)N * (double)N * (double)cfg.batch;
    double gflops = flops / (t_ms * 1e-3) / 1e9;

    std::printf("%s,%s,%d,%.6f,%.6f\n", solver.c_str(), prec.c_str(), N, t_ms,
                gflops);

    std::fflush(stdout);
  }
  return 0;
}

static float bench_full_options_once_ms(int N, const RunCfg &cfg) {
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
        gemm::gemm_cuda_full_options(Ad, Bd, Cd, s);
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


static int tb_full_options_float(const RunCfg &cfg,
                          const std::string& solver,
                          const std::string& prec) {
    std::printf("Solver,Precision,Size,Time_ms,GFLOPs\n");
    for (int N = cfg.minN; N <= cfg.maxN; N += cfg.step) {
        float t_ms = bench_full_options_once_ms(N, cfg);

        // FLOPs = 2*N^3 per GEMM (batch=1 qui; se vuoi batch, moltiplica)
        double flops = 2.0 * (double)N * (double)N * (double)N * (double)cfg.batch;
        double gflops = flops / (t_ms * 1e-3) / 1e9;

        std::printf("%s,%s,%d,%.6f,%.6f\n", solver.c_str(), prec.c_str(), N, t_ms,
                    gflops);

        std::fflush(stdout);
    }
    return 0;
}

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
      std::fprintf(stderr,
                   "Unknown option: %s\n"
                   "Usage: %s --config <file.prm> [--check N]\n",
                   a.c_str(), argv[0]);
      return 3;
    }
  }

  if (config_file.empty()) {
    std::fprintf(stderr,
                 "ERROR: missing required --config <file.prm>\n"
                 "Usage: %s --config <file.prm> [--check N]\n"
                 "Example: %s --config configs/naive_float.prm --check 256\n",
                 argv[0], argv[0]);
    return 1;
  }

  // carica prm
  RunCfg cfg;
  auto prm = parse_prm(config_file);

  auto require_key = [&](const char *k) -> std::string {
    std::string key = lower(std::string(k));
    auto it = prm.find(key);
    if (it == prm.end()) {
      std::fprintf(stderr, "ERROR: missing key '%s' in %s\n", k,
                   config_file.c_str());
      std::exit(4);
    }
    return it->second;
  };

  std::string solver = lower(require_key("solver"));
  std::string prec = lower(require_key("precision"));

  cfg.minN = std::stoi(require_key("min"));
  cfg.maxN = std::stoi(require_key("max"));
  cfg.step = std::stoi(require_key("step"));
  cfg.reps = std::stoi(require_key("reps"));
  cfg.warmup = std::stoi(require_key("warmup"));
  cfg.seed = (unsigned)std::stoul(require_key("seed"));

  // check correttezza (solver vs naive) se richiesto
  if (checkN > 0) {
    auto fn = solver_from_name(solver);
    if (!fn) {
      std::fprintf(stderr, "ERROR: unknown solver for --check: %s\n",
                   solver.c_str());
      return 2;
    }
    std::fprintf(stderr, "Running correctness check: %s vs naive (N=%d)\n",
                 solver.c_str(), checkN);
    check_solver_vs_naive(checkN, cfg, fn);
    std::fprintf(stderr, "CHECK_OK\n");
  }

  // dispatcher
  if (solver == "naive" && prec == "float") {
    return tb_naive_float(cfg, solver, prec);
  }
  else if(solver == "full_options" && prec == "float"){
      return tb_full_options_float(cfg, solver, prec);
  }

  std::fprintf(stderr,
               "ERROR: unsupported solver/precision from config: %s %s\n"
               "Supported: naive float\n",
               solver.c_str(), prec.c_str());
  return 2;
}

