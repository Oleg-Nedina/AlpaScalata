#include "gemm/gemm.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

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

struct RunCfg {
  int warmup = 5;
  int reps = 20;
  int batch = 1;
  int minN = 256;
  int maxN = 4096;
  int step = 256;
  unsigned seed = 123;
};

static void cpu_ref_gemm(const std::vector<float> &A,
                         const std::vector<float> &B, std::vector<float> &C,
                         int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0.f;
      for (int k = 0; k < N; k++)
        acc += A[i * N + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
  }
}

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

static float bench_once_ms(int N, const RunCfg &cfg) {
  size_t elems = (size_t)N * (size_t)N;

  std::vector<float> A(elems), B(elems), C(elems);

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &x : A)
    x = dist(rng);
  for (auto &x : B)
    x = dist(rng);

  gemm::GemmShape s{N, N, N};

  // warmup
  for (int i = 0; i < cfg.warmup; i++)
    gemm::gemm_alpaka_naive(A.data(), B.data(), C.data(), s);

  // timing
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < cfg.reps; i++)
    gemm::gemm_alpaka_naive(A.data(), B.data(), C.data(), s);
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return (float)(ms / cfg.reps);
}

static void check_correctness(int N, const RunCfg &cfg) {
  size_t elems = (size_t)N * (size_t)N;
  std::vector<float> A(elems), B(elems), C(elems), Ref(elems);

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &x : A)
    x = dist(rng);
  for (auto &x : B)
    x = dist(rng);

  gemm::GemmShape s{N, N, N};
  gemm::gemm_alpaka_naive(A.data(), B.data(), C.data(), s);
  cpu_ref_gemm(A, B, Ref, N);

  check_close(C, Ref);
  std::fprintf(stderr, "CHECK_OK\n");
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
      std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
      return 3;
    }
  }

  if (config_file.empty()) {
    std::fprintf(stderr, "ERROR: missing --config <file.prm>\n");
    return 1;
  }

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

  if (solver != "naive" || prec != "float") {
    std::fprintf(stderr, "ERROR: supported only naive float for now\n");
    return 2;
  }

  if (checkN > 0)
    check_correctness(checkN, cfg);

  std::printf("Solver,Precision,Size,Time_ms,GFLOPs\n");
  for (int N = cfg.minN; N <= cfg.maxN; N += cfg.step) {
    float t_ms = bench_once_ms(N, cfg);
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = flops / (t_ms * 1e-3) / 1e9;
    std::printf("%s,%s,%d,%.6f,%.6f\n", solver.c_str(), prec.c_str(), N, t_ms,
                gflops);
    std::fflush(stdout);
  }
  return 0;
}
