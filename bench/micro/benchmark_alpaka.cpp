
#include "gemm/gemm.hpp"
#include <external/alpaka/alpaka.hpp>

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

// --------------------
// Helpers
// --------------------
static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

static std::unordered_map<std::string, std::string>
parse_prm(const std::string &path) {
  std::unordered_map<std::string, std::string> kv;
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "ERROR: cannot open config file: %s\n", path.c_str());
    std::exit(2);
  }
  std::string line;
  while (std::getline(in, line)) {
    // strip comments
    auto pos_hash = line.find('#');
    if (pos_hash != std::string::npos)
      line.resize(pos_hash);

    // trim whitespace
    auto trim = [](std::string &x) {
      auto is_ws = [](unsigned char c) { return std::isspace(c); };
      while (!x.empty() && is_ws((unsigned char)x.front()))
        x.erase(x.begin());
      while (!x.empty() && is_ws((unsigned char)x.back()))
        x.pop_back();
    };
    trim(line);
    if (line.empty())
      continue;

    auto pos = line.find('=');
    if (pos == std::string::npos)
      continue;

    std::string k = line.substr(0, pos);
    std::string v = line.substr(pos + 1);
    trim(k);
    trim(v);
    kv[lower(k)] = v;
  }
  return kv;
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
  // naive GEMM for square matrices (N x N)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < N; ++k) {
        acc += A[(size_t)i * N + k] * B[(size_t)k * N + j];
      }
      C[(size_t)i * N + j] = acc;
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

// --------------------
// Alpaka context (independent from CUDA APIs)
// --------------------
namespace alpaka_bench {
using Dim2 = alpaka::DimInt<2>;
using Dim1 = alpaka::DimInt<1>;
using Idx = std::size_t;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
#else
using Acc = alpaka::AccCpuSerial<Dim2, Idx>;
#endif

// Host device for staging buffers
using AccHost = alpaka::AccCpuSerial<Dim2, Idx>;

struct Ctx {
  decltype(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u)) devAcc;
  alpaka::Queue<decltype(devAcc), alpaka::Blocking> queue;

  decltype(alpaka::getDevByIdx(alpaka::Platform<AccHost>{}, 0u)) devHost;

  Ctx()
      : devAcc(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u)), queue(devAcc),
        devHost(alpaka::getDevByIdx(alpaka::Platform<AccHost>{}, 0u)) {}
};

// Run the alpaka solver on DEVICE buffers, copying inputs/outputs via Alpaka.
// NOTE: buffers are 1D contiguous to match CUDA semantics and avoid pitch
// issues.
static void run_naive_device(const Ctx &ctx, const float *Ah, const float *Bh,
                             float *Ch, int N, int warmup, int reps,
                             float *out_avg_ms /*nullable*/) {
  Idx const elems = (Idx)N * (Idx)N;
  Idx const bytes = elems * (Idx)sizeof(float);

  auto const extent1 = alpaka::Vec<Dim1, Idx>{elems};

  // device buffers (contiguous 1D)
  auto bufA_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent1);
  auto bufB_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent1);
  auto bufC_d = alpaka::allocBuf<float, Idx>(ctx.devAcc, extent1);

  // host staging buffers
  auto bufA_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent1);
  auto bufB_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent1);
  auto bufC_h = alpaka::allocBuf<float, Idx>(ctx.devHost, extent1);

  std::memcpy(alpaka::getPtrNative(bufA_h), Ah, (size_t)bytes);
  std::memcpy(alpaka::getPtrNative(bufB_h), Bh, (size_t)bytes);

  alpaka::memcpy(ctx.queue, bufA_d, bufA_h, extent1);
  alpaka::memcpy(ctx.queue, bufB_d, bufB_h, extent1);
  alpaka::wait(ctx.queue);

  gemm::GemmShape s{N, N, N};
  auto *Ad = alpaka::getPtrNative(bufA_d);
  auto *Bd = alpaka::getPtrNative(bufB_d);
  auto *Cd = alpaka::getPtrNative(bufC_d);

  // warmup (not timed)
  for (int i = 0; i < warmup; ++i) {
    gemm::gemm_alpaka_naive(Ad, Bd, Cd, s);
  }

  // timing (kernel only; copies excluded)
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) {
    gemm::gemm_alpaka_naive(Ad, Bd, Cd, s);
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  alpaka::wait(ctx.queue); // ensure all pending memcpy finished (solver does
                           // its own wait)

  if (out_avg_ms) {
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    *out_avg_ms = (float)(ms / reps);
  }

  // D->H for correctness / output
  alpaka::memcpy(ctx.queue, bufC_h, bufC_d, extent1);
  alpaka::wait(ctx.queue);
  std::memcpy(Ch, alpaka::getPtrNative(bufC_h), (size_t)bytes);
}

} // namespace alpaka_bench

// --------------------
// Benchmark logic
// --------------------
static float bench_once_ms(int N, const RunCfg &cfg) {
  size_t elems = (size_t)N * (size_t)N;

  std::vector<float> A(elems), B(elems), C(elems);

  std::mt19937 rng(cfg.seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &x : A)
    x = dist(rng);
  for (auto &x : B)
    x = dist(rng);

  alpaka_bench::Ctx ctx;
  float avg_ms = 0.0f;
  alpaka_bench::run_naive_device(ctx, A.data(), B.data(), C.data(), N,
                                 cfg.warmup, cfg.reps, &avg_ms);
  return avg_ms;
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

  alpaka_bench::Ctx ctx;
  alpaka_bench::run_naive_device(ctx, A.data(), B.data(), C.data(), N,
                                 /*warmup=*/1, /*reps=*/1,
                                 /*out_avg_ms=*/nullptr);

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

  // For now we only support naive/float in alpaka TB (as requested).
  if (solver != "naive") {
    std::fprintf(stderr,
                 "ERROR: alpaka benchmark currently supports only solver=naive "
                 "(got '%s')\n",
                 solver.c_str());
    return 5;
  }
  if (prec != "float" && prec != "fp32") {
    std::fprintf(stderr,
                 "ERROR: alpaka benchmark currently supports only "
                 "precision=float (got '%s')\n",
                 prec.c_str());
    return 6;
  }

  RunCfg cfg;
  if (prm.count("warmup"))
    cfg.warmup = std::stoi(prm["warmup"]);
  if (prm.count("reps"))
    cfg.reps = std::stoi(prm["reps"]);
  if (prm.count("batch"))
    cfg.batch = std::stoi(prm["batch"]);
  if (prm.count("minn"))
    cfg.minN = std::stoi(prm["minn"]);
  if (prm.count("maxn"))
    cfg.maxN = std::stoi(prm["maxn"]);
  if (prm.count("step"))
    cfg.step = std::stoi(prm["step"]);
  if (prm.count("seed"))
    cfg.seed = (unsigned)std::stoul(prm["seed"]);

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
