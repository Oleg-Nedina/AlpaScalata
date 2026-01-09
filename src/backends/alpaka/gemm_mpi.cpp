#define GEMM_ENABLE_ALPAKA
#include "gemm/gemm.hpp"
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

/**
 * @file gemm_mpi.cpp
 * @brief Distributed GEMM implementation using MPI and Alpaka.
 *
 * This file serves as the main entry point for the distributed benchmark.
 * It implements a **1D Spatial Decomposition** strategy to distribute the
 * Matrix Multiplication workload across multiple nodes (GPUs).
 *
 * **Key Responsibilities:**
 * - **Host-Side Padding:** Aligns matrix dimensions to multiples of 4 to enable
 * vectorized loads on the GPU.
 * - **MPI Communication:** Uses `MPI_Scatter` to distribute matrix A and
 * `MPI_Bcast` to replicate matrix B.
 * - **Alpaka Integration:** Initializes the Alpaka platform and launches the
 * optimized kernel on the local GPU.
 * - **Verification:** Collects results (`MPI_Gather`) and verifies correctness
 * against expected values.
 */

using Dim2 = alpaka::DimInt<2>;
using Idx = std::size_t;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;

/**
 * @brief Calculates the padded dimension for memory alignment.
 *
 * Rounds up the input dimension `n` to the next multiple of 4.
 * This ensures that every row of the matrix starts at a 128-bit aligned
 * address, which is a strict requirement for using `float4` vectorized loads in
 * the GPU kernel.
 *
 * @param n The original dimension (e.g., M, N, or K).
 * @return The smallest integer `p >= n` such that `p % 4 == 0`.
 */
int get_padded_dim(int n) { return (n + 3) / 4 * 4; }

/**
 * @brief Main execution flow for the Distributed GEMM Benchmark.
 *
 * Orchestrates the entire benchmark process:
 * 1. **Initialization:** Sets up MPI and parses command-line arguments.
 * 2. **Padding Logic:** Calculates padded dimensions (`M_pad`, `N_pad`,
 * `K_pad`) to ensure alignment.
 * 3. **Memory Allocation:** Allocates host buffers with padding (initialized to
 * zero).
 * 4. **Data Distribution:**
 * - **A:** Scattered row-wise among MPI ranks (`MPI_Scatter`). Each rank gets a
 * slice of rows.
 * - **B:** Broadcasted to all ranks (`MPI_Bcast`). Everyone needs full B.
 * 5. **GPU Execution:** Each rank initializes its local GPU via Alpaka and
 * launches the `gemm_alpaka_full_options` kernel.
 * 6. **Result Collection:** The master rank gathers the partial results C from
 * all workers (`MPI_Gather`).
 * 7. **Validation & Reporting:** Verifies the result at specific check-points
 * and reports Performance (TFLOPS).
 *
 * @param argc Argument count.
 * @param argv Argument vector (usage: `./benchmark_mpi <M> <N> <K>`).
 * @return 0 on success, non-zero on failure.
 */
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int M_real = 16384;
  int N_real = 16384;
  int K_real = 16384;

  if (world_rank == 0) {
    if (argc >= 4) {
      M_real = std::atoi(argv[1]);
      N_real = std::atoi(argv[2]);
      K_real = std::atoi(argv[3]);
    } else {
      std::cout << "Use: mpirun ... ./benchmark_mpi <M> <N> <K>" << std::endl;
      std::cout << "Defaulting to 16384x16384x16384" << std::endl;
    }
  }

  int dims[3] = {M_real, N_real, K_real};
  MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
  M_real = dims[0];
  N_real = dims[1];
  K_real = dims[2];

  int M_pad = get_padded_dim(M_real);
  int N_pad = get_padded_dim(N_real);
  int K_pad = get_padded_dim(K_real);

  if (M_pad % world_size != 0) {
    int rem = M_pad % world_size;
    M_pad += (world_size - rem);
  }

  int M_local_pad = M_pad / world_size;

  size_t size_B_pad = static_cast<size_t>(K_pad) * N_pad;
  size_t size_A_local_pad = static_cast<size_t>(M_local_pad) * K_pad;
  size_t size_C_local_pad = static_cast<size_t>(M_local_pad) * N_pad;

  std::vector<float> h_B;
  std::vector<float> h_A_local;
  std::vector<float> h_C_local;
  std::vector<float> h_A_full;
  std::vector<float> h_C_full;

  try {
    h_B.resize(size_B_pad, 0.0f);
    h_A_local.resize(size_A_local_pad, 0.0f);
    h_C_local.resize(size_C_local_pad, 0.0f);

    if (world_rank == 0) {
      double gb_req =
          (double)(static_cast<size_t>(M_pad) * K_pad +
                   static_cast<size_t>(M_pad) * N_pad + size_B_pad) *
          4.0 / 1e9;
      std::cout << "Master: Allocation of RAM (Padded) ~" << gb_req << " GB..."
                << std::endl;
      std::cout << "Padding: [" << M_real << "x" << N_real << "x" << K_real
                << "] -> [" << M_pad << "x" << N_pad << "x" << K_pad << "]"
                << std::endl;

      h_A_full.resize(static_cast<size_t>(M_pad) * K_pad, 0.0f);
      h_C_full.resize(static_cast<size_t>(M_pad) * N_pad, 0.0f);

#pragma omp parallel for
      for (int r = 0; r < M_real; ++r) {
        float val = (float)(r % 100);
        for (int c = 0; c < K_real; ++c) {
          h_A_full[r * K_pad + c] = val;
        }
        for (int c = K_real; c < K_pad; ++c) {
          h_A_full[r * K_pad + c] = 0.0f;
        }
      }

#pragma omp parallel for
      for (int r = 0; r < K_real; ++r) {
        for (int c = 0; c < N_real; ++c) {
          h_B[r * N_pad + c] = 1.0f;
        }
        for (int c = N_real; c < N_pad; ++c) {
          h_B[r * N_pad + c] = 0.0f;
        }
      }
      for (int r = K_real; r < K_pad; ++r) {
        for (int c = 0; c < N_pad; ++c) {
          h_B[r * N_pad + c] = 0.0f;
        }
      }
    }
  } catch (const std::bad_alloc &e) {
    std::cerr << "ERROR: insufficient RAM ! " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (size_B_pad < 2000000000) {
    MPI_Bcast(h_B.data(), size_B_pad, MPI_FLOAT, 0, MPI_COMM_WORLD);
  } else {
    if (world_rank == 0)
      std::cout << "Warning: B to big, local generation." << std::endl;

    std::fill(h_B.begin(), h_B.end(), 0.0f);
    for (int r = 0; r < K_real; ++r)
      for (int c = 0; c < N_real; ++c)
        h_B[r * N_pad + c] = 1.0f;
  }

  if (world_rank == 0)
    std::cout << "partitiong A" << std::endl;

  MPI_Scatter(h_A_full.data(), M_local_pad * K_pad, MPI_FLOAT, h_A_local.data(),
              M_local_pad * K_pad, MPI_FLOAT, 0, MPI_COMM_WORLD);

  auto platform = alpaka::PlatformCudaRt{};
  int num_gpus = (int)alpaka::getDevCount(platform);

  if (num_gpus == 0) {
    if (world_rank == 0)
      std::cerr << "ERROR : none GPU " << std::endl;
    MPI_Finalize();
    return 1;
  }
  int my_device_id = world_rank % num_gpus;
  auto dev = alpaka::getDevByIdx(platform, my_device_id);
  QueueType queue(dev);

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  if (world_rank == 0)
    std::cout << ">>> START GPU <<<" << std::endl;

  gemm::GemmShape local_shape = {M_local_pad, N_pad, K_pad};

  gemm::gemm_alpaka_full_options(queue, h_A_local.data(), h_B.data(),
                                 h_C_local.data(), local_shape);

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();

  if (world_rank == 0)
    std::cout << " Gather results..." << std::endl;

  MPI_Gather(h_C_local.data(), M_local_pad * N_pad, MPI_FLOAT, h_C_full.data(),
             M_local_pad * N_pad, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    double elapsed = end_time - start_time;
    double gflops = (2.0 * (double)M_real * N_real * K_real) / (elapsed * 1e9);

    std::cout << "------------------------------------------------"
              << std::endl;
    std::cout << "Real dim :  " << M_real << " x " << N_real << " x " << K_real
              << std::endl;
    std::cout << "Total time:      " << elapsed << " s" << std::endl;
    std::cout << "TFLOPS utilized:      " << gflops / 1000.0 << " TFLOPS"
              << std::endl;
    std::cout << "------------------------------------------------"
              << std::endl;

    std::cout << "Verification ..." << std::endl;

    int errors = 0;
    std::vector<int> rows_to_check = {0, M_real / 2, M_real - 1, M_real / 4};

    for (int r : rows_to_check) {
      if (r < 0 || r >= M_real)
        continue;

      float expected = (float)(r % 100) * (float)K_real;

      float val_first = h_C_full[r * N_pad + 0];
      float val_last = h_C_full[r * N_pad + (N_real - 1)];

      bool row_pass = true;
      if (std::abs(val_first - expected) > 0.1f)
        row_pass = false;
      if (std::abs(val_last - expected) > 0.1f)
        row_pass = false;

      if (!row_pass) {
        std::cout << "FAIL at row " << r << " -> Expected: " << expected
                  << ", Found: " << val_first << std::endl;
        errors++;
      }
    }

    if (errors == 0)
      std::cout << "RESULT: OK " << std::endl;
    else
      std::cout << "RESULT: FAIL (found " << errors << " error)" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
