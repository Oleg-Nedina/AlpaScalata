# AlpaScalata

AlpaScalata is a High-Performance Computing (HPC) project focused on the **optimization and benchmarking of dense matrix multiplication (GEMM)** on GPUs.

The project currently provides:
- a **CUDA naive baseline** implementation
- a **config-driven benchmark pipeline**
- **correctness verification** against a golden reference
- **automatic performance analysis and plotting**
- infrastructure ready for **CUDA-optimized kernels** and **Alpaka-based portable backends**

The repository is designed to be **reproducible, extensible, and cluster-friendly**.

---

## Project Goals

- Study performance characteristics of dense GEMM on modern GPUs
- Compare different implementation strategies (naive, tiled, portable)
- Provide a clean benchmarking and verification pipeline
- Support execution on HPC clusters (PBS-based)

---

## Repository Structure

AlpaScalata/
├── include/ # Public GEMM API (solver interfaces)
├── src/ # Backend implementations (CUDA, Alpaka)
├── bench/ # Micro-benchmarks
├── tests/ # Unit correctness tests
├── configs/ # Benchmark configuration files (.prm)
├── scripts/ # Plotting and utility scripts
├── data/ # Benchmark results (CSV)
├── plots/ # Generated plots (PDF)
├── jobs/ # PBS job scripts
└── docs/ # Project documentation
--- 



Each subdirectory contains its own README with more details.

---

## Requirements

- NVIDIA GPU
- CUDA Toolkit (loaded via Spack on the cluster)
- C++17 compatible compiler
- Python 3 with:
  - `pandas`
    - `matplotlib`

    ---

    ## Quick Start (Local or GPU Node)

    ```bash
    # Load CUDA (cluster)
    source /software/spack-v1.0/share/spack/setup-env.sh
    spack load cuda

    # Compile benchmark
    nvcc -O3 -std=c++17 -Iinclude -arch=sm_89 \
      bench/micro/benchmark_test.cu \
      src/backends/cuda/gemm_cuda_naive.cu \
      -o benchmark_test

    # Run benchmark
    ./benchmark_test --config configs/parameters.prm --check 256 \
      > data/results/naive_float.csv

    # Plot results
    python3 scripts/plot/plot_results.py data/results/naive_float.csv


   ## Extending the Project

    # To add a new solver:

   1) Implement the solver with the same GEMM API

   2) Register it in the benchmark dispatcher

   3) Reuse the same .prm configuration and correctness check

   4) Compare results via generated plots

  5) See the README files in src/ and bench/ for details.



