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

.
├── CMakeLists.txt        # Top-level build orchestration
├── include/              # Public GEMM interfaces
│   └── gemm/
├── src/                  # All solver implementations
│   ├── backends/
│   │   ├── cuda/         # Native CUDA solvers
│   │   └── alpaka/       # Alpaka solvers (CPU + GPU)
│   └── common/
├── bench/
│   └── micro/            # Benchmark executables
├── tests/                # Correctness tests (small, deterministic)
├── config/               # .prm experiment definitions
├── data/
│   └── results/          # Raw CSV benchmark output
├── plots/                # Generated PDF plots (not versioned)
├── scripts/              # Post-processing & plotting
├── jobs/                 # Cluster (PBS) job scripts
├── external/
│   └── alpaka/           # Alpaka (git submodule)
└── docs/                 # Design notes and reports





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


## **Every Useful Command can be Found in Docs**