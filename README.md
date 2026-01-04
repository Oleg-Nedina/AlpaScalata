# AlpaScalata

AlpaScalata is a High-Performance Computing
(HPC) project focused on the **optimization and benchmarking
of dense matrix multiplication (GEMM)** on GPUs.
It serves as a testbed for comparing native **CUDA** implementations
against the performance portability of **Alpaka**.

The project provides a reproducible pipeline that includes baseline solvers,
optimized kernels, config-driven benchmarking, and automated plotting.

---

## Documentation Index
**Start here** to navigate the documentation included in `docs/`:


| Filename                       | Description                                                                                                                               |
|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| **`docs/Architecture.md`**     | **Code Structure**. Explains the separation between Interface (gemm.hpp), Engine (Kernels), and Benchmark Driver.                         |
| **`docs/Cluster_Workflow.md`** | **Build & Run Guide**. Detailed instructions for setting up Spack, compiling with CMake/OpenMP, and running MPI jobs on the cluster.      |
| **`docs/Sync_Workflow.md`**    | **Offline Workflow**. How to sync code and results between your local machine and the cluster (which has no internet access) using rsync. |
| **`AlpaScalata_repo.pdf`**     | **Complete Report**. This pdf is the whole report of the project, with implementation details and data analysis.                          |
---

## Key Featured
- **Multi-Backend Support:**
- - **CUDA:** Native baseline (naive) and optimized solvers.
- - **Alpaka:** Portable kernel implementation usable on CUDA, HIP, and CPU backends.
- **Scalability:**
- - **MPI Support:** Distributed execution for horizontal scalability across multi-GPU nodes.
- - **Hybrid Parallelism:** Support for OpenMP host multithreading alongside GPU execution.
- **Robust Benchmarking:**
- - **Config-Driven:** Experiments defined in `.prm` files for reproducibility.
- - **Correctness Checks:** Automated verification against CPU gold references.
- - **CSV Output:** Standardized output format for easy parsing.
- **Optimization Strategies:**
- - Implements **Logical Padding** to handle arbitrary matrix dimensions without segmentation faults.
- - Uses **Vectorized Loads** (`float4`) and **2D Register Tiling** for maximizing memory throughput.

---

## Project Goals

- Study performance characteristics of dense GEMM on modern GPUs
- Compare different implementation strategies (naive, tiled, portable)
- Provide a clean benchmarking and verification pipeline
- Support execution on HPC clusters (PBS-based)
- Compare results obtained with **Alpaka** and with **native CUDA**
---

## Repository Structure
The repository is organized to separate solver logic from benchmarking infrastructure:
```text
├── bench/
│   └── micro/            # Benchmark executables
├── build/                # 
├── config/               # .prm experiment definitions
├── data/
│   └── results/          # Raw CSV benchmark output
├── docs/                 # Design notes and reports
├── external/
│   └── alpaka/           # Alpaka (git submodule)
├── include/              # Public GEMM interfaces
│   └── gemm/
├── jobs/                 # Cluster (PBS) job scripts
├── plots/                # Generated PDF plots (not versioned)
│   ├── comparison/       # Comparisons between CUDA and Alpaka
│   └── individual/       # Individual graphs, for time and GFLOPS
├── scripts/              # Post-processing & plotting
│   └── plot/             # Code to plot graphs
├── src/                  # All solver implementations
│   └── backends/
│       ├── cuda/         # Native CUDA solvers
│       └── alpaka/       # Alpaka solvers (CPU + GPU)
└── CMakeLists.txt        # Top-level build orchestration

```
- Each subdirectory has a corresponding `README.md` file, in order to explain each detail of the present material.

---

## Requirements

- **Hardware:** NVIDIA GPU (L4/V100/etc.).
- **Cluster Modules:** `spack load cuda`, `spack load openmpi`.
- **Compiler:** C++17 compatible (GCC 13+ recommended).
- Python 3 with:
  - `pandas`
    - `matplotlib`
    ---