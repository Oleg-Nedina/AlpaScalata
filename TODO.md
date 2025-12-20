# AlpaScalata – Development TODO & Roadmap

This document outlines the planned development steps for the AlpaScalata project.
The goal is to evolve the project from a baseline CUDA implementation into a
comparative, extensible, and well-documented HPC benchmarking framework.

---

## 1. CUDA Solver Development

**Objective:** extend beyond the naive baseline and explore meaningful GPU optimization strategies.

### Tasks
- [ ] Implement additional CUDA GEMM solvers, including:
  - [ ] Shared-memory tiled GEMM
  - [ ] SIMD / vectorized variants (where applicable)
  - [ ] Dynamic tiling strategies (runtime tile size selection)
  - [ ] Queue- or block-level scheduling approaches (e.g., block queues, work decomposition)
- [ ] Ensure all CUDA solvers:
  - conform to the common GEMM API
  - pass correctness checks against the naive solver
- [ ] Document design choices and expected performance characteristics for each solver

**Goal:** implement at least **2–3 distinct CUDA optimization strategies** to enable meaningful performance comparisons.

---

## 2. Alpaka Backend Implementation

**Objective:** introduce performance portability using Alpaka and compare it with native CUDA.

### Tasks
- [ ] Implement an Alpaka-based **naive GEMM** solver
- [ ] Implement one or more optimized Alpaka variants (e.g., tiled/shared-memory style)
- [ ] Ensure API compatibility with existing CUDA solvers
- [ ] Integrate Alpaka solvers into:
  - the benchmark dispatcher
  - the correctness verification pipeline
- [ ] Analyze portability vs performance trade-offs between CUDA and Alpaka

**Goal:** replicate the CUDA solver progression (naive → optimized) within Alpaka.

---

## 3. Repository Cleanup and Documentation

**Objective:** improve maintainability and clarity of the codebase.

### Tasks
- [ ] Review and clean unused or placeholder files
- [ ] Remove empty directories or files that are no longer needed
- [ ] Ensure every remaining directory has a meaningful README
- [ ] Verify consistency and accuracy across all README files
- [ ] Clearly distinguish between:
  - core code
  - benchmarks
  - tests
  - experimental or future components

**Goal:** a clean, readable repository suitable for public release or academic evaluation.

---

## 4. Benchmark Execution and Comparative Analysis

**Objective:** perform systematic performance evaluation across implementations.

### Tasks
- [ ] Run micro-benchmarks for all implemented CUDA solvers
- [ ] Run equivalent benchmarks for Alpaka solvers
- [ ] Generate and archive CSV results for all configurations
- [ ] Produce comparative plots (time and GFLOPs)
- [ ] Analyze whether:
  - CUDA mid-level optimizations fully utilize the GPU
  - more complex strategies provide diminishing returns
- [ ] Investigate whether “half-optimized” CUDA kernels are sufficient for practical GPU utilization

**Goal:** understand which optimizations are actually worthwhile on the target GPU architecture.

---

## 5. CPU-Side Optimizations

**Objective:** provide a CPU baseline and explore lightweight CPU optimizations.

### Tasks
- [ ] Implement a simple CPU GEMM baseline
- [ ] Introduce fast CPU optimizations such as:
  - loop reordering
  - cache-friendly blocking
  - basic SIMD (if applicable)
- [ ] Integrate CPU benchmarks into the existing framework
- [ ] Compare CPU vs GPU performance for small and medium problem sizes

**Goal:** contextualize GPU performance and highlight when GPU acceleration is beneficial.

---

## Final Goal

Deliver a **complete, reproducible, and well-documented benchmarking framework**
that enables:
- fair comparison between CUDA and Alpaka
- evaluation of multiple optimization strategies
- informed conclusions about performance portability and optimization effort

---
