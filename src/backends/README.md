# Backend Implementations

This directory groups solver implementations by **execution backend**.

A backend represents a specific programming model or portability layer
(e.g., CUDA, Alpaka).

---

## Available Backends

### CUDA (`cuda/`)
Native CUDA implementations targeting NVIDIA GPUs.

- Used for baseline and optimized kernels
- Serves as reference for performance comparisons
- Provides the golden reference solver (naive)

### Alpaka (`alpaka/`)
Portable implementations using the Alpaka abstraction layer.

- Designed for performance portability
- Intended to run on multiple architectures
- Currently planned / under development

---

## Adding a New Backend

To add a new backend:
1. Create a new subdirectory
2. Implement solvers matching the GEMM API
3. Register them in the benchmark dispatcher

---
