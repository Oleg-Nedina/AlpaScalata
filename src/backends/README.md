# CUDA Backends

This directory contains CUDA-based implementations of GEMM.

## Available Solvers

### `gemm_cuda_naive`

- Simple, baseline CUDA implementation
- One thread computes one output element
- Used as:
  - performance baseline
  - **golden reference** for correctness checks

## Design Notes

- Implementations follow the common GEMM API
- No assumptions about alignment or special memory layouts
- Optimized solvers (tiled, shared memory) will be added here
