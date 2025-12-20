# Unit Tests

This directory contains low-level correctness tests for GEMM solvers.

Each test:
- allocates small matrices
- runs a solver
- compares the result against a reference implementation

---

## Reference Strategy

- CPU reference for very small matrices
- CUDA naive solver for GPU-level validation

These tests ensure correctness independently of performance.

---
