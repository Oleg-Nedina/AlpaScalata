
# Tests

This directory contains **unit tests** focused on correctness.

The purpose of these tests is to:
- validate solver implementations
- catch regressions early
- provide confidence before benchmarking

---

## Test Philosophy

- Tests are **deterministic**
- Matrix sizes are small
- Results are compared against a known-correct reference
- Performance is NOT measured here

---

## Structure
tests/
└── unit/
├── test_correctness_cpu.cpp
└── test_correctness_cuda.cpp

---

## Running Tests

Example (CUDA correctness test):

```bash
nvcc -O2 -std=c++17 -Iinclude -arch=sm_89 \
  src/backends/cuda/gemm_cuda_naive.cu \
  tests/unit/test_correctness_cuda.cpp \
  -o test_cuda_naive

./test_cuda_naive

Successful execution prints OK.
