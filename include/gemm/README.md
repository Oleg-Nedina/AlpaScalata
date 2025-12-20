
---

# `include/gemm/README.md`

```markdown
# GEMM Public API

This directory defines the **public interface** for all GEMM solvers used in the project.

## Design Principle

All solvers must expose the **same function signature**, enabling:
- solver interchangeability
- automatic correctness verification
- unified benchmarking

## API

```cpp
namespace gemm {

struct GemmShape {
  int m, n, k;
};

void gemm_cuda_naive(const float* A,
                     const float* B,
                     float* C,
                     GemmShape s);

}


Matrices are assumed to be row-major:

A: [m × k]

B: [k × n]

C: [m × n]

Future solvers (CUDA optimized, Alpaka) must conform to this interface.
