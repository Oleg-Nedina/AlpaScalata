# AlpaScalata Source Review (excluding external/ and build/)

## Overview
AlpaScalata implements CUDA and Alpaka backends for GEMM, with MPI-distributed variants and micro-benchmarks for performance evaluation. The codebase includes:
- CUDA kernels (naive and optimized) and MPI driver (`AlpaScalata/src/backends/cuda`).
- Alpaka kernels (naive, tiled, full options) and MPI driver (`AlpaScalata/src/backends/alpaka`).
- Micro-benchmarks for CUDA and Alpaka (`AlpaScalata/bench/micro`).
- Plotting scripts for benchmark output (`AlpaScalata/scripts/plot`).

## Potential errors / risks
1) **MPI padding for K dimension is not fully initialized**
   - In the MPI drivers, only the `K_real` columns/rows are initialized, but computation uses `K_pad`. Uninitialized padding can participate in GEMM, yielding incorrect results when `K_real` is not a multiple of 4.
   - CUDA MPI path: `AlpaScalata/src/backends/cuda/gemm_cuda_mpi.cu:137` and `AlpaScalata/src/backends/cuda/gemm_cuda_mpi.cu:149`.
   - Alpaka MPI path: `AlpaScalata/src/backends/alpaka/gemm_mpi.cpp:140` and `AlpaScalata/src/backends/alpaka/gemm_mpi.cpp:148`.

2) **Vectorized float4 loads are not tail-safe**
   - The optimized kernels load `float4` whenever `globalCol < K` or `globalCol < N`, but do not verify `globalCol + 3 < K/N`. This can read beyond valid data when dimensions are not multiples of 4 (or if padding is not zeroed), leading to incorrect values or memory access issues.
   - CUDA kernel: `AlpaScalata/src/backends/cuda/gemm_cuda_full_options.cu:133` and `AlpaScalata/src/backends/cuda/gemm_cuda_full_options.cu:155`.
   - Alpaka kernel: `AlpaScalata/src/backends/alpaka/gemm_alpaka_full_options.cpp:137` and `AlpaScalata/src/backends/alpaka/gemm_alpaka_full_options.cpp:159`.

3) **Host-path padding in CUDA full options is not enforced**
   - `gemm_cuda_full_options` aligns internal strides but does not initialize padded columns/rows in device buffers. If callers provide unpadded host inputs with non-multiple-of-4 dimensions, the vectorized loads can consume garbage values.
   - CUDA full-options host path: `AlpaScalata/src/backends/cuda/gemm_cuda_full_options.cu:431` and `AlpaScalata/src/backends/cuda/gemm_cuda_full_options.cu:445`.

## Notes
- The kernels are designed for padded, multiple-of-4 dimensions, but the public API does not enforce this contract. Either input padding should be guaranteed upstream, or the kernels should be made tail-safe.
- MPI drivers already pad M/N/K, but the padding for K (and any unused portion of B) should be explicitly zeroed to avoid contamination of results.
