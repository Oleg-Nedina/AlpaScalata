# CUDA GEMM Implementations

This directory contains three tiers of Matrix Multiplication implementations for NVIDIA GPUs, ranging from a baseline educational kernel to a distributed, multi-GPU out-of-core solver.

## Files Overview

### 1. `gemm_cuda_naive.cu`
**The Baseline Reference.**
This file implements the simplest form of matrix multiplication on the GPU to serve as a correctness verification tool.
* **Kernel Logic:** Each thread computes exactly one element of the result matrix $C$ by iterating through the $K$ dimension.
* **Configuration:** Uses a fixed block size of 16x16 threads.
* **Memory:** Relies entirely on global memory reads without shared memory caching or tiling.

### 2. `gemm_cuda_full_options.cu`
**The Optimized Solver.**
This file contains a high-performance implementation designed for single-node execution, capable of handling matrices larger than the GPU memory.
* **Tiling & Shared Memory:** Uses dynamic shared memory (`Ms_Ns`) to load tiles of $A$ and $B$, reducing global memory traffic.
* **Dynamic Tuning:** The `get_optimal_tile_width` function calculates the block size (up to 32) at runtime based on the specific GPU's shared memory capacity.
* **Out-of-Core Processing:** If the matrix exceeds VRAM (checked via `cudaMemGetInfo`), the function `gemm_out_of_core` breaks the operation into 4096-element chunks.
* **Stream Pipelining:** Implements double-buffering with 2 CUDA streams to overlap the upload/download of data tiles with computation.

### 3. `gemm_cuda_mpi.cu`
**The Distributed Layer.**
This file wraps the optimized solver in an MPI (Message Passing Interface) layer to scale across multiple GPUs or nodes.
* **Data Partitioning:**
  * **Scatter:** Matrix $A$ is split row-wise (`MPI_Scatterv`) among ranks.
  * **Broadcast:** Matrix $B$ is sent in its entirety (`MPI_Bcast`) to all ranks.
  * **Gather:** Result rows of $C$ are collected (`MPI_Gatherv`) back to the root rank.
* **Hybrid Execution:** The local computation on every MPI rank is offloaded to the optimized `gemm_cuda_full_options` solver.
* **Multi-GPU Assignment:** Uses round-robin logic (`rank % deviceCount`) to assign specific GPUs to specific MPI processes on a single node.

---

## Technical Architecture

### Memory Management Strategy
The implementations use a tiered approach to memory:
1.  **Naive:** Direct global memory access.
2.  **Full Options (In-Core):** If the data fits in VRAM (plus a safety margin), it allocates device buffers once and launches a single grid.
3.  **Full Options (Out-of-Core):** If data is too large, it utilizes pinned host memory (`safe_host_register`) and streams to cycle chunks through smaller GPU buffers.

### Distributed Algorithm
The MPI implementation assumes the matrices are row-major.
* Total Rows ($M$) are divided by the number of MPI ranks (`size`).
* Any remainder rows ($M \% size$) are distributed one per rank starting from rank 0.
* Host memory for MPI buffers is allocated using `cudaMallocHost` (pinned memory) to accelerate the subsequent transfer to the GPU.

---

## Compilation Guidelines

*Note: These files depend on a common header `gemm/gemm.hpp` (not included in this directory).*

### Prerequisites
* NVIDIA CUDA Toolkit (`nvcc`)
* MPI Implementation (e.g., OpenMPI, MPICH)
* C++ Compiler compatible with CUDA

### Build Steps
Since `gemm_cuda_mpi.cu` depends on the definitions in `gemm_cuda_full_options.cu`, you must compile them as separate objects and link them.

```bash
# 1. Compile the Optimized Solver (Relocatable Device Code)
nvcc -c gemm_cuda_full_options.cu -o gemm_opt.o -O3

# 2. Compile the Naive Solver
nvcc -c gemm_cuda_naive.cu -o gemm_naive.o -O3

# 3. Compile the MPI Wrapper
# Note: Ensure the include path (-I) points to the directory containing gemm/gemm.hpp
nvcc -c gemm_cuda_mpi.cu -o gemm_mpi.o -O3

# 4. Link (Using mpicxx to handle MPI libraries)
mpicxx -o gemm_solver gemm_mpi.o gemm_opt.o gemm_naive.o -L/usr/local/cuda/lib64 -lcudart