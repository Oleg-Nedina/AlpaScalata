# AlpaScalata: High-Performance Distributed GEMM

**AlpaScalata** is a robust, portable, and highly optimized engine for General Matrix Multiplication ().
It is designed to bridge the gap between high-level hardware abstraction (via **Alpaka**) and bare-metal performance, leveraging **MPI** for multi-GPU scalability and advanced **CUDA** optimization techniques for maximum throughput.

---

## 🌟 Key Features

* **Distributed Computing:** Scales across multiple nodes/GPUs using MPI (1D Spatial Decomposition).
* **Hardware Agnostic:** Built on **Alpaka**, allowing compilation for NVIDIA (CUDA), AMD (HIP), and CPUs from a single source code.
* **Vectorized Execution:** Utilizes **128-bit Vectorized Loads** (`float4`) for maximum memory bandwidth.
* **2D Register Tiling:** Implements "Thread Coarsening" where each thread computes a 4x4 micro-tile to increase arithmetic intensity.
* **Logical Padding:** Automatically handles arbitrary matrix dimensions (odd/prime sizes) ensuring memory alignment without crashes.
* **Async Pipeline:** Implements a 3-stage software pipeline (Compute/Upload/Download) to hide PCIe latency for large datasets.

---

## 🏗 System Architecture

The implementation is divided into three logical layers working in synergy:

### 1. Host Layer: Safety & Distribution (MPI + Padding)

Before touching the GPU, the Host prepares the data to ensure **stability** and **alignment**.

* **Logical Padding:** GPU hardware works best with 128-bit aligned memory. If a user requests a matrix of size , a naive kernel would crash or require slow boundary checks.
* *Our Solution:* We round up dimensions to the nearest multiple of 4 (e.g., ).
* We allocate a "frame" of zeros around the valid data.
* **Result:** The Kernel *always* sees perfect alignment. `float4` instructions are safe to use everywhere.


* **MPI Decomposition:**
* **Matrix A:** Sliced horizontally. Each worker receives  rows.
* **Matrix B:** Broadcasted fully to all workers.
* **Matrix C:** Computed locally and gathered back to the Master node.



### 2. Management Layer: The Hybrid Pipeline (Alpaka)

The system decides at runtime how to execute the problem based on available VRAM.

* **In-Core Path (Standard):** If the matrix fits in VRAM, pointers are passed directly to the kernel (Zero-Copy). Max throughput (~2.7 TFLOPS on L4).
* **Out-of-Core Path (Pipelined):** If the matrix is too large, it is split into tiles (e.g., ). We use **3 Async Streams**:
1. **Stream 1:** Compute Tile .
2. **Stream 2:** Upload Tile .
3. **Stream 3:** Download Tile .


* *Benefit:* Hides the slow PCIe bus latency behind the GPU computation.



### 3. Compute Layer: The "Ultimate" Kernel

The kernel (`GemmCoarsenedKernel`) is where the raw performance comes from. It moves away from the naive "1 Thread = 1 Pixel" approach.

* **Vectorized Global Loads (`float4`):**
Instead of loading 1 `float` at a time (32-bit), we load 4 `floats` (128-bit) in a single instruction.
* *Impact:* Reduces memory transaction overhead by 75%.


* **2D Register Tiling:**
Each thread calculates a **4x4 block (16 pixels)** of the output matrix.
* Data is loaded into **Shared Memory** (Macro-Tile ).
* Then loaded into **Registers** (Micro-Tile ).
* *Impact:* drastic reduction of Shared Memory bandwidth pressure.


* **Loop Unrolling:**
Inner loops are unrolled via `#pragma unroll` to allow the compiler to pipeline Fused Multiply-Add (FMA) instructions.

---

## 📊 Performance & Optimization Logic

We compared different implementation strategies on an NVIDIA L4 GPU:

| Kernel Version | Architecture Strategy | Performance (FP32) | Bottleneck |
| --- | --- | --- | --- |
| **Naive** | 1 Thread = 1 Pixel, Direct Global Mem | ~0.5 TFLOPS | Memory Latency |
| **Tiled** | Shared Memory Blocking | ~1.1 TFLOPS | Memory Bandwidth |
| **Coarsened** | 1 Thread = 16 Pixels, Scalar Loads | ~1.4 TFLOPS | Load Instruction Overhead |
| **Full_Option** | **Coarsened + Vectorized (`float4`)** | **~2.7 TFLOPS** | Compute Bound (FP32) |

**Why `float4` matters:**
Without vectorization, the GPU execution units (CUDA Cores) were starving, waiting for data. By fetching 128 bits at once, we saturated the memory bandwidth, allowing the compute units to run at full speed.

---

## 🛠 Usage

### Prerequisites

* CMake 3.18+
* CUDA Toolkit 11.0+
* MPI Implementation (OpenMPI, MPICH)
* C++17 Compiler

### Build

```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON -DENABLE_ALPAKA=ON -DCMAKE_BUILD_TYPE=Release
make -j

```

### Run (MPI Distributed)

To run a distributed benchmark on 2 GPUs with a matrix size of :

```bash
mpirun -np 2 ./src/benchmark_mpi 16384 16384 16384

```

To test the **Padding Robustness** (Odd dimensions):

```bash
mpirun -np 2 ./src/benchmark_mpi 16385 16385 16385

```

*Expected Output: `RESULT: OK` (System automatically handles padding).*

---

## 📝 Conclusion

The `gemm_alpaka_full_options` ensures:

1. **Maximum Performance** via Vectorization and Register Tiling.
2. **Maximum Reliability** via Host-side Padding and Batching.
3. **Scalability** via MPI.

