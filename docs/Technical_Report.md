# AlpaScalata: High-Performance Distributed GEMM

**AlpaScalata** is an engine for GEMM projected to be scalable (Multi-GPU), portable (Hardware Agnostic) and performing (close to theoretic hardware limits in FP32).

---

## 1. Design Philosophy

The project is based on three fundamental pillars to solve classic HPC problems:

### A. "Single-Source" Portability(Alpaka)

Instead of writing pure CUDA (which only runs on NVIDIA) or HIP (only on AMD), it is used **Alpaka**.

* **Idea:** Writing the abstract kernel in C++ only one time.
* **Result:** The code can be compiled for backend CUDA, HIP, OpenMP (CPU) or TBB without changing the mathematical logic.

### B. Horizontal Scalability (MPI)

A single GPU has limited memory (es. 24GB). To compute huge matrices, the computation must be distributed.

* **Idea:** 1D Spatial Decomposition.
* **Result:** The memory and computational capacity scales linearly with the number of nodes.

### C. "Safety through Architecture" (Padding)

Hardware optimizations (Vectorization) requires a perfect memory alignment. Handle the edge cases inside the GPU kernel costs performance (warp divergence).

* **Idea:** Moving complexity from Kernel (GPU) to Host (CPU).
* **Result:** The GPU always work over perfect data. The edge cases are managed preparing data before everything.

---

## 🛡 2. Padding Management (Stability Secret)

This is the critical component which permits to use vectorized instructions (`float4`) without crashing if "weird" dimensions are submitted (es. 16385).

### The Problem

Modern GPU reads memory in blocks of 128-bit (4 float). If a matrix row ends in an index non-multiple of 16 byte, or if the matrix has an odd dimension, a vectorized upload would cause **Segmentation Fault**, or it would read dirty memory.

### The Solution: "Logical Padding"

Decoupling **Real** dimensions (user) from **Physic** dimensions (memory allocated).

1. **Rounding:**
Each dimension is rounded at the upper multiple of 4.



*(In the code are usedmultiple of Tile Size for ulterior safety, but 4 is the minimum required).*
2. **The "Frame":**
If the matrix would be an image, then it is allocated the frame.
* The valid zone contains data.
* The padding zone gets filled with **Zeros (0.0f)**.


3. **Invariant Math:**
The extended computation to the padding zone does not change the result in the valid zone, thanks to the zero-fill.

### Implementazione Host (`gemm_mpi.cpp`)

```cpp
// 1. Compute safe dimension
int M_pad = (M_real + 3) / 4 * 4;

// 2. Padded Allocation (Init to 0.0f!)
std::vector<float> h_A(M_pad * K_pad, 0.0f);

// 3. Valid data fill
for(int i=0; i<M_real; ++i)
    for(int j=0; j<K_real; ++j)
        h_A[i * K_pad + j] = val; // To note the K_pad stride!

```

---

## 3. Asyncronous Pipeline (Alpaka Host-Side)

To maximise the GPU usage, there is no wait for each data to arrive. It is used a three-stage pipeline ("Double/Triple Buffering").

### Out-of-Core Architecture

The local matrix is chopped in **Chunk**.
**3 Stream** are generated, each one with dedicated buffers.

While **Stream 0** executes the kernel on the Chunk:

* **Stream 1** downloads the results of the Chunk.
* Lo **Stream 2** uploadd the data of the Chunk.

This hides almost completely the PCI-Express bus latency.

---

## 4. The Kernel (GPU Optimization)

The heart of the performance. It has been transformed a 0.5 TFLOPS kernel in a **2.7+ TFLOPS** one by using three tecniques.

### A. 2D Register Tiling (Thread Coarsening)

Instead of assigning 1 thread to 1 output pixel (inefficient), each thread is assigned with a **4x4 block (16 pixel)**.

* **Advantage:** Shared Memory Accesses reduce. An uploaded data in a register is reused to compute more grid points.



### B. Vectorized Global Loads (`float4`)

Thanks to **Padding** implemented Host-side, it is known that each row starts on an index aligned with 128-bit.
Scalar reads (`float`) are replaced with vectorized reads (`float4`).

* **Effect:** Reduce of the number of load instructions by 75%.
* **Implementation:**
```cpp
using float4 = ::float4;
// Safe upload thanks to padding
float4 loaded = *reinterpret_cast<const float4*>(&A[idx]);

```



### C. Loop Unrolling

`#pragma unroll` is used in internal loops which work on registers.
This permits to the compiler to:

1. Remove loop check overhead.
2. Interleave mathematical instructions (FMA) to hide the arithmetic pipeline latency.

---

## 5. Results and Conclusions

The combination of these strategies produces a fast and sound system.

| Metric          | Value / Notes                                         |
|-----------------|-------------------------------------------------------|
| **Precision**   | FP32 (Single Precision)                               |
| **Performance** | **~2.7 TFLOPS** (su NVIDIA L4)                        |
| **Speedup**     | **+440%** respect to base version (Naive)             |
| **Scalability** | Multi-GPU via MPI                                     |
| **Robustness**  | Handle arbitrary dimensions (odd/prime) without crash |

**AlpaScalata** shows that it is possible to write a **portable** code (thanks to Alpaka) without renouncing to extreme performances, given that the memory architecture is well-projected(Padding and Vectorization).
