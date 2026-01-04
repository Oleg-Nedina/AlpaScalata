### 1.(The Architecture)

Code divided in **three distinct logic components**:

1. **Interface (`gemm.hpp`)**:
* The "contract" that each file must follow.
* Contains the definitions of each data structure (`GemmShape`) and the signatures of the functions (`gemm_alpaka_naive`, `gemm_cuda_naive`).
* **Crucial:** Use `#ifdef GEMM_ENABLE_ALPAKA` to hide heavy Alpaka header to files which do not need it (like the pure CUDA solvers), avoiding compilation conflicts.


2. **The engine (`gemm_alpaka_naive.cpp`)**:
* Contains the real **computation logic** (Alpaka Kernel).
* It is compiled by itself.
* Uses **Explicit Instance** (`template void ...`) at the end of the file. This says to the compiler: *"Prepare the binary code for this kernel with these types (GPU), in order to avoid compilation to callers"*.


3. **The Benchmark Driver (`benchmark_alpaka.cu`)**:
* Does not contain computation logic, **orchestration** only.
* Reads each configuration file (`.prm`).
* Manages test cycle (`min`, `max`, `step`).
* Executes the **Warmup**.
* Implements the **Gold Check (`verify_correctness`)**: compares the GPU result with a CPU version, slow but safe, to grant value soundness.
* Produces standardised CSV output.



---

### 2. Why (Solutions to Problems)


* **"Allocator is not a template":**
* *Cause:* NVCC (NVIDIA compiler) cannot read new standard headers C++20 di GCC 13 (in particular `<string>` and `<memory>`).
* *Solution: force standard **C++17** and added flag `-D_GLIBCXX_USE_CXX11_ABI=0`. This grants binary compatibility.


* **"Namespace Pollution":**
* *Caus:* Included `<alpaka/alpaka.hpp>` *inside* `namespace gemm`. This instructed teh compiler to search `std::vector` inside `gemm::std::vector`, causing errors.
* *Solution:* Moved include **outside** namespace and protected with macro.


* **"Alpaka over CPU instead of GPU":**
* *Cause:* CMake does not correctly pass the definition to the preprocessor, causing the `#else` which activated the CPU.
* *Solution:* Forced `-DALPAKA_ACC_GPU_CUDA_ENABLED` directly into the compilation flags, and removed the CPU fallback from the code to be sure that, if it compiles, it is using the GPU.



---

### 3. Why it is easy to add new Solver

Thanks to this structure, to add a new solver (es. `tiled` or `shared_memory`), it is not needed to write the benchmark from scratch. It is only required to:

1. **Copy the solver into the file:**
Create `gemm_alpaka_tiled.cpp` copying the `naive` one. Only change the code inside `operator()` (kernel).
2. **Register it into the header:**
Add a row in `gemm.hpp`:
```cpp
template <typename TQueue>
void gemm_alpaka_tiled(TQueue &queue, ...);

```


3. **Update the Benchmark:**
In `main` of `benchmark_alpaka.cu`, add `else if`:
```cpp
if (cfg.solver == "alpaka_naive") {
    gemm::gemm_alpaka_naive(...);
} else if (cfg.solver == "alpaka_tiled") {
    gemm::gemm_alpaka_tiled(...); // <--- New call
}

```


4. **Configuration:**
Create a new file `.prm` writing `solver = alpaka_tiled` and start the benchmark. Other stuff (parsing, CSV, error verify, graphs) will automatically work.
