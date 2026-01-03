# AlpaScalata — GPU Build & Run Guide (Spack + GCC 13 + MPI/OpenMP)

This guide shows step by step how to configure the environment, to compile and to execute **AlpaScalata** over a GPU node belonging to the cluster, including the support **Distributed (MPI)** and the parallelism **Host (OpenMP)**.

---

## Prerequisites: Access the GPU Node

Check to be logged over a node with GPU (not the login node).

1.  **Upload Spack Environment:**
    ```bash
    source /software/spack-v1.0/share/spack/setup-env.sh
    ```

2.  **Upload needed modules (CUDA and MPI):**
    ```bash
    spack load cuda
    spack load openmpi
    ```

3.  **Verify GPU:**
    ```bash
    nvidia-smi
    ```

---

## Environment Variables Setup

Set the path for CUDA 13.0.2 and the required libraries.

```bash
# Specific path for Spack Download
export MY_CUDA_ROOT="/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap"

# Upload PATH and Libraries
export PATH="${MY_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MY_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}"

```

---

## Cleanup and CMake Configuration

Robust configuration for compatibility GCC 13 + CUDA (C++17 + ABI Fix) and ** OpenMP activation** (used for `#pragma`).

```bash
cd ~/AlpaScalata

# 1. Cleanup of last build
rm -rf build

# 2. Configure
# Note: Added -fopenmp in CXX_FLAGS to activate CPU multithreading
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_ALPAKA=ON \
  -DCMAKE_CUDA_COMPILER="${MY_CUDA_ROOT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${MY_CUDA_ROOT}" \
  -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -DALPAKA_ACC_GPU_CUDA_ENABLED -fopenmp" \
  -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -allow-unsupported-compiler -DALPAKA_ACC_GPU_CUDA_ENABLED"

```

---

## Compilation

Compile both micro-benchmark and MPI executable.

```bash
cmake --build build -j

```

---

## Soundness Check (Micro-Benchmarks)

Check that computation is sound (GPU vs CPU) over single GPU.

**Alpaka (GPU):**

```bash
./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm --check 64

```

*Should print: `CHECK_OK*`

**Native CUDA:**

```bash
./build/bench/micro/benchmark_cuda --config config/cuda_naive_float_small.prm --check 64

```

*Should print: `CHECK_OK*`

---

## Execution Distributed Benchmark (MPI)

Execute benchmark over multiple processes (multi-GPU) using `mpirun`.

**Sintax:** `mpirun -np <NUM_GPU> ./build/src/benchmark_mpi <M> <N> <K>`

1. **Standard Test (Matrix 16k su 2 GPU):**  (note that oversubscribe is required to use 2 gpu)

```bash
mpirun --oversubscribe -n 2 ./build/src/benchmark_mpi 16384 16384 16384

```

2. **Padding Test / Robustness (odd dimensions):**

```bash
mpirun --oversubscribe -np 2 ./build/src/benchmark_mpi 16385 16385 16385

```

*Check that the output is `RESULT: OK`.*

---

## Salvataggio Dati Micro-Benchmarks

Execute the local completed tests by saving the output CSV.

1. **Create result folder:**

```bash
mkdir -p data/results

```

2. **Execute and Save Alpaka (GPU):**

```bash
echo "Running Alpaka Benchmark..."
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_gpu.csv

```

3. **Execute and Save Native CUDA:**

```bash
echo "Running CUDA Benchmark..."
./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm \
  > data/results/cuda_naive_float_gpu.csv

```

---

## Full Workflow (Fast Copy-Paste)

When entering the node, use this whole block:

```bash
# 1. Environment Setup
source /software/spack-v1.0/share/spack/setup-env.sh
spack load cuda
spack load openmpi
export MY_CUDA_ROOT="/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap"
export PATH="${MY_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MY_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}"

# 2. Build (with OpenMP)
cd ~/AlpaScalata
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DENABLE_ALPAKA=ON \
  -DCMAKE_CUDA_COMPILER="${MY_CUDA_ROOT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${MY_CUDA_ROOT}" \
  -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++" \
  -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -DALPAKA_ACC_GPU_CUDA_ENABLED -fopenmp" \
  -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -allow-unsupported-compiler -DALPAKA_ACC_GPU_CUDA_ENABLED"

cmake --build build -j

# 3. Results Folder
mkdir -p data/results

# 4. Distributed Execution (Ex. 2 GPU)
echo ">>> Running MPI Benchmark (16k)..."
mpirun -np 2 ./build/src/benchmark_mpi 16384 16384 16384

# 5. Micro-Benchmarks Execution
echo ">>> Saving Alpaka results..."
./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm > data/results/alpaka_naive_float_gpu.csv

echo ">>> Saving CUDA results..."
./build/bench/micro/benchmark_cuda --config config/cuda_naive_float_small.prm > data/results/cuda_naive_float_gpu.csv

echo "Done! The files are in data/results/"

```

```

```
