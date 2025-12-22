````md
# 🚀 AlpaScalata — GPU Build & Run Guide (CUDA + Alpaka)

This guide explains **step-by-step** how to configure, build, and run **AlpaScalata** on a **GPU node** of the cluster, enabling **both native CUDA and Alpaka**.

Goals:
- avoid recurring CUDA/path/arch errors
- always build **both benchmark executables** (`benchmark_cuda` and `benchmark_alpaka`)
- provide a single reference for the whole team

---

## 0️⃣ Prerequisites

You must be on a **GPU node** (not the login node).

Verify:
```bash
nvidia-smi
````
You should see at least one GPU (e.g. **NVIDIA L4**).

---

## 1️⃣ CUDA Environment Setup (Spack)

On the cluster, CUDA is provided via **Spack**.

> ⚠️ Use the REAL CUDA path (no `-XXXX` placeholders).

```bash
export CUDA_HOME=/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap
export PATH="$CUDA_HOME/bin:$PATH"
```

Sanity checks:

```bash
nvcc --version
nvidia-smi
```

Optional (recommended) extra check:

```bash
ls -l "$CUDA_HOME/bin/nvcc"
```

---

## 2️⃣ Go to the Project Root

⚠️ All CMake commands must be run from the **repo root**, where `CMakeLists.txt` lives.

```bash
cd ~/AlpaScalata
ls CMakeLists.txt
```

If `CMakeLists.txt` is missing → you are in the wrong directory.

---

## 3️⃣ Clean Previous Build (recommended)

```bash
rm -rf build
```

---

## 4️⃣ Configure with CMake (CUDA + Alpaka)

### GPU architecture

For **NVIDIA L4**, compute capability is **8.9** → set:

```text
CMAKE_CUDA_ARCHITECTURES=89
```

### Full configure command (build BOTH backends + BOTH benchmarks)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_ALPAKA=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

✅ If this step fails, **stop here** and fix the configuration first.

---

## 5️⃣ Build

```bash
cmake --build build -j
```

Expected targets include:

* `gemm_backend_cuda`
* `gemm_backend_alpaka`
* `benchmark_cuda`
* `benchmark_alpaka`

---

## 6️⃣ Verify generated executables

```bash
ls build/bench/micro/
```

You must see:

```text
benchmark_cuda
benchmark_alpaka
```

---

## 7️⃣ Run Alpaka Benchmark (GPU)

### Correctness check (recommended)

```bash
unset LD_LIBRARY_PATH

./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  --check 64
```

Expected output includes:

```text
CHECK_OK
```

### Run benchmark (CSV printed to stdout)

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm
```

---

## 8️⃣ Run Native CUDA Benchmark

### Correctness check

```bash
unset LD_LIBRARY_PATH

./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm \
  --check 64
```

### Run benchmark

```bash
./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm
```

---

## 9️⃣ Save results for plotting

Benchmarks print CSV to **stdout**. Redirect to files under `data/results/`.

```bash
mkdir -p data/results
```

### Alpaka (GPU)

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_gpu.csv
```

### CUDA (GPU)

```bash
./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm \
  > data/results/cuda_naive_float_gpu.csv
```

---

## 🔁 Full Workflow (TL;DR)

```bash
# On a GPU node
export CUDA_HOME=/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap
export PATH="$CUDA_HOME/bin:$PATH"

cd ~/AlpaScalata
rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DENABLE_ALPAKA=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build -j

unset LD_LIBRARY_PATH

./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm --check 64
./build/bench/micro/benchmark_cuda   --config config/cuda_naive_float_small.prm   --check 64

mkdir -p data/results

./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_gpu.csv

./build/bench/micro/benchmark_cuda --config config/cuda_naive_float_small.prm \
  > data/results/cuda_naive_float_gpu.csv
```

