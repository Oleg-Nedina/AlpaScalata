### Appendix A – Cluster Access & CUDA Execution (Operational Recap)

This section summarizes **all commands required to access the cluster, request GPU resources, and run CUDA workloads**.
It is meant to avoid command hunting and to serve as a **single operational reference**.

---

### A.1 VPN Connection (Required)

Before accessing the cluster, connect to the **Politecnico di Milano VPN**.

* Use Cisco AnyConnect / OpenConnect
* Verify connectivity:

```bash
ping 10.78.18.100
```

If the host responds, the VPN is active.

---

### A.2 Login to the Cluster

```bash
ssh u10905938@10.78.18.100
```

You are now on the **login node** (`login01`).

**Do NOT run CUDA code or heavy compilation on the login node.**

---

### A.3 Enable PBS Commands

PBS utilities are not available by default.

```bash
. /etc/profile.d/pbs.sh
```

This enables:

* `qsub`
* `qstat`
* `qdel`

---

### A.4 Interactive GPU Session (Development & Debug)

Use this mode for:

* compiling CUDA code
* quick testing
* debugging

#### Request a GPU interactively

```bash
qsub -I -q gpu -l select=1:ncpus=2:ngpus=1 -l walltime=00:10:00
```

When ready, PBS prints:

```text
qsub: job <ID>.login01 ready
```

You are now on a **GPU compute node** (e.g. `gpu01`).

---

### A.5 Load CUDA Environment (via Spack)

On the GPU node:

```bash
source /software/spack-v1.0/share/spack/setup-env.sh
spack load cuda
```

Verify:

```bash
which nvcc
nvcc --version
nvidia-smi
```

---

### A.6 Compile CUDA Code (Example)

```bash
cd ~/AlpaScalata

nvcc -O3 -std=c++17 -Iinclude -arch=sm_89 \
  bench/micro/benchmark_test.cu \
  src/backends/cuda/gemm_cuda_naive.cu \
  -o benchmark_test
```

---

### A.7 Run CUDA Benchmark (Interactive)

```bash
mkdir -p data/results

./benchmark_test \
  --config configs/parameters.prm \
  --check 256 \
  > data/results/naive_float.csv
```

Exit the GPU node:

```bash
exit
```

---

### A.8 Batch Execution via PBS (Recommended for Benchmarks)

For long or systematic runs, use **batch jobs**.

#### Submit a job (from `login01`)

```bash
cd ~/AlpaScalata
qsub jobs/bench_naive_gpu.pbs
```

PBS returns a job ID:

```text
5169.login01
```

---

### A.9 Monitor Jobs

```bash
qstat -u $USER
```

Job states:

* `Q` → queued
* `R` → running
* `E` → exiting (finishing)

PBS does **not** preserve job history after completion.

---

### A.10 Logging (Critical)

Because PBS job history is not retained, **manual logging is mandatory**.

In PBS scripts, logs are redirected explicitly, e.g.:

```bash
LOG="/home/u10905938/bench_naive_${PBS_JOBID}.log"
exec > >(tee -a "$LOG") 2>&1
```

Read logs:

```bash
tail -f /home/u10905938/bench_naive_<JOBID>.log
```

or

```bash
less /home/u10905938/bench_naive_<JOBID>.log
```

---

### A.11 Cancel a Job

```bash
qdel <JOBID>
```

Example:

```bash
qdel 5169
```

---

### A.12 Operational Rules (Summary)

*  Never run CUDA or heavy compilation on `login01`
*  Use interactive jobs for debugging
*  Use batch jobs for benchmarks
*  Always load CUDA via Spack
*  Always redirect benchmark output to CSV
*  Always log explicitly (PBS does not keep history)

---

### A.13 Minimal Command Cheat Sheet

```bash
# Login
ssh u10905938@10.78.18.100
. /etc/profile.d/pbs.sh

# Interactive GPU
qsub -I -q gpu -l select=1:ncpus=2:ngpus=1 -l walltime=00:10:00
source /software/spack-v1.0/share/spack/setup-env.sh
spack load cuda

# Batch
qsub jobs/bench_naive_gpu.pbs
qstat -u $USER
tail -f ~/bench_naive_<JOBID>.log
```

---

# AlpaScalata – How to Build & Run (Local + Cluster)

This document is the **single entry point** to understand how to build and run the AlpaScalata project.

It assumes:

* detailed READMEs already exist in subdirectories
* this file is a **practical recap**, not a design document

If you only read one file to run the project correctly, **read this one**.

---

## 1. What This Project Does (Short)

AlpaScalata benchmarks **GEMM (matrix multiplication)** implementations across:

* **Native CUDA**
* **Alpaka (CPU and GPU)**

The workflow is **config-driven** and **cluster-oriented**:

> build → run benchmark with `.prm` → redirect CSV → plot offline

---

## 2. Repository Structure (Mental Map)

```text
src/        → GEMM implementations (no I/O, no timing)
bench/      → benchmark executables
tests/      → correctness checks (small & deterministic)
config/     → .prm files defining experiments
data/       → raw CSV benchmark output
plots/      → generated plots (PDF)
scripts/    → plotting & post-processing
jobs/       → cluster (PBS) scripts
external/   → third-party deps (alpaka as submodule)
docs/       → documentation (this file lives here)
```

**Rule of thumb**

> If you want to *change performance*, edit `src/`
> If you want to *change experiments*, edit `config/`

---

## 3. Dependencies

### Required

* CMake ≥ 3.20
* C++17 compiler
* Git (with submodules)

### For CUDA / Alpaka GPU

* CUDA Toolkit (cluster or local)
* `nvcc` available

Alpaka is included as a **git submodule**.

---

## 4. Getting the Code

```bash
git clone --recursive https://github.com/<repo>/AlpaScalata.git
cd AlpaScalata
```

If the submodule was not fetched:

```bash
git submodule update --init --recursive
```

---

## 5. Build Configurations Overview

The build supports **multiple backends in one CMake project**.

Backends are enabled via flags:

| Flag                                | Meaning             |
| ----------------------------------- | ------------------- |
| `ENABLE_CUDA`                       | Native CUDA backend |
| `ENABLE_ALPAKA`                     | Alpaka backend      |
| `alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE` | Alpaka CPU          |
| `alpaka_ACC_GPU_CUDA_ENABLE`        | Alpaka GPU          |

---

## 6. Local Build – Alpaka CPU Only (Fast & Safe)

This is the **recommended starting point** for development and correctness.

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=OFF \
  -DENABLE_ALPAKA=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON

cmake --build build -j
```

This builds:

* Alpaka CPU GEMM backend
* `benchmark_alpaka`

---

## 7. Running Alpaka CPU Benchmark

### Correctness check (recommended)

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  --check 64
```

If everything is correct:

```text
CHECK_OK
```

### Benchmark run

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm
```

### Save results

Benchmarks **write CSV to stdout**.
You must redirect manually.

```bash
mkdir -p data/results

./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_cpu.csv
```

---

## 8. Cluster Build – CUDA + Alpaka GPU

### Load CUDA (example via Spack)

```bash
export CUDA_HOME=/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-XXXX
```

### Configure & build

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_ALPAKA=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_HOME

cmake --build build -j
```

**Important**

* CUDA runtime paths are embedded via **RPATH**
* You do **not** need `LD_LIBRARY_PATH` at runtime

---

## 9. Running Alpaka GPU Benchmark

Always unset `LD_LIBRARY_PATH` to ensure RPATH correctness:

```bash
unset LD_LIBRARY_PATH
```

### Correctness check

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  --check 64
```

### Benchmark run

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_gpu.csv
```

---

## 10. Configuration Files (`.prm`)

Benchmarks **will not run without a config file**.

Example:

```ini
solver = naive
precision = float

min = 256
max = 2048
step = 256

reps = 10
warmup = 5
seed = 123
```

Why configs matter:

* reproducibility
* clean cluster jobs
* easy parameter sweeps
* version-controlled experiments

---

## 11. Output Data

* Benchmarks produce **CSV only**
* No plots are generated during execution
* Raw data lives in:

```text
data/results/
```

Example:

```text
alpaka_naive_float_cpu.csv
alpaka_naive_float_gpu.csv
```

---

## 12. Plotting Results

```bash
python3 scripts/plot/plot_results.py data/results/
```

Outputs:

```text
plots/
└── <solver>/<precision>/
    ├── time.pdf
    └── gflops.pdf
```

Plots are **derived artifacts** and usually not versioned.

---

## 13. Typical Daily Workflow

```text
1. Edit or add solver in src/
2. Build (CPU or GPU)
3. Run benchmark with .prm
4. Redirect CSV to data/results/
5. Plot offline
```

---

## 14. Common Pitfalls

*  Running benchmark without redirect → data lost
*  Using large matrices with naive GEMM → very slow
*  Forgetting `--check` after changes
*  Using `LD_LIBRARY_PATH` instead of RPATH

---

## 15. TL;DR Cheat Sheet

### CPU (local)

```bash
cmake -S . -B build -DENABLE_CUDA=OFF -DENABLE_ALPAKA=ON
cmake --build build
./build/bench/micro/benchmark_alpaka --config config/xxx.prm
```

### GPU (cluster)

```bash
cmake -S . -B build -DENABLE_CUDA=ON -DENABLE_ALPAKA=ON
cmake --build build
unset LD_LIBRARY_PATH
./build/bench/micro/benchmark_alpaka --config config/xxx.prm
```

---

## Final Note

This file is meant to be:

* **stable**
* **shared with collaborators**
* **the only place to look for commands**

If something changes, update **this file first**.


