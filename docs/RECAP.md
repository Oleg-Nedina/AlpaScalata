# AlpaScalata — Worklog & Progress Summary (Project-to-Date)

> **Purpose:** This document is the single, high-signal recap of all work completed so far on **AlpaScalata**.  
> It consolidates the “yesterday” Alpaka integration notes (included and refined), plus the CUDA baseline + cluster workflow + benchmarking/plotting pipeline implemented to date.  
>
> **Audience:** future-you, collaborators, reviewers.  
> **Scope:** *what exists*, *why it exists*, *how to run it*, *what decisions were made*, *what’s next*.

---

## 0. Executive Summary

AlpaScalata is an HPC-oriented project to **benchmark and compare dense matrix multiplication (GEMM)** across multiple implementations and backends:

- **CUDA (native)**: baseline naive solver + benchmark harness + cluster execution.
- **Alpaka (portable)**: integrated into the main repository as a first-class backend, with both CPU and GPU paths (as the project evolves).

The project now has a **reproducible pipeline**:

1. Implement solver in `src/` (CUDA or Alpaka) following a shared GEMM API
2. Run correctness check (solver vs reference)
3. Run benchmark sweep using `.prm` config
4. Save results as CSV under `data/results/`
5. Generate **PDF plots** under `plots/<solver>/<precision>/` via Python scripts
6. Execute in a cluster-friendly way with PBS jobs and robust logging

The repository structure is now stable and ready for the “real work”: **2–3 optimized solvers**, Alpaka equivalents, and systematic comparisons.

---

## 1. Repository Organization (Current)

The repo is organized to keep solver code, benchmarking, tests, cluster jobs, and plotting separate:

```

AlpaScalata/
├── include/                 # Public API (GEMM headers, shape structs)
├── src/                     # Solver implementations (no I/O, no timing)
│   └── backends/
│       ├── cuda/            # Native CUDA solvers (naive baseline + future optimized)
│       └── alpaka/          # Alpaka solvers (naive + future optimized)
├── bench/                   # Benchmark executables (micro-benchmarks)
│   └── micro/
├── tests/                   # Unit tests (correctness, small/deterministic)
│   └── unit/
├── configs/                 # Config files (.prm) defining experiments
├── data/                    # Raw benchmark results (CSV)
│   └── results/
├── plots/                   # Generated plots (PDF)
├── scripts/                 # Utilities, plotting scripts
│   └── plot/
├── jobs/                    # PBS job scripts for cluster runs
└── docs/                    # Documentation (this worklog lives here)

````

**Rule of thumb:**
- If you want to change performance → edit `src/`
- If you want to change experiments → edit `configs/`
- If you want to change measurements/output format → edit `bench/`
- If you want plots → edit `scripts/plot/`

---

## 2. CUDA Baseline (Naive) — Implemented and Working

### 2.1 GEMM API Unification

A shared GEMM API is defined in `include/gemm/gemm.hpp`.  
All solvers (CUDA/Alpaka/CPU) must conform to it.

**Design intent:**
- Benchmark harness should call solvers through a consistent interface
- Correctness checks can compare solvers without special cases
- Adding new solvers should require minimal wiring (register + dispatch)

### 2.2 CUDA Naive Solver

A baseline CUDA kernel exists (naive GEMM):
- one thread computes one element of `C`
- row-major layout
- used as **baseline performance** and as **golden reference** for correctness checks

Implemented in:
- `src/backends/cuda/gemm_cuda_naive.cu`

---

## 3. Benchmark Harness — Config-driven, CSV output, correctness option

### 3.1 Benchmark executable

A micro-benchmark driver exists in:
- `bench/micro/benchmark_test.cu`

Key properties:
- reads config from `.prm` (**mandatory**)
- optional correctness check vs naive via `--check N`
- prints performance results as **CSV to stdout** (redirect to file)
- intended to support multiple solvers via dispatch

### 3.2 `.prm` configuration files

Configs live in:
- `configs/`

Example:
```ini
solver = naive
precision = float

min = 256
max = 4096
step = 256

reps = 20
warmup = 5
seed = 123
````

Why `.prm`:

* reproducibility
* clean PBS scripts
* easy sweeps
* versioned experiments

### 3.3 Output format

Benchmark prints CSV such as:

```
Solver,Precision,Size,Time_ms,GFLOPs
naive,float,256,0.123456,45.67
...
```

**Policy:**

* CSV goes to **stdout**
* logs and check messages go to **stderr**
  This keeps output redirect safe and clean.

---

## 4. Plotting Pipeline — Python → PDF

A plotting script exists:

* `scripts/plot/plot_results.py`

Behavior:

* reads CSV
* expects columns: `Solver`, `Precision`, `Size`, `Time_ms`, `GFLOPs`
* generates plots in **PDF format** (publication-friendly)
* auto-creates directories:

  * `plots/<solver>/<precision>/time.pdf`
  * `plots/<solver>/<precision>/gflops.pdf`

**Note:** plots are derived artifacts; usually not versioned.

---

## 5. Cluster Execution (PBS) — End-to-End Working

### 5.1 Access and environment setup

Workflow used successfully:

1. Connect VPN (Polimi)
2. SSH to login node:

```bash
ssh u10905938@10.78.18.100
```

3. Enable PBS tools:

```bash
. /etc/profile.d/pbs.sh
```

4. Interactive GPU job for debug:

```bash
qsub -I -q gpu -l select=1:ncpus=2:ngpus=1 -l walltime=00:10:00
```

5. Load CUDA via Spack (on GPU node):

```bash
source /software/spack-v1.0/share/spack/setup-env.sh
spack load cuda
```

6. Verify:

```bash
which nvcc
nvcc --version
nvidia-smi
```

### 5.2 Compile on GPU node (working example)

```bash
nvcc -O3 -std=c++17 -Iinclude -arch=sm_89 \
  bench/micro/benchmark_test.cu \
  src/backends/cuda/gemm_cuda_naive.cu \
  -o benchmark_test
```

### 5.3 Run benchmark (interactive)

```bash
mkdir -p data/results
./benchmark_test --config configs/parameters.prm --check 256 \
  > data/results/naive_float.csv
```

### 5.4 PBS batch jobs and logging

Jobs must be submitted from `login01` (not from compute nodes).
The following was validated:

```bash
cd ~/AlpaScalata
qsub jobs/test_cuda_naive.pbs
```

**Important cluster detail:** PBS is not configured to keep job history, so jobs disappear quickly after completion.
Therefore, robust logging is required.

**Best practice implemented:** manual log redirection in PBS scripts:

```bash
LOG="/home/u10905938/<jobname>_${PBS_JOBID}.log"
exec > >(tee -a "$LOG") 2>&1
```

Read logs:

```bash
tail -f /home/u10905938/<jobname>_<JOBID>.log
```

Monitor:

```bash
qstat -u $USER
```

Cancel:

```bash
qdel <JOBID>
```

---

## 6. Git + Cluster realities (what was encountered and solved)

### 6.1 GitHub clone issues (SSH keys)

At some point, cloning via SSH failed due to missing SSH key permissions:

* `Permission denied (publickey).`

Resolution in practice:

* use HTTPS clone for cluster environments (recommended), or
* configure SSH keys properly on the cluster account

This is now considered part of the “cluster onboarding checklist”.

### 6.2 Container attempt was not available

An initial attempt to run an NVHPC container via Apptainer failed because the `.sif` path was not present.
Solution adopted:

* use the cluster’s **Spack-provided CUDA toolchain** (`spack load cuda`)
* rely on native environment rather than missing container images

---

## 7. Alpaka Integration (Worklog: consolidated and refined)

> This section incorporates and refines the “yesterday” Alpaka worklog you provided.

### 7.1 Initial Situation

At the beginning of Alpaka work:

* Alpaka experiments lived in a separate `alpaka_test/` directory (playground/smoke-test)
* Build logic was fragmented and ad-hoc
* CUDA runtime issues required manual fixes (historically)
* No single unified way to:

  * select backend (CUDA vs Alpaka)
  * run benchmarks consistently
  * share commands with collaborators

### 7.2 Goals Achieved (Milestones Completed)

*  Alpaka integrated into the main project structure
*  Legacy scaffolding identified as removable noise
*  Single coherent build philosophy established (project-wide)
*  Benchmark logic kept backend-agnostic and config-driven
*  Cluster execution validated (GPU nodes + CUDA toolchain + PBS)

### 7.3 Backend layout after integration

Target structure (current or planned, depending on implementation stage):

```text
src/backends/
├── cuda/
│   ├── gemm_cuda_naive.cu
│   └── (future) optimized CUDA solvers
└── alpaka/
    ├── gemm_alpaka_naive.cpp
    └── kernels/
```

**Key constraint:** each backend must implement the same public GEMM API.

### 7.4 Recommendation: remove `alpaka_test/`

`alpaka_test/` was confirmed safe to delete because:

* it was only a playground / smoke-test
* it is not referenced by current build logic
* keeping it adds historical noise

Safe removal:

```bash
rm -rf alpaka_test
```

---

## 8. Correctness Strategy (Project-wide)

Correctness is treated as a gate, not an afterthought:

* Naive CUDA is the **reference** for GPU-family solvers
* New solvers must pass correctness check:

  * compare solver output to naive for a small N (e.g., 64/128/256)
  * fail fast if mismatch

This strategy enables:

* safe optimization iterations
* confidence in comparative performance results
* quick regression detection

---

## 9. What We Have Right Now (Current Capabilities)

###  Working

* CUDA toolchain via Spack on cluster (`spack load cuda`)
* CUDA naive compilation and execution on GPU nodes
* PBS job submission (from login node) and monitoring
* Robust logging strategy for PBS (manual log redirection)
* Benchmark harness with:

  * `.prm` configuration model
  * CSV output
  * optional correctness checks
* Plotting pipeline producing **PDF** plots with consistent folder structure

### “In progress / Next”

* additional CUDA solvers (tiled/shared memory, etc.)
* Alpaka solvers and integration into the same benchmark harness
* CPU baseline / quick CPU optimizations
* full comparative benchmark sweeps across solvers/backends

---

## 10. Next Steps (Short, High-Impact)

1. **CUDA optimized solvers (2–3 total)**
   Suggested minimal set:

   * shared-memory tiled GEMM
   * a variant with dynamic tiling or scheduling (or vectorized loads)

2. **Alpaka parity with CUDA**

   * Alpaka naive
   * at least one Alpaka “tiled-like” strategy

3. **Repository cleanup**

   * remove placeholders and empty files not needed
   * ensure all READMEs are consistent and useful

4. **Systematic comparisons**

   * same configs, same metrics, consistent plots
   * answer: “Is mid-level CUDA optimization enough to saturate the GPU?”

5. **Fast CPU baseline**

   * naive + cache-blocking baseline
   * contextualize when GPU acceleration wins

---

## 11. Commands Cheat Sheet (Minimal)

### Cluster access

```bash
ssh u10905938@10.78.18.100
. /etc/profile.d/pbs.sh
```

### Interactive GPU node

```bash
qsub -I -q gpu -l select=1:ncpus=2:ngpus=1 -l walltime=00:10:00
source /software/spack-v1.0/share/spack/setup-env.sh
spack load cuda
```

### Compile (CUDA benchmark)

```bash
nvcc -O3 -std=c++17 -Iinclude -arch=sm_89 \
  bench/micro/benchmark_test.cu \
  src/backends/cuda/gemm_cuda_naive.cu \
  -o benchmark_test
```

### Run benchmark

```bash
mkdir -p data/results
./benchmark_test --config configs/parameters.prm --check 256 \
  > data/results/naive_float.csv
```

### Plot

```bash
python3 scripts/plot/plot_results.py data/results/naive_float.csv
```

### PBS batch (submit from login)

```bash
qsub jobs/<job>.pbs
qstat -u $USER
tail -f /home/u10905938/<job>_<JOBID>.log
```

---

## 12. One-line Takeaway

> The project has been transformed from ad-hoc experiments into a **reproducible, cluster-ready GEMM benchmarking framework**, with a working CUDA baseline and a stable pipeline ready for meaningful optimization and Alpaka comparisons.
```
