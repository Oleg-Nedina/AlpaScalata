
# Micro-Benchmarks

This directory contains the main benchmark executable: `benchmark_test`.

## Purpose

- Measure performance of GEMM solvers on square matrices
- Perform scalability studies
- Produce CSV outputs for post-processing

## Benchmark Flow

1. Load configuration from a `.prm` file (mandatory)
2. Optionally verify solver correctness against naive
3. Run multiple timed repetitions
4. Output results as CSV to stdout

## Usage

```bash
./benchmark_test --config configs/parameters.prm [--check N]




Correctness Check
--check N


Runs a solver-vs-naive correctness check on an N × N matrix before benchmarking.

CSV output:

Solver,Precision,Size,Time_ms,GFLOPs
naive,float,256,0.123456,45.67
