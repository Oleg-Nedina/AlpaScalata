# PBS Job Scripts

This directory contains PBS job scripts for running benchmarks on the HPC cluster.

---

## Features

- GPU resource allocation
- CUDA environment setup via Spack
- Manual logging (PBS job history is not preserved)
- Reproducible benchmark execution

---

## Typical Workflow

```bash
cd ~/AlpaScalata
. /etc/profile.d/pbs.sh
qsub jobs/bench_naive_gpu.pbs

Logs are written explicitly to the user's home directory.
