# PBS Job Scripts

This directory contains PBS scripts for running benchmarks on the HPC cluster.

## Key Features

- GPU resource requests
- CUDA environment setup via Spack
- Manual logging (PBS does not retain job history)
- Fully reproducible execution

## Typical Workflow

```bash
cd ~/AlpaScalata
. /etc/profile.d/pbs.sh
qsub jobs/bench_naive_gpu.pbs
