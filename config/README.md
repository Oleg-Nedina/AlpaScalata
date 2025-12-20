# Benchmark Configuration Files
 
This directory contains `.prm` files used to configure benchmarks.

## File Format

Simple key-value format:

solver = naive
precision = float

min = 256
max = 4096
step = 256

reps = 20
warmup = 5
seed = 123

---

### Why .prm?

1) Reproducibility

2) Cleaner PBS scripts

3) Easy parameter sweeps

4) Version-controlled experiments

# The benchmark will not run without a configuration file.
