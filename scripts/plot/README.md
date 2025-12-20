# Plotting Scripts

This directory contains Python scripts to analyze benchmark results.

## `plot_results.py`

- Reads CSV benchmark output
- Groups results by solver and precision
- Generates performance plots in PDF format

## Output Structure

plots/
└── naive/
└── float/
├── time.pdf
└── gflops.pdf


## Usage

```bash
python3 scripts/plot/plot_results.py data/results/naive_float.csv


