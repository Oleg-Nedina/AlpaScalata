# scripts/plot/plot_results.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/plot/plot_results.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    # sanity
    required = {"Size", "Time_ms", "GFLOPs"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got {set(df.columns)}")

    os.makedirs("plots", exist_ok=True)

    # Time plot
    plt.figure()
    plt.plot(df["Size"], df["Time_ms"])
    plt.xlabel("Size (N for NxN)")
    plt.ylabel("Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    out1 = os.path.join("plots", "time.png")
    plt.savefig(out1, dpi=150)
    plt.close()

    # GFLOPs plot
    plt.figure()
    plt.plot(df["Size"], df["GFLOPs"])
    plt.xlabel("Size (N for NxN)")
    plt.ylabel("GFLOPs")
    plt.grid(True)
    plt.tight_layout()
    out2 = os.path.join("plots", "gflops.png")
    plt.savefig(out2, dpi=150)
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

if __name__ == "__main__":
    main()

