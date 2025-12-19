import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/plot/plot_results.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    required = {"Solver","Precision","Size","Time_ms","GFLOPs"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got {set(df.columns)}")

    # Se nel CSV ci sono più solver/precision, li plottiamo separati
    for (solver, prec), g in df.groupby(["Solver","Precision"]):
        outdir = os.path.join("plots", str(solver), str(prec))
        os.makedirs(outdir, exist_ok=True)

        g = g.sort_values("Size")

        # Time
        plt.figure()
        plt.plot(g["Size"], g["Time_ms"])
        plt.xlabel("Size (N for NxN)")
        plt.ylabel("Time (ms)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "time.pdf"), bbox_inches="tight")
        plt.close()

        # GFLOPs
        plt.figure()
        plt.plot(g["Size"], g["GFLOPs"])
        plt.xlabel("Size (N for NxN)")
        plt.ylabel("GFLOPs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "gflops.pdf"), bbox_inches="tight")
        plt.close()

        print(f"Saved plots in: {outdir}")

if __name__ == "__main__":
    main()
