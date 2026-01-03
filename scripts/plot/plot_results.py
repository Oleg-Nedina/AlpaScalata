
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLS = ["Solver", "Precision", "Size", "Time_ms", "GFLOPs"]

def read_one_csv(path: str) -> pd.DataFrame | None:
    try:
        try:
            d = pd.read_csv(path)
        except pd.errors.ParserError:
            print(f"Warning: {path} seems malformed. Trying to skip bad lines...")
            d = pd.read_csv(path, on_bad_lines="skip")

        # clean column names
        d.columns = [c.strip() for c in d.columns]

        if "Solver" not in d.columns:
            print(f"  Skipping {path}: header mancante o file sporco.")
            return None

        # keep only needed columns
        missing = [c for c in REQUIRED_COLS if c not in d.columns]
        if missing:
            print(f"  Skipping {path}: colonne mancanti {missing}")
            return None

        d = d[REQUIRED_COLS].copy()

        # normalize string values
        d["Solver"] = d["Solver"].astype(str).str.strip()
        d["Precision"] = d["Precision"].astype(str).str.strip()

        d["Solver"] = d["Solver"].str.lower()
        d["Precision"] = d["Precision"].str.lower()

        d["Size"] = pd.to_numeric(d["Size"], errors="coerce")
        d["Time_ms"] = pd.to_numeric(d["Time_ms"], errors="coerce")
        d["GFLOPs"] = pd.to_numeric(d["GFLOPs"], errors="coerce")

        before = len(d)
        d = d.dropna(subset=["Size", "Time_ms", "GFLOPs"])
        d = d[d["Size"] > 0]
        if len(d) == 0:
            print(f"  Skipping {path}: no valid row after cleaning.")
            return None
        if len(d) < before:
            print(f"  {path}: dropped {before - len(d)} invalid rows.")

        return d

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/plot/plot_results.py <file1.csv> [file2.csv ...]")
        sys.exit(1)

    dfs = []
    for path in sys.argv[1:]:
        d = read_one_csv(path)
        if d is not None:
            dfs.append(d)

    if not dfs:
        print("No valid data found. Check CSV files.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("Size")

    base_dir = "plots"

    # ---------------------------------------------------------
    # 1) INDIVIDUAL GRAPHS: 3 graphs for (Solver, Precision)
    # ---------------------------------------------------------
    print(f"Generation individual graphs in '{base_dir}/Individual/...'")

    for (solver, prec), g in df.groupby(["Solver", "Precision"], dropna=False):
        g = g.sort_values("Size")

        out_dir = os.path.join(base_dir, "Individual", solver, prec)
        os.makedirs(out_dir, exist_ok=True)

        # --- Dual Axis ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Matrix Size (N)")
        ax1.set_ylabel("Time (ms)")
        ax1.plot(g["Size"], g["Time_ms"], marker="x", label="Time")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.set_ylabel("GFLOPs")

        ax2.plot(
                    g["Size"],
                    g["GFLOPs"],
                    marker="o",
                    color="red",
                    label="GFLOPs"
        )
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title(f"Performance: {solver} ({prec})")
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "dual_axis.pdf"))
        plt.close(fig)

        # --- Time solo ---
        plt.figure(figsize=(10, 6))
        plt.plot(g["Size"], g["Time_ms"], marker="x")
        plt.title(f"{solver} Time ({prec})")
        plt.xlabel("N")
        plt.ylabel("Time (ms)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "time.pdf"))
        plt.close()

        # --- GFLOPs solo ---
        plt.figure(figsize=(10, 6))
        plt.plot(g["Size"], g["GFLOPs"], marker="o")
        plt.title(f"{solver} GFLOPs ({prec})")
        plt.xlabel("N")
        plt.ylabel("GFLOPs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gflops.pdf"))
        plt.close()

        print(f"  {solver}/{prec}: dual_axis.pdf, time.pdf, gflops.pdf")

    # ---------------------------------------------------------
    # 2) COMPARATIVE GRAPHS
    # ---------------------------------------------------------
    print(f"Generation comparative graphs in '{base_dir}/Comparison/...'")
    comp_dir = os.path.join(base_dir, "Comparison")
    os.makedirs(comp_dir, exist_ok=True)

    for prec, g_prec in df.groupby("Precision"):
        g_prec = g_prec.sort_values("Size")

        # GFLOPs comparison
        plt.figure(figsize=(10, 6))
        for solver, g_sol in g_prec.groupby("Solver"):
            g_sol = g_sol.sort_values("Size")
            plt.plot(g_sol["Size"], g_sol["GFLOPs"], marker="o", label=solver)

        plt.title(f"Comparison: GFLOPs ({prec})")
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("GFLOPs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, f"compare_gflops_{prec}.pdf"))
        plt.close()

        # Time comparison
        plt.figure(figsize=(10, 6))
        for solver, g_sol in g_prec.groupby("Solver"):
            g_sol = g_sol.sort_values("Size")
            plt.plot(g_sol["Size"], g_sol["Time_ms"], marker="x", label=solver)

        plt.title(f"Comparison: Time ({prec})")
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, f"compare_time_{prec}.pdf"))
        plt.close()

        print(f"  Comparison/{prec}: compare_gflops_{prec}.pdf, compare_time_{prec}.pdf")

    print("\n--- DEBUG found groups ---")
    print(df.groupby(["Solver", "Precision"]).size().sort_values(ascending=False).to_string())

    print("\nDone!")

if __name__ == "__main__":
    main()
