import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/plot/plot_results.py <file1.csv> [file2.csv ...]")
        sys.exit(1)

    dfs = []
    # 1. Caricamento e Pulizia Dati
    for path in sys.argv[1:]:
        try:
            # Tenta di leggere il CSV. Se fallisce per righe sporche, prova a saltarle
            try:
                d = pd.read_csv(path)
            except pd.errors.ParserError:
                print(f"Warning: {path} seems malformed. Trying to skip bad lines...")
                d = pd.read_csv(path, on_bad_lines='skip')

            # Controllo se la prima riga è l'header corretto
            if "Solver" not in d.columns:
                print(f"⚠️  Skipping {path}: Header mancante o file sporco. (Controlla la riga 'Accelerator...')")
                continue
                
            # Rimuove spazi dai nomi delle colonne
            d.columns = [c.strip() for c in d.columns]
            dfs.append(d)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not dfs:
        print("❌ Nessun dato valido trovato. Controlla i tuoi file CSV.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("Size")

    base_dir = "plots"
    
    # ---------------------------------------------------------
    # 1. GRAFICI INDIVIDUALI (Una cartella per ogni Solver)
    # ---------------------------------------------------------
    print(f"Generazione grafici individuali in '{base_dir}/Individual/...'")
    
    for (solver, prec), g in df.groupby(["Solver", "Precision"]):
        # Crea cartella es: plots/Individual/alpaka_naive/float/
        out_dir = os.path.join(base_dir, "Individual", solver, prec)
        os.makedirs(out_dir, exist_ok=True)
        
        # --- Grafico Combinato (Dual Axis) ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Matrix Size (N)')
        ax1.set_ylabel('Time (ms)', color=color)
        ax1.plot(g["Size"], g["Time_ms"], color=color, marker='x', label='Time')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('GFLOPs', color=color)  
        ax2.plot(g["Size"], g["GFLOPs"], color=color, marker='o', label='GFLOPs')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Performance: {solver} ({prec})")
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "dual_axis.pdf"))
        plt.close()
        
        # --- Grafici Singoli ---
        # GFLOPs
        plt.figure()
        plt.plot(g["Size"], g["GFLOPs"], marker='o')
        plt.title(f"{solver} GFLOPs")
        plt.xlabel("N")
        plt.ylabel("GFLOPs")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "gflops.pdf"))
        plt.close()

    # ---------------------------------------------------------
    # 2. GRAFICI DI CONFRONTO (Tutti insieme)
    # ---------------------------------------------------------
    print(f"Generazione grafici comparativi in '{base_dir}/Comparison/...'")
    comp_dir = os.path.join(base_dir, "Comparison")
    os.makedirs(comp_dir, exist_ok=True)

    for prec, g_prec in df.groupby("Precision"):
        # Confronto GFLOPs
        plt.figure(figsize=(10, 6))
        for solver, g_sol in g_prec.groupby("Solver"):
            plt.plot(g_sol["Size"], g_sol["GFLOPs"], marker='o', label=solver)
        
        plt.title(f"Comparison: GFLOPs ({prec})")
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("GFLOPs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(comp_dir, f"compare_gflops_{prec}.pdf"))
        plt.close()

        # Confronto Time
        plt.figure(figsize=(10, 6))
        for solver, g_sol in g_prec.groupby("Solver"):
            plt.plot(g_sol["Size"], g_sol["Time_ms"], marker='x', label=solver)
        
        plt.title(f"Comparison: Time ({prec})")
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(comp_dir, f"compare_time_{prec}.pdf"))
        plt.close()

    print("✅ Fatto!")

if __name__ == "__main__":
    main()
