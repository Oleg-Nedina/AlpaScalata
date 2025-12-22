### 1.(L'Architettura)

 diviso il codice in **tre componenti logiche distinte**:

1. **L'Interfaccia (`gemm.hpp`)**:
* È il "contratto" che tutti i file devono rispettare.
* Contiene le definizioni delle strutture dati (`GemmShape`) e le firme delle funzioni (`gemm_alpaka_naive`, `gemm_cuda_naive`).
* **Cruciale:** Usa `#ifdef GEMM_ENABLE_ALPAKA` per nascondere gli header pesanti di Alpaka ai file che non ne hanno bisogno (come i solver CUDA puri), evitando conflitti di compilazione.


2. **Il Motore (`gemm_alpaka_naive.cpp`)**:
* Contiene la **logica di calcolo** reale (il Kernel Alpaka).
* È compilato separatamente.
* Usa l'**Istanziazione Esplicita** (`template void ...`) alla fine del file. Questo dice al compilatore: *"Prepara già il codice binario per questo kernel con questi tipi (GPU), così chi lo chiama non deve ricompilarlo"*.


3. **Il Driver di Benchmark (`benchmark_alpaka.cu`)**:
* Non contiene logica di calcolo, ma solo di **orchestrazione**.
* Legge i file di configurazione (`.prm`).
* Gestisce il ciclo di test (`min`, `max`, `step`).
* Esegue il **Warmup** (giri a vuoto per scaldare la GPU).
* Implementa il **Gold Check (`verify_correctness`)**: confronta il risultato della GPU con una versione CPU lenta ma sicura per garantire che i numeri siano giusti.
* Produce output CSV standardizzato.



---

### 2. Perché (Le Soluzioni ai Problemi)


* **Il Problema "Allocator is not a template":**
* *Causa:* NVCC (il compilatore NVIDIA) non riesce a leggere i nuovi header standard C++20 di GCC 13 (in particolare `<string>` e `<memory>`).
* *Soluzione: forzato lo standard **C++17** e aggiunto il flag `-D_GLIBCXX_USE_CXX11_ABI=0`. Questo "calma" il compilatore e garantisce la compatibilità binaria.


* **Il Problema "Namespace Pollution":**
* *Causa:* Avevo incluso `<alpaka/alpaka.hpp>` *dentro* il `namespace gemm`. Questo faceva sì che il compilatore cercasse `std::vector` dentro `gemm::std::vector`, causando errori a cascata.
* *Soluzione:*  spostato gli include **fuori** dai namespace e protetti da macro.


* **Il Problema "Alpaka su CPU invece che GPU":**
* *Causa:* CMake non passava correttamente la definizione al preprocessore, facendo scattare l'`#else` che attivava la CPU.
* *Soluzione:* Abbiamo forzato `-DALPAKA_ACC_GPU_CUDA_ENABLED` direttamente nei flag di compilazione e rimosso il fallback CPU dal codice per essere sicuri che, se compila, usa la GPU.



---

### 3. Perché sarà facilissimo aggiungere nuovi Solver

Grazie a questa struttura, per aggiungere un nuovo solver (es. `tiled` o `shared_memory`), non devi riscrivere il benchmark. Devi solo:

1. **Copiare il file del solver:**
Crea `gemm_alpaka_tiled.cpp` copiando quello `naive`. Cambia solo il codice dentro `operator()` (il kernel).
2. **Registrarlo nell'header:**
Aggiungi una riga in `gemm.hpp`:
```cpp
template <typename TQueue>
void gemm_alpaka_tiled(TQueue &queue, ...);

```


3. **Aggiornare il Benchmark:**
Nel `main` di `benchmark_alpaka.cu`, aggiungi solo un `else if`:
```cpp
if (cfg.solver == "alpaka_naive") {
    gemm::gemm_alpaka_naive(...);
} else if (cfg.solver == "alpaka_tiled") {
    gemm::gemm_alpaka_tiled(...); // <--- Nuova chiamata
}

```


4. **Configurazione:**
Crei un file `.prm` nuovo scrivendo `solver = alpaka_tiled` e lanci il benchmark. Tutto il resto (parsing, CSV, verifica errori, grafici) funzionerà automaticamente senza toccare nulla.
