# 🚀 AlpaScalata: High-Performance Distributed GEMM

Questo progetto implementa un solver per la moltiplicazione di matrici () che è **Distribuito**, **Portabile** e **Altamente Ottimizzato**.

## 🏗 1. Architettura di Alto Livello

Il sistema utilizza un approccio ibrido **MPI + Alpaka**:

* **MPI (Message Passing Interface):** Gestisce la comunicazione tra nodi e multi-GPU. Divide il lavoro (Decomposizione 1D su righe) distribuendo porzioni della matrice  e la matrice  completa a tutti i worker.
* **Alpaka:** Fornisce un livello di astrazione hardware. Il codice è scritto in C++ standard ma viene compilato per il backend specifico (CUDA nel nostro caso), mantenendo la portabilità su CPU o altre GPU (AMD/Intel).

---

## 🧠 2. Host-Side: Gestione Memoria e Padding (`gemm_mpi.cpp`)

Il collo di bottiglia principale per le ottimizzazioni spinte è l'allineamento della memoria. Per risolvere crash su dimensioni dispari, abbiamo implementato un sistema di **Logical Padding**.

### Logica del Padding

La GPU lavora meglio con blocchi da 128-bit (4 float).

1. **Input:** L'utente chiede una matrice  (es. 16385).
2. **Trasformazione:** L'Host arrotonda le dimensioni al multiplo di 4 superiore.


3. **Allocazione:** Si allocano buffer più grandi riempiti di zeri nella "cornice" esterna.
4. **Vantaggio:** Il Kernel GPU può usare istruzioni vettoriali (`float4`) in sicurezza senza `if` costosi per i bordi.

### Flusso MPI

1. **Master:** Genera i dati e applica il padding.
2. **Scatter:** Invia chunk di  ai worker (ogni worker riceve  righe).
3. **Broadcast:** Invia tutta  ai worker.
4. **Compute:** Ogni worker calcola la sua  parziale.
5. **Gather:** Il Master raccoglie i risultati e scarta il padding in fase di output/verifica.

---

## ⚡ 3. Alpaka Pipeline: Asincronia e Batching (`gemm_alpaka_full_options.cpp`)

Per matrici enormi che non stanno in VRAM, o per nascondere la latenza del PCI-Express, usiamo una **Pipeline Asincrona a 3 Stadi**.

### Strategia "Out-of-Core"

La matrice viene divisa in **Tile** (es. ).
Invece di elaborare tutto in serie, usiamo **3 Stream Indipendenti** (Code `NonBlocking`):

| Stream 1 | Stream 2 | Stream 3 |
| --- | --- | --- |
| **Calcola** Tile  | **Upload** Tile  | **Download** Tile  |

In questo modo, mentre la GPU macina numeri, il bus PCI-E sta già portando i dati successivi.

### Struttura `StreamContext`

Ogni stream ha il suo contesto dedicato per evitare race condition:

* `alpaka::Queue<..., NonBlocking>`: La coda di comandi.
* `alpaka::Buf`: Buffer pinned dedicati per quel chunk.

---

## 🔥 4. Il Kernel "Ultimate": Ottimizzazioni CUDA

Questa è la parte che ha portato le performance da 0.5 a **2.7+ TFLOPS**.

### A. Thread Coarsening (2D Register Tiling)

Invece di far calcolare 1 pixel a 1 thread (inefficiente), ogni thread calcola un **blocco 4x4 (16 pixel)**.

* **Macro-Tile (Shared Memory):** .
* **Micro-Tile (Registri):** .
* **Threads per Blocco:**  Thread.

**Vantaggio:** Aumenta l'intensità aritmetica. Ogni dato caricato dai registri viene riutilizzato più volte per calcolare i 16 risultati parziali.

### B. Vectorized Global Loads (`float4`)

Il collo di bottiglia principale era la banda di memoria globale.

* **Prima:** Caricamento scalare (`float`). 1 richiesta per elemento.
* **Dopo:** Caricamento vettoriale (`float4`). 1 richiesta per 4 elementi.

```cpp
// Esempio concettuale del Kernel
using float4 = ::float4;
// Carica 128 bit in un colpo solo
float4 loaded = *reinterpret_cast<const float4*>(&A[globalRow * K + globalCol]);
// Scrive in Shared Memory
*reinterpret_cast<float4*>(&As[vecRow][col]) = loaded;

```

*Grazie al Padding lato Host, questo cast è sempre sicuro e allineato.*

### C. Loop Unrolling

Usiamo `#pragma unroll` sui loop interni (quelli che scorrono sui 4x4 registri). Questo permette al compilatore NVCC di:

1. Rimuovere il costo del controllo del loop (`i++`, `i < 4`).
2. Mettere in pipeline le istruzioni FMA (Fused Multiply-Add).

### D. Shared Memory in Alpaka

Per gestire la memoria condivisa in modo portabile (senza usare `__shared__` che rompe la compatibilità CPU), usiamo:

```cpp
float (&As)[BM][BK] = alpaka::declareSharedVar<float[BM][BK], __COUNTER__>(acc);

```

---

## 📊 Riepilogo Performance

L'evoluzione del codice ha seguito questi step:

| Versione Kernel | Caratteristiche | Performance (L4 GPU) | Collo di Bottiglia |
| --- | --- | --- | --- |
| **Naive** | 1 Thread = 1 Pixel, Global Mem diretta | ~0.5 TFLOPS | Latenza Memoria |
| **Tiled** | Uso base Shared Memory | ~1.1 TFLOPS | Banda Memoria |
| **Coarsened** | 1 Thread = 16 Pixel, Scalar Load | ~1.4 TFLOPS | Load Efficiency |
| **Ultimate** | **Coarsened + Vectorized float4** | **~2.7 TFLOPS** | Compute Bound (FP32) |

## 🛠 Comandi di Build & Run

**Compilazione:**

```bash
cmake --build build -j

```

**Esecuzione (Matrice  su 2 GPU):**

```bash
mpirun -np 2 ./build/src/benchmark_mpi 16384 16384 16384

```

**Test Robustezza Padding (Dimensione dispari):**

```bash
mpirun -np 2 ./build/src/benchmark_mpi 16385 16385 16385


```
