# 🚀 AlpaScalata: High-Performance Distributed GEMM

**AlpaScalata** è un engine per la moltiplicazione di matrici () progettato per essere scalabile (Multi-GPU), portabile (Hardware Agnostic) e performante (vicino ai limiti teorici dell'hardware in FP32).

---

## 🎨 1. Filosofia di Design

Il progetto si basa su tre pilastri fondamentali per risolvere i problemi classici dell'HPC:

### A. Portabilità "Single-Source" (Alpaka)

Invece di scrivere codice CUDA puro (che gira solo su NVIDIA) o HIP (solo su AMD), utilizziamo **Alpaka**.

* **Idea:** Scrivere il kernel in C++ astratto una sola volta.
* **Risultato:** Il codice può essere compilato per backend CUDA, HIP, OpenMP (CPU) o TBB senza cambiare una virgola della logica matematica.

### B. Scalabilità Orizzontale (MPI)

Una singola GPU ha memoria limitata (es. 24GB). Per calcolare matrici enormi (es. ), il calcolo deve essere distribuito.

* **Idea:** Decomposizione 1D spaziale.
* **Risultato:** Ogni nodo MPI gestisce una fetta orizzontale di  e una copia completa di . La capacità di memoria e calcolo scala linearmente con il numero di nodi.

### C. "Safety through Architecture" (Padding)

Le ottimizzazioni hardware (Vectorization) richiedono allineamento di memoria perfetto. Gestire i "bordi" (casi limite) dentro il kernel GPU costa performance (divergenza dei warp).

* **Idea:** Spostare la complessità dal Kernel (GPU) all'Host (CPU).
* **Risultato:** La GPU lavora sempre su dati perfetti. I casi limite sono gestiti preparando i dati a monte.

---

## 🛡 2. Gestione del Padding (Il Segreto della Stabilità)

Questa è la componente critica che permette di usare istruzioni vettoriali (`float4`) senza crashare su dimensioni "strane" (es. 16385).

### Il Problema

Le GPU moderne leggono memoria in blocchi da 128-bit (4 float). Se una riga della matrice finisce in un indirizzo non multiplo di 16 byte, o se la matrice ha una dimensione dispari, un caricamento vettoriale causerebbe un **Segmentation Fault** o leggerebbe memoria sporca.

### La Soluzione: "Logical Padding"

Disaccoppiamo le dimensioni **Reali** (dell'utente) dalle dimensioni **Fisiche** (allocate in memoria).

1. **Arrotondamento:**
Ogni dimensione () viene arrotondata al multiplo di 4 superiore.



*(Nel codice usiamo anche multipli del Tile Size per sicurezza estrema, ma 4 è il minimo sindacale).*
2. **La "Cornice":**
Immagina la matrice reale come una foto . Noi allochiamo una cornice .
* La zona valida () contiene i dati.
* La zona di padding () viene riempita di **Zeri (0.0f)**.


3. **Matematica Invariante:**
Poiché  e , il calcolo esteso alla zona di padding non influenza il risultato nella zona valida.

### Implementazione Host (`gemm_mpi.cpp`)

```cpp
// 1. Calcolo dimensione sicura
int M_pad = (M_real + 3) / 4 * 4;

// 2. Allocazione Padded (Init a 0.0f fondamentale!)
std::vector<float> h_A(M_pad * K_pad, 0.0f);

// 3. Riempimento dati validi
for(int i=0; i<M_real; ++i)
    for(int j=0; j<K_real; ++j)
        h_A[i * K_pad + j] = val; // Nota lo stride K_pad!

```

---

## 🔄 3. Pipeline Asincrona (Alpaka Host-Side)

Per massimizzare l'uso della GPU, non aspettiamo che i dati arrivino. Usiamo una pipeline a 3 stadi ("Double/Triple Buffering").

### Architettura Out-of-Core

La matrice locale viene divisa in **Chunk** (es. ).
Vengono creati **3 Stream** (code comandi asincrone), ognuno con i propri buffer dedicati.

Mentre lo **Stream 0** esegue il kernel sul Chunk :

* Lo **Stream 1** scarica (Download) i risultati del Chunk .
* Lo **Stream 2** carica (Upload) i dati del Chunk .

Questo nasconde quasi completamente la latenza del bus PCI-Express.

---

## 🔥 4. Il Kernel "Ultimate" (GPU Optimization)

Il cuore delle performance. Abbiamo trasformato un kernel da 0.5 TFLOPS in uno da **2.7+ TFLOPS** attraverso tre tecniche.

### A. 2D Register Tiling (Thread Coarsening)

Invece di assegnare 1 thread a 1 pixel di output (inefficiente), assegniamo a ogni thread un **blocco 4x4 (16 pixel)**.

* **Vantaggio:** Riduciamo gli accessi alla Shared Memory. Un dato caricato in un registro viene riusato per calcolare più punti della griglia.
* **Configurazione:**
* Blocco Shared (Macro-Tile): .
* Lavoro per Thread (Micro-Tile): .
* Threads totali: .



### B. Vectorized Global Loads (`float4`)

Grazie al **Padding** implementato lato Host, sappiamo che ogni riga inizia su un indirizzo allineato a 128-bit.
Sostituiamo le letture scalari (`float`) con letture vettoriali (`float4`).

* **Effetto:** Riduciamo il numero di istruzioni di load del 75%.
* **Implementazione:**
```cpp
using float4 = ::float4;
// Caricamento sicuro grazie al padding
float4 loaded = *reinterpret_cast<const float4*>(&A[idx]);

```



### C. Loop Unrolling

Utilizziamo `#pragma unroll` sui loop interni che operano sui registri.
Questo permette al compilatore di:

1. Rimuovere l'overhead di controllo del ciclo.
2. Intervallare le istruzioni matematiche (FMA) per nascondere la latenza della pipeline aritmetica.

---

## 📊 5. Risultati e Conclusioni

La combinazione di queste tecniche ha prodotto un sistema robusto e veloce.

| Metrica | Valore / Note |
| --- | --- |
| **Precisione** | FP32 (Single Precision) |
| **Performance** | **~2.7 TFLOPS** (su NVIDIA L4) |
| **Speedup** | **+440%** rispetto alla versione base (Naive) |
| **Scalabilità** | Multi-GPU via MPI |
| **Robustezza** | Gestisce dimensioni arbitrarie (dispari/prime) senza crash |

**AlpaScalata** dimostra che è possibile scrivere codice **portabile** (grazie ad Alpaka) senza sacrificare le prestazioni estreme, a patto di curare attentamente l'architettura della memoria (Padding e Vectorization).
