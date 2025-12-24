# Alpaka GEMM: Architettura "Full Options" Ibrida

Questo documento descrive l'implementazione finale del kernel GEMM (General Matrix Multiply) sviluppato utilizzando la libreria di astrazione **Alpaka**. L'obiettivo del design è stato creare un solutore **robusto, portabile e performante**, capace di adattarsi dinamicamente sia a matrici di piccole dimensioni (massimizzando il throughput) che a matrici "Big Data" che superano la capacità della VRAM (evitando crash).

## 1. Funzionalità Implementate

Il codice finale (`gemm_alpaka_full_options.cpp`) integra tre meccanismi principali che lavorano in sinergia:

### A. Gestione Ibrida della Memoria (In-Core vs Out-of-Core)

Il sistema decide a runtime quale strategia di memoria utilizzare basandosi sulle risorse disponibili hardware e sulla dimensione del problema.

* **Standard Path (In-Core):** Se le matrici entrano nella VRAM disponibile, il codice adotta un approccio **Zero-Copy**. I puntatori device vengono passati direttamente al kernel senza allocazioni intermedie o copie ridondanti.
* *Vantaggio:* Massimizza la banda passante della memoria (fino a ~1 TB/s su L4).


* **Batching Path (Out-of-Core):** Se la memoria richiesta supera quella disponibile (stimata dinamicamente), il codice attiva il **Tiling a livello Host**. Le matrici vengono suddivise in "Chunk" (es. ), trasferite sulla GPU pezzo per pezzo, elaborate e accumulate.
* *Vantaggio:* Permette di elaborare matrici di dimensione arbitraria (limitate solo dalla RAM di sistema) senza causare errori *Out Of Memory*.



### B. Tiling Rettangolare Parametrico

Invece di limitarsi a blocchi quadrati fissi (es. ), il kernel è stato generalizzato tramite template C++ per supportare dimensioni `TM x TN x TK` arbitrarie.

* Il kernel supporta il **caricamento collaborativo (Coalesced Loading)** anche quando il numero di thread nel blocco non corrisponde alla dimensione della tile di dati, disaccoppiando la geometria del calcolo dalla geometria della memoria.

### C. Dispatcher Dinamico (Heuristic Hardware)

All'avvio, il software interroga l'hardware sottostante (tramite `alpaka::getAccDevProps`) per determinare la configurazione ottimale:

* **High-End (es. A100, L4):** Se supporta 1024 thread/blocco  Usa Tile .
* **Mid-Range:** Se limitato nei registri  Usa Tile Rettangolare  (512 thread).
* **Legacy:** Fallback a  (256 thread).

---

## 2. Scelte di Design e Pattern Utilizzati

### Pattern: Runtime Strategy Dispatcher

Abbiamo implementato una variante del pattern Strategy. Invece di avere un unico kernel monolitico, il codice compila diverse specializzazioni del template (`launch_rect_kernel<32,32,32>`, `<16,32,32>`, etc.). A runtime, un `if-else` basato sulle proprietà dell'hardware devia l'esecuzione verso la specializzazione più efficiente.

### Pattern: Double Buffering (Implicit) & Accumulation

Nel percorso *Out-of-Core*, abbiamo implementato la logica di accumulo parziale.

1. Il kernel accetta un flag `accumulate`.
2. Al primo passaggio (`k=0`), sovrascrive il buffer di output (`C = A*B`).
3. Ai passaggi successivi (`k>0`), somma al risultato esistente (`C += A*B`).
Questo permette di ricostruire il risultato finale della moltiplicazione di matrici giganti processando solo sotto-blocchi.

### Astrazione "Safe" delle View

Per risolvere i problemi di allineamento di memoria (`cudaErrorInvalidPitchValue`) riscontrati con le API di basso livello, abbiamo incapsulato i puntatori raw in **Alpaka Views**.

* Utilizziamo `createView` con stride espliciti per gestire correttamente sia buffer contigui (nel caso In-Core) che sottomatrici con pitch (nel caso Out-of-Core), garantendo la *Type Safety* imposta dalle versioni recenti di Alpaka.

---

## 3. Perché Alpaka? (Confronto con CUDA Nativo)

Avremmo potuto scrivere tutto in CUDA C++ puro (`.cu`). Ecco un confronto critico delle scelte:

| Funzionalità | Implementazione CUDA Nativa | Implementazione Alpaka (La nostra scelta) |
| --- | --- | --- |
| **Portabilità** | Funziona solo su hardware NVIDIA. | Funziona su NVIDIA (CUDA), AMD (HIP), Intel (SYCL) e CPU (OpenMP) con lo stesso codice sorgente. |
| **Gestione Memoria** | Accesso diretto a `cudaMemGetInfo` per byte esatti liberi. | Accesso a `getAccDevProps` (Totale VRAM). Abbiamo dovuto implementare una stima (`Totale - Riserva`) per mantenere la portabilità "pura" senza includere header CUDA. |
| **Kernel Launch** | Sintassi `<<<grid, block>>>`. Semplice ma rigida. | Oggetto `WorkDiv`. Più verboso, ma astrae la griglia di calcolo su architetture diverse (es. CPU threads vs GPU warps). |
| **Ottimizzazioni** | Possibilità di usare `WMMA` (Tensor Cores) e `__ldg` intrinsics. | Limitato alle funzionalità esposte dall'API (principalmente FP32 SIMT standard nel nostro caso). |

**Perché abbiamo scelto questo approccio:**
L'obiettivo di "AlpaScalata" è la scalabilità e la portabilità. Pur sacrificando l'accesso a `cudaMemGetInfo` (risolto con una stima conservativa), abbiamo ottenuto un codice che può teoricamente girare su un supercomputer basato su AMD Instinct senza cambiare una virgola, mantenendo logiche avanzate come il batching.

---

## 4. Limitazioni Attuali

Nonostante la robustezza, l'implementazione presenta alcune limitazioni note:

1. **Stima della Memoria Conservativa:**
Non potendo usare chiamate native del driver (per non rompere la portabilità), stimiamo la VRAM libera come `VRAM_Totale - 1GB`. Su sistemi con molti processi in background, questa stima potrebbe essere imprecisa.
2. **Collo di Bottiglia PCIe (Batching):**
Nel modo *Out-of-Core*, le performance crollano da ~2500 GFLOPS a ~1700 GFLOPS. Questo è fisiologico (il bus PCIe è molto più lento della VRAM), ma potrebbe essere mitigato implementando **Streams Asincroni** (copia del chunk N+1 mentre calcolo il chunk N), che però aggiungerebbero notevole complessità al codice Alpaka.
3. **Mancanza di Tensor Cores:**
Il kernel attuale usa istruzioni scalari FP32 (`float`). Non sfrutta le unità matriciali hardware (Tensor Cores) presenti sulla NVIDIA L4, che richiederebbero API specifiche non ancora pienamente standardizzate nel layer alto di Alpaka.

## 5. Conclusione

Il modulo `gemm_alpaka_full_options` rappresenta lo stato dell'arte per un'implementazione portabile. Garantisce:

1. **Massima Performance** su dati che stanno in memoria (Zero-Copy).
2. **Massima Affidabilità** su dati giganti (Batching automatico).
3. **Adattabilità** su hardware diverso (Tiling Dinamico).

È la soluzione definitiva per benchmark che devono esplorare limiti hardware senza fallire in condizioni di stress.

