
# Dense GEMM Optimization – CUDA & Alpaka

Questo progetto studia e confronta l’ottimizzazione del prodotto tra matrici dense (GEMM)
utilizzando:
- CUDA (backend nativo NVIDIA)
- Alpaka (astrazione per portabilità HPC)

L’obiettivo è confrontare correttezza, prestazioni e scalabilità dei diversi approcci
su cluster GPU, mantenendo un’API comune e benchmark riproducibili.

La struttura della repository è organizzata per **componenti funzionali**
(API, backend, test, benchmark) e non per singola tecnologia.
