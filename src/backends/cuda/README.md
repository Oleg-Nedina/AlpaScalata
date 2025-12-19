
## src/backends/cuda/

Backend CUDA nativo.

Contiene:
- implementazioni GEMM CUDA
- kernel con diversi livelli di ottimizzazione
  (naive, tiled, vectorized, ecc.)

Questo backend serve come riferimento diretto per le prestazioni
massime su GPU NVIDIA, senza astrazioni aggiuntive.
