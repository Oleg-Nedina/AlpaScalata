
## src/backends/alpaka/

Backend Alpaka.

Contiene l’implementazione GEMM basata su Alpaka, con l’obiettivo di:
- mantenere portabilità
- confrontare overhead e flessibilità rispetto a CUDA nativo

La struttura dei kernel ricalca quella del backend CUDA,
per consentire confronti il più possibile equi.
