# include/

**Public Project Headers**

This directory defines the **Common API** for Dense Matrix Multiplication (GEMM). It is designed to be completely independent of the underlying execution backend (e.g., CUDA or Alpaka).

##  Directory Contents

This abstraction layer contains the following components:

* **Interfaces:** Core function definitions (e.g., `gemm.hpp`).
* **Common Types:** Shared definitions for memory layouts, datatypes, and configurations.
* **Tensor Views:** Non-owning views for efficient matrix data handling.

---

##  Implementation Contract

> **Crucial for Performance Benchmarking**

All backends **must** implement this interface without modifying its semantics. This strict adherence is required to ensure fair and accurate performance comparisons across different hardware and programming models.