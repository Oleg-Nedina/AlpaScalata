
# Source Code

This directory contains the **core implementations** of the GEMM solvers used in the project.

The code is organized by backend in order to:
- separate concerns cleanly
- support multiple execution models (CUDA, Alpaka)
- make performance comparisons explicit and reproducible

Each backend must implement solvers conforming to the **public GEMM API** defined in `include/gemm`.

---

## Structure
src/

└── backends/

├── cuda/ # CUDA implementations

└── alpaka/ # Alpaka-based portable implementations (future)


---

## Design Guidelines

- Each solver lives in its own translation unit
- All solvers expose the same API
- No benchmark logic is allowed here
- No I/O or plotting code belongs in this directory

This separation allows solvers to be reused consistently across:
- unit tests
- micro-benchmarks
- correctness verification

---

