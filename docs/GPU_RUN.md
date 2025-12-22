# 🚀 AlpaScalata — GPU Build & Run Guide (CUDA + Alpaka)

Questa guida descrive **passo per passo** come compilare ed eseguire **AlpaScalata** su un **nodo GPU** del cluster, abilitando **sia CUDA nativo che Alpaka**.

L’obiettivo è:
- evitare errori ricorrenti (CUDA, architetture, path)
- garantire che **entrambi i benchmark (`cuda` e `alpaka`) vengano sempre costruiti**
- fornire un riferimento unico per tutti i membri del team

---

## 0️⃣ Prerequisiti

Devi essere su un **GPU node** (non login).

Verifica:
```bash
nvidia-smi
````

Devi vedere almeno una GPU (es. NVIDIA L4).

---

## 1️⃣ Setup ambiente CUDA (Spack)

Sul cluster CUDA è fornito tramite **Spack**.

```bash
export CUDA_HOME=/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap
export PATH=$CUDA_HOME/bin:$PATH
```

Verifica:

```bash
nvcc --version
nvidia-smi
```

---

## 2️⃣ Vai nella root del progetto

⚠️ **Fondamentale**: tutti i comandi CMake devono essere lanciati dalla **root del repository**, dove si trova `CMakeLists.txt`.

```bash
cd ~/AlpaScalata
ls CMakeLists.txt
```

Se `CMakeLists.txt` non esiste → sei nella directory sbagliata.

---

## 3️⃣ Pulizia build precedente (sempre consigliata)

```bash
rm -rf build
```

---

## 4️⃣ Configurazione CMake (CUDA + Alpaka)

### Architettura GPU

* NVIDIA **L4** → compute capability **8.9**
* Impostiamo esplicitamente:

```text
CMAKE_CUDA_ARCHITECTURES=89
```

### Configurazione completa

Questa configurazione **abilita sempre entrambi**:

* backend CUDA
* backend Alpaka (CPU + GPU)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_ALPAKA=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

⚠️ Se **questa fase fallisce**, non proseguire.

---

## 5️⃣ Build

```bash
cmake --build build -j
```

Target attesi:

* `gemm_backend_cuda`
* `gemm_backend_alpaka`
* `benchmark_cuda`
* `benchmark_alpaka`

---

## 6️⃣ Verifica eseguibili generati

```bash
ls build/bench/micro/
```

Devi vedere almeno:

```text
benchmark_cuda
benchmark_alpaka
```

---

## 7️⃣ Run benchmark Alpaka (GPU)

### Test di correttezza (consigliato)

```bash
unset LD_LIBRARY_PATH

./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  --check 64
```

Output atteso:

```text
CHECK_OK
```

---

### Benchmark Alpaka (GPU)

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm
```

Output (CSV su stdout):

```text
Solver,Precision,Size,Time_ms,GFLOPs
naive,float,256,15.279561,2.196034
...
```

---

## 8️⃣ Run benchmark CUDA nativo

### Test di correttezza

```bash
unset LD_LIBRARY_PATH

./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm \
  --check 64
```

---

### Benchmark CUDA

```bash
./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm
```

---

## 9️⃣ Salvare i risultati per plotting

I benchmark **scrivono su stdout**.
Per salvare i risultati:

```bash
mkdir -p data/results
```

### Alpaka (GPU)

```bash
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_gpu.csv
```

### CUDA nativo

```bash
./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm \
  > data/results/cuda_naive_float_gpu.csv
```

---

## 🔁 Workflow completo (TL;DR)

```bash
# su GPU node
export CUDA_HOME=...
export PATH=$CUDA_HOME/bin:$PATH

cd ~/AlpaScalata
rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DENABLE_ALPAKA=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build -j

unset LD_LIBRARY_PATH

./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm --check 64
./build/bench/micro/benchmark_cuda   --config config/cuda_naive_float_small.prm   --check 64

./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_gpu.csv

./build/bench/micro/benchmark_cuda --config config/cuda_naive_float_small.prm \
  > data/results/cuda_naive_float_gpu.csv

