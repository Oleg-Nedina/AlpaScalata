# 🚀 AlpaScalata — GPU Build & Run Guide (Spack + GCC 13 + MPI/OpenMP)

Questa guida spiega passo passo come configurare l'ambiente, compilare ed eseguire **AlpaScalata** su un nodo GPU del cluster, includendo il supporto **Distribuito (MPI)** e il parallelismo **Host (OpenMP)**.

---

## 0️⃣ Prerequisiti: Entrare nel Nodo GPU

Assicurati di essere loggato su un nodo con GPU (non il login node).

1.  **Carica l'ambiente Spack:**
    ```bash
    source /software/spack-v1.0/share/spack/setup-env.sh
    ```

2.  **Carica i moduli necessari (CUDA e MPI):**
    ```bash
    spack load cuda
    spack load openmpi
    ```

3.  **Verifica la GPU:**
    ```bash
    nvidia-smi
    ```

---

## 1️⃣ Setup Variabili Ambiente

Impostiamo il percorso di CUDA 13.0.2 e le librerie necessarie.

```bash
# Percorso specifico dell'installazione Spack
export MY_CUDA_ROOT="/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap"

# Aggiorna PATH e Librerie
export PATH="${MY_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MY_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}"

```

---

## 2️⃣ Pulizia e Configurazione CMake

Configurazione robusta per compatibilità GCC 13 + CUDA (C++17 + ABI Fix) e **attivazione di OpenMP** (per i `#pragma`).

```bash
cd ~/AlpaScalata

# 1. Pulisci la build precedente
rm -rf build

# 2. Configura
# Nota: Aggiunto -fopenmp in CXX_FLAGS per attivare il multithreading CPU
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_ALPAKA=ON \
  -DCMAKE_CUDA_COMPILER="${MY_CUDA_ROOT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${MY_CUDA_ROOT}" \
  -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -DALPAKA_ACC_GPU_CUDA_ENABLED -fopenmp" \
  -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -allow-unsupported-compiler -DALPAKA_ACC_GPU_CUDA_ENABLED"

```

---

## 3️⃣ Compilazione

Compila sia i micro-benchmark che l'eseguibile MPI.

```bash
cmake --build build -j

```

---

## 4️⃣ Check di Correttezza (Micro-Benchmarks)

Verifica che i calcoli siano corretti (GPU vs CPU) su singola GPU.

**Alpaka (GPU):**

```bash
./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm --check 64

```

*Deve stampare: `CHECK_OK*`

**CUDA Nativo:**

```bash
./build/bench/micro/benchmark_cuda --config config/cuda_naive_float_small.prm --check 64

```

*Deve stampare: `CHECK_OK*`

---

## 5️⃣ Esecuzione Benchmark Distribuito (MPI) 🌐

Eseguiamo il benchmark su più processi (multi-GPU) usando `mpirun`.

**Sintassi:** `mpirun -np <NUM_GPU> ./build/src/benchmark_mpi <M> <N> <K>`

1. **Test Standard (Matrice 16k su 2 GPU):**  (nota che oversubscribe è necessario per usare 2 gpu)

```bash
mpirun --oversubscribe -n 2 ./build/src/benchmark_mpi 16384 16384 16384

```

2. **Test Padding / Robustezza (Dimensioni dispari):**

```bash
mpirun -np 2 ./build/src/benchmark_mpi 16385 16385 16385

```

*Verifica che l'output sia `RESULT: OK`.*

---

## 6️⃣ Salvataggio Dati Micro-Benchmarks 📊

Eseguiamo i test locali completi salvando l'output CSV.

1. **Crea la cartella per i risultati:**

```bash
mkdir -p data/results

```

2. **Esegui e Salva Alpaka (GPU):**

```bash
echo "Running Alpaka Benchmark..."
./build/bench/micro/benchmark_alpaka \
  --config config/alpaka_naive_float_small.prm \
  > data/results/alpaka_naive_float_gpu.csv

```

3. **Esegui e Salva CUDA Nativo:**

```bash
echo "Running CUDA Benchmark..."
./build/bench/micro/benchmark_cuda \
  --config config/cuda_naive_float_small.prm \
  > data/results/cuda_naive_float_gpu.csv

```

---

## 🔁 Full Workflow (Copia-Incolla Rapido)

Se sei appena entrato nel nodo, usa questo blocco unico:

```bash
# 1. Setup Ambiente
source /software/spack-v1.0/share/spack/setup-env.sh
spack load cuda
spack load openmpi
export MY_CUDA_ROOT="/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap"
export PATH="${MY_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MY_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}"

# 2. Build (con OpenMP)
cd ~/AlpaScalata
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DENABLE_ALPAKA=ON \
  -DCMAKE_CUDA_COMPILER="${MY_CUDA_ROOT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${MY_CUDA_ROOT}" \
  -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++" \
  -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -DALPAKA_ACC_GPU_CUDA_ENABLED -fopenmp" \
  -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -allow-unsupported-compiler -DALPAKA_ACC_GPU_CUDA_ENABLED"

cmake --build build -j

# 3. Cartella Risultati
mkdir -p data/results

# 4. Esecuzione Distribuita (Esempio 2 GPU)
echo ">>> Running MPI Benchmark (16k)..."
mpirun -np 2 ./build/src/benchmark_mpi 16384 16384 16384

# 5. Esecuzione Micro-Benchmarks
echo ">>> Saving Alpaka results..."
./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm > data/results/alpaka_naive_float_gpu.csv

echo ">>> Saving CUDA results..."
./build/bench/micro/benchmark_cuda --config config/cuda_naive_float_small.prm > data/results/cuda_naive_float_gpu.csv

echo "Fatto! I file sono in data/results/"

```

```

```
