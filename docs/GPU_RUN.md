# 🚀 AlpaScalata — GPU Build & Run Guide (Spack + GCC 13 Fix)

Questa guida spiega passo passo come configurare l'ambiente, compilare ed eseguire **AlpaScalata** su un nodo GPU del cluster, salvando i dati per i grafici.

---

## 0️⃣ Prerequisiti: Entrare nel Nodo GPU

Assicurati di essere loggato su un nodo con GPU (non il login node).

1.  **Carica l'ambiente Spack:**
    ```bash
    source /software/spack-v1.0/share/spack/setup-env.sh
    ```

2.  **Carica il modulo CUDA:**
    ```bash
    spack load cuda
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

Configurazione robusta per compatibilità GCC 13 + CUDA (C++17 + ABI Fix).

```bash
cd ~/AlpaScalata

# 1. Pulisci la build precedente
rm -rf build

# 2. Configura
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DENABLE_ALPAKA=ON \
  -DCMAKE_CUDA_COMPILER="${MY_CUDA_ROOT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${MY_CUDA_ROOT}" \
  -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -DALPAKA_ACC_GPU_CUDA_ENABLED" \
  -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -allow-unsupported-compiler -DALPAKA_ACC_GPU_CUDA_ENABLED"

```

---

## 3️⃣ Compilazione

```bash
cmake --build build -j

```

---

## 4️⃣ Check di Correttezza (Obbligatorio prima dei benchmark)

Verifica che i calcoli siano corretti (GPU vs CPU).

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

## 5️⃣ Esecuzione Benchmark e Salvataggio Dati 📊

Qui eseguiamo i test completi e usiamo `>` per salvare l'output CSV nei file.

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
export MY_CUDA_ROOT="/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap"
export PATH="${MY_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MY_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}"

# 2. Build
cd ~/AlpaScalata
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON -DENABLE_ALPAKA=ON \
  -DCMAKE_CUDA_COMPILER="${MY_CUDA_ROOT}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${MY_CUDA_ROOT}" \
  -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++" \
  -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -DALPAKA_ACC_GPU_CUDA_ENABLED" \
  -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -allow-unsupported-compiler -DALPAKA_ACC_GPU_CUDA_ENABLED"

cmake --build build -j

# 3. Cartella Risultati
mkdir -p data/results

# 4. Esecuzione e Salvataggio
echo "Saving Alpaka results..."
./build/bench/micro/benchmark_alpaka --config config/alpaka_naive_float_small.prm > data/results/alpaka_naive_float_gpu.csv

echo "Saving CUDA results..."
./build/bench/micro/benchmark_cuda --config config/cuda_naive_float_small.prm > data/results/cuda_naive_float_gpu.csv

echo "Fatto! I file sono in data/results/"

```

```

```
