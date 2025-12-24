<<<<<<< HEAD
=======
#define GEMM_ENABLE_ALPAKA
>>>>>>> 2ef9c240930f87157f282e459c78e9c31b4f9843
#include "gemm/gemm.hpp"
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cmath>
#include <cstdlib> // Per atoi
#include <iostream>
#include <mpi.h>
#include <vector>

// Setup Backend Alpaka (Copia dal tuo progetto)
using Dim2 = alpaka::DimInt<2>;
using Idx = std::size_t;
using Acc = alpaka::AccGpuCudaRt<Dim2, Idx>;
using QueueType = alpaka::Queue<Acc, alpaka::Blocking>;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // 1. GESTIONE INPUT (M, N, K)
  int M = 16384; // Default
  int N = 16384;
  int K = 16384;

  if (world_rank == 0) {
    // Il Master legge gli argomenti
    if (argc >= 4) {
      M = std::atoi(argv[1]);
      N = std::atoi(argv[2]);
      K = std::atoi(argv[3]);
    } else {
      std::cout << "Uso: mpirun ... ./benchmark_mpi <M> <N> <K>" << std::endl;
      std::cout << "Defaulting to 16384x16384x16384" << std::endl;
    }
  }

  // Fondamentale: Il Master deve dire a tutti le dimensioni scelte!
  // Spediamo 3 interi dal processo 0 a tutti gli altri.
  int dims[3] = {M, N, K};
  MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
  M = dims[0];
  N = dims[1];
  K = dims[2];

  // Check Divisibilità
  if (M % world_size != 0) {
    if (world_rank == 0)
      std::cerr << "Errore: M (" << M
                << ") deve essere divisibile per il numero di GPU ("
                << world_size << ")!" << std::endl;
    MPI_Finalize();
    return 1;
  }

  int M_local = M / world_size;
  size_t size_B = static_cast<size_t>(K) * N;
  size_t size_A_local = static_cast<size_t>(M_local) * K;
  size_t size_C_local = static_cast<size_t>(M_local) * N;
  size_t size_full_matrix = static_cast<size_t>(M) * N; // Per verifica Master

  // 2. ALLOCAZIONE MEMORIA HOST (RAM CPU)
  std::vector<float> h_B;
  std::vector<float> h_A_local;
  std::vector<float> h_C_local;

  // Vettori completi solo sul Master
  std::vector<float> h_A_full;
  std::vector<float> h_C_full;

  try {
    // Tutti allocano i buffer locali e B
    h_B.resize(size_B);
    h_A_local.resize(size_A_local);
    h_C_local.resize(size_C_local);

    if (world_rank == 0) {
      // Solo il Master alloca le matrici giganti intere
      double gb_req =
          (double)(size_full_matrix * 2 + size_full_matrix / N * K) * 4.0 / 1e9;
      std::cout << "Master: Allocazione RAM per matrici complete (~" << gb_req
                << " GB)..." << std::endl;

      h_A_full.resize(static_cast<size_t>(M) * K, 1.0f); // Inizializza a 1.0
      h_C_full.resize(static_cast<size_t>(M) * N, 0.0f); // Inizializza a 0.0

      // Inizializza B
      std::fill(h_B.begin(), h_B.end(), 2.0f);
    }
  } catch (const std::bad_alloc &e) {
    std::cerr << "ERRORE CRITICO (Rank " << world_rank
              << "): RAM Insufficiente! " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // 3. DISTRIBUZIONE DATI
  //
  // A. Broadcast B (Tutti ricevono B intera)
  // Nota: Se la matrice supera i 2GB (INT_MAX elementi), MPI_Bcast standard
  // potrebbe fallire. Per un test rapido va bene, per produzione servirebbero
  // chunk.
  if (size_B < 2000000000) {
    MPI_Bcast(h_B.data(), size_B, MPI_FLOAT, 0, MPI_COMM_WORLD);
  } else {
    if (world_rank == 0)
      std::cerr << "Warning: Matrice B troppo grande per singolo MPI_Bcast "
                   "(>2GB). Implementare chunking."
                << std::endl;
    // Per ora assumiamo che B venga generata localmente se troppo grande,
    // ma per correttezza matematica dovremmo spedirla.
    // Workaround rapido: generiamo B uguale ovunque.
    std::fill(h_B.begin(), h_B.end(), 2.0f);
  }

  // B. Scatter A (Spezzetta A dal Master ai Worker)
  // Idem per i limiti di dimensione.
  if (world_rank == 0)
    std::cout << "Distribuendo A ai worker..." << std::endl;
  MPI_Scatter(h_A_full.data(), M_local * K, MPI_FLOAT, h_A_local.data(),
              M_local * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // 4. SETUP DEVICE ALPAKA
  // CORREZIONE: Istanziamo prima la piattaforma
  auto platform = alpaka::PlatformCudaRt{};

  // CORREZIONE: Passiamo l'istanza alla funzione getDevCount
  int num_gpus = (int)alpaka::getDevCount(platform);

  // Safety check (opzionale ma consigliato)
  if (num_gpus == 0) {
    if (world_rank == 0)
      std::cerr << "ERRORE: Nessuna GPU rilevata!" << std::endl;
    MPI_Finalize();
    return 1;
  }

  int my_device_id = world_rank % num_gpus;

  // Ora usiamo la stessa platform istanziata sopra
  auto dev = alpaka::getDevByIdx(platform, my_device_id);
  QueueType queue(dev);

  // ... resto del codice ...
  // Sincronizzazione prima del via
  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  if (world_rank == 0)
    std::cout << ">>> AVVIO CALCOLO GPU <<<" << std::endl;

  // 5. ESECUZIONE (Chiama il tuo codice Full Options)
  // Passiamo le dimensioni LOCALI (M_local)
  gemm::GemmShape local_shape = {M_local, N, K};
  gemm::gemm_alpaka_full_options(queue, h_A_local.data(), h_B.data(),
                                 h_C_local.data(), local_shape);

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();

  // 6. RACCOLTA RISULTATI
  if (world_rank == 0)
    std::cout << "Raccolta risultati (Gather)..." << std::endl;
  MPI_Gather(h_C_local.data(), M_local * N, MPI_FLOAT, h_C_full.data(),
             M_local * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // 7. OUTPUT
  if (world_rank == 0) {
    double elapsed = end_time - start_time;
    double gflops = (2.0 * (double)M * N * K) / (elapsed * 1e9);

    std::cout << "------------------------------------------------"
              << std::endl;
    std::cout << "Dimensione: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "Tempo:      " << elapsed << " s" << std::endl;
    std::cout << "TFLOPS:     " << gflops / 1000.0 << " TFLOPS" << std::endl;
    std::cout << "------------------------------------------------"
              << std::endl;

    // Verifica rapida su un elemento a caso
    // A=1.0, B=2.0 -> C[i] = K * 1.0 * 2.0 = 2*K
    float expected = 2.0f * K;
    std::cout << "Verifica: C[0] = " << h_C_full[0] << " (Atteso: " << expected
              << ")" << std::endl;

    if (std::abs(h_C_full[0] - expected) < 0.1)
      std::cout << "RESULT: OK" << std::endl;
    else
      std::cout << "RESULT: FAIL" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
