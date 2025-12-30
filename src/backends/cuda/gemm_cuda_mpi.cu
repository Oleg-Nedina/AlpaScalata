#include "gemm/gemm.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <vector>

// Helper to align dimensions to multiples of 4 (for float4 vectorization)
int get_padded_dim(int n) { return (n + 3) / 4 * 4; }

// Select the specific GPU based on the MPI Rank
static void select_gpu_device(int rank) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int deviceId = rank % deviceCount;
        cudaSetDevice(deviceId);
    }
}

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Setup GPU
    select_gpu_device(rank);

    // Parse arguments (M, N, K)
    int M_real = 16384;
    int N_real = 16384;
    int K_real = 16384;

    if (rank == 0) {
        if (argc >= 4) {
            M_real = std::atoi(argv[1]);
            N_real = std::atoi(argv[2]);
            K_real = std::atoi(argv[3]);
        } else {
            std::cout << "Usage: mpirun ... ./gemm_cuda_mpi <M> <N> <K>" << std::endl;
            std::cout << "Defaulting to 16384x16384x16384" << std::endl;
        }
    }

    // Broadcast dimensions to all ranks
    int dims[3] = {M_real, N_real, K_real};
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    M_real = dims[0];
    N_real = dims[1];
    K_real = dims[2];

    // Calculate padding
    // First, ensure alignment to 4 (for vector loads)
    int M_pad = get_padded_dim(M_real);
    int N_pad = get_padded_dim(N_real);
    int K_pad = get_padded_dim(K_real);

    // Second, ensure M is perfectly divisible by the number of ranks
    if (M_pad % size != 0) {
        int rem = M_pad % size;
        M_pad += (size - rem);
    }

    int M_local_pad = M_pad / size; // Rows per rank

    // Memory Allocation
    // Using cudaMallocHost (Pinned Memory) for faster PCI-E transfers

    float *h_A_full = nullptr;
    float *h_B = nullptr;
    float *h_C_full = nullptr;

    float *h_A_local = nullptr;
    float *h_C_local = nullptr;

    size_t size_B_pad = static_cast<size_t>(K_pad) * N_pad;
    size_t size_A_local_pad = static_cast<size_t>(M_local_pad) * K_pad;
    size_t size_C_local_pad = static_cast<size_t>(M_local_pad) * N_pad;

    // Allocate local buffers on all ranks
    cudaMallocHost((void**)&h_A_local, size_A_local_pad * sizeof(float));
    cudaMallocHost((void**)&h_C_local, size_C_local_pad * sizeof(float));
    cudaMallocHost((void**)&h_B, size_B_pad * sizeof(float));

    // Master allocates full matrices for initialization and gathering
    if (rank == 0) {
        size_t size_A_full = static_cast<size_t>(M_pad) * K_pad;
        size_t size_C_full = static_cast<size_t>(M_pad) * N_pad;

        double gb_req = (double)(size_A_full + size_C_full + size_B_pad) * 4.0 / 1e9;
        std::cout << "Master: Allocation of RAM (Padded) ~" << gb_req << " GB..." << std::endl;
        std::cout << "Padding: [" << M_real << "x" << N_real << "x" << K_real
                  << "] -> [" << M_pad << "x" << N_pad << "x" << K_pad << "]" << std::endl;

        cudaMallocHost((void**)&h_A_full, size_A_full * sizeof(float));
        cudaMallocHost((void**)&h_C_full, size_C_full * sizeof(float));

        // Initialization (Master Only)
        // A row-major = r%100, B = 1.0
        #pragma omp parallel for
        for (int r = 0; r < M_real; ++r) {
            float val = (float)(r % 100);
            for (int c = 0; c < K_real; ++c) {
                h_A_full[r * K_pad + c] = val;
            }
        }
        // Zero out padding rows in A if any
        for (int r = M_real; r < M_pad; ++r) {
            for (int c = 0; c < K_pad; ++c) h_A_full[r * K_pad + c] = 0.0f;
        }

        #pragma omp parallel for
        for (int r = 0; r < K_real; ++r) {
            for (int c = 0; c < N_real; ++c) {
                h_B[r * N_pad + c] = 1.0f;
            }
        }
    }

    // Distribute data
    // Broadcast B
    if (size_B_pad < 2000000000) {
        MPI_Bcast(h_B, size_B_pad, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        if (rank == 0) std::cout << "Warning: B too big, local generation." << std::endl;
        // Local generation if broadcast is skipped
        for (int r = 0; r < K_real; ++r)
            for (int c = 0; c < N_real; ++c)
                h_B[r * N_pad + c] = 1.0f;
    }

    if (rank == 0) std::cout << "Partitioning A..." << std::endl;

    // Scatter A
    // Since M_pad is divisible by size, using MPI_Scatter
    MPI_Scatter(h_A_full, M_local_pad * K_pad, MPI_FLOAT,
                h_A_local, M_local_pad * K_pad, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    if (rank == 0) std::cout << ">>> START GPU (CUDA) <<<" << std::endl;

    // Execute kernel
    gemm::GemmShape local_shape = {M_local_pad, N_pad, K_pad};
    gemm::gemm_cuda_full_options(h_A_local, h_B, h_C_local, local_shape);

    // Ensure GPU is finished before stopping timer
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) std::cout << "Gather results..." << std::endl;

    // Gather results
    MPI_Gather(h_C_local, M_local_pad * N_pad, MPI_FLOAT,
               h_C_full, M_local_pad * N_pad, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    // Verification & reporting
    if (rank == 0) {
        double elapsed = end_time - start_time;
        double gflops = (2.0 * (double)M_real * N_real * K_real) / (elapsed * 1e9);

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Real dim :  " << M_real << " x " << N_real << " x " << K_real << std::endl;
        std::cout << "Total time:      " << elapsed << " s" << std::endl;
        std::cout << "TFLOPS utilized:      " << gflops / 1000.0 << " TFLOPS" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Verification ..." << std::endl;

        int errors = 0;
        std::vector<int> rows_to_check = {0, M_real / 2, M_real - 1, M_real / 4};

        for (int r : rows_to_check) {
            if (r < 0 || r >= M_real) continue;

            float expected = (float)(r % 100) * (float)K_real;

            // Access padded C matrix
            float val_first = h_C_full[r * N_pad + 0];
            float val_last  = h_C_full[r * N_pad + (N_real - 1)];

            bool row_pass = true;
            if (std::abs(val_first - expected) > 0.1f) row_pass = false;
            if (std::abs(val_last - expected) > 0.1f)  row_pass = false;

            if (!row_pass) {
                std::cout << "FAIL at row " << r << " -> Expected: " << expected
                          << ", Found: " << val_first << std::endl;
                errors++;
            }
        }

        if (errors == 0)
            std::cout << "RESULT: OK " << std::endl;
        else
            std::cout << "RESULT: FAIL (found " << errors << " errors)" << std::endl;

        // Cleanup master
        cudaFreeHost(h_A_full);
        cudaFreeHost(h_C_full);
    }

    // Cleanup local
    cudaFreeHost(h_A_local);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_local);

    MPI_Finalize();
    return 0;
}