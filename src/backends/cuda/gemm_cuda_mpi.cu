
#include "gemm/gemm.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <mpi.h>

namespace gemm {

    static void select_gpu_device(int rank) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) {
            // Round-Robin assignment.
            int deviceId = rank % deviceCount;
            cudaSetDevice(deviceId);
        }
    }

    void gemm_cuda_mpi(const float *A, const float *B, float *C, GemmShape s) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Setup GPU Device for this rank
        select_gpu_device(rank);

        // Calculate Partitioning of A
        // Determine how many rows each rank handles.
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);

        int rows_per_proc = s.m / size;
        int remainder = s.m % size;

        // Calculate counts / displacements for Scatterv / Gatherv
        int current_displ = 0;
        for (int i = 0; i < size; ++i) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            // Count in ELEMENTS (floats)
            sendcounts[i] = rows * s.k;
            displs[i] = current_displ;
            current_displ += sendcounts[i];
        }

        // Prepare Local Data
        int local_m_rows = sendcounts[rank] / s.k;
        int local_A_size = sendcounts[rank];
        int B_size = s.k * s.n;
        int local_C_size = local_m_rows * s.n;

        // Allocate host memory for local parts
        std::vector<float> local_A(local_A_size);
        std::vector<float> local_B(B_size);
        std::vector<float> local_C(local_C_size);

        // Distribute Data

        // Scatter A: Root sends slices of A to everyone
        MPI_Scatterv(const_cast<float*>(A), sendcounts.data(), displs.data(), MPI_FLOAT,
                     local_A.data(), local_A_size, MPI_FLOAT,
                     0, MPI_COMM_WORLD);

        // Broadcast B: Root sends entire B to everyone
        if (rank == 0) {
            // Root copies its B to local buffer to simplify logic or just sends directly
            MPI_Bcast(const_cast<float*>(B), B_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
            // Root needs B in its local buffer for the compute function
            std::copy(B, B + B_size, local_B.begin());
        } else {
            MPI_Bcast(local_B.data(), B_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Local Compute (GPU Offload)
        GemmShape local_shape = {local_m_rows, s.n, s.k};

        gemm_cuda_full_options(local_A.data(), local_B.data(), local_C.data(), local_shape);

        // Gather Results
        // Recalculate counts for C (C has dimensions M x N, not M x K)
        std::vector<int> recvcounts_C(size);
        std::vector<int> displs_C(size);

        if (rank == 0) {
            int current_displ_C = 0;
            for (int i = 0; i < size; ++i) {
                int rows = rows_per_proc + (i < remainder ? 1 : 0);
                recvcounts_C[i] = rows * s.n;
                displs_C[i] = current_displ_C;
                current_displ_C += recvcounts_C[i];
            }
        }

        MPI_Gatherv(local_C.data(), local_C_size, MPI_FLOAT,
                    C, recvcounts_C.data(), displs_C.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    }

} // namespace gemm

