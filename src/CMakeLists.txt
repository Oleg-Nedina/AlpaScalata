# src/CMakeLists.txt

set(PROJ_INC_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src/include
)

if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
          backends/cuda/gemm_cuda_naive.cu
   # backends/cuda/gemm_cuda_full_options.cu
    #backends/cuda/gemm_cuda_mpi.cu
  )
  target_include_directories(gemm_backend_cuda PUBLIC ${PROJ_INC_DIRS})
  target_link_libraries(gemm_backend_cuda PUBLIC CUDA::cudart)
  target_compile_options(gemm_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )
endif()
if(ENABLE_ALPAKA)
  set_source_files_properties(
    backends/alpaka/gemm_alpaka_naive.cpp
    backends/alpaka/gemm_alpaka_tiling.cpp
    backends/alpaka/gemm_alpaka_full_options.cpp
    PROPERTIES LANGUAGE CUDA
  )

  add_library(gemm_backend_alpaka STATIC
    backends/alpaka/gemm_alpaka_naive.cpp
    backends/alpaka/gemm_alpaka_tiling.cpp
    backends/alpaka/gemm_alpaka_full_options.cpp
  )

  target_include_directories(gemm_backend_alpaka PUBLIC ${PROJ_INC_DIRS})
  target_link_libraries(gemm_backend_alpaka PUBLIC alpaka CUDA::cudart)

  target_compile_options(gemm_backend_alpaka PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )

  # --- CONFIGURAZIONE MPI ---
  find_package(MPI REQUIRED)

  # 1. IMPORTANTE: Diciamo a CMake che questo file contiene codice CUDA (Alpaka)
  #    altrimenti usa g++ e fallisce sugli header __device__
  set_source_files_properties(
               backends/alpaka/gemm_mpi.cpp
                   PROPERTIES LANGUAGE CUDA
                     )

  # 2. Definiamo l'eseguibile
  add_executable(benchmark_mpi backends/alpaka/gemm_mpi.cpp)

  # 3. Linkiamo le librerie
  target_link_libraries(benchmark_mpi
                                 PRIVATE
                                       MPI::MPI_CXX
                                             gemm_backend_alpaka
                                                   alpaka
                                                         CUDA::cudart
                                                           )

  target_include_directories(benchmark_mpi PRIVATE ${CMAKE_SOURCE_DIR}/include)

endif()
