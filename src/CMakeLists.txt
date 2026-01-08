# src/CMakeLists.txt

set(PROJ_INC_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src/include
)

if(ENABLE_CUDA OR ENABLE_ALPAKA)
  if(NOT MPI_CXX_HEADER_DIR)
    find_path(MPI_CXX_HEADER_DIR mpi.h
      HINTS ENV MPI_HOME
      PATHS /usr/lib/x86_64-linux-gnu/openmpi/include
      PATH_SUFFIXES include openmpi/include
    )
  endif()
  if(NOT MPI_CXX_COMPILER)
    find_program(MPI_CXX_COMPILER NAMES mpicxx mpic++ mpicxx.openmpi)
  endif()
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
          backends/cuda/gemm_cuda_naive.cu
          backends/cuda/gemm_cuda_full_options.cu
  )
  target_include_directories(gemm_backend_cuda PUBLIC ${PROJ_INC_DIRS})
  target_link_libraries(gemm_backend_cuda PUBLIC CUDA::cudart)

  target_compile_options(gemm_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )

  #add_executable(benchmark_cuda backends/cuda/benchmark_test.cu)
  #target_link_libraries(benchmark_cuda PRIVATE gemm_backend_cuda CUDA::cudart)
  #target_include_directories(benchmark_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)

  add_executable(gemm_cuda_mpi backends/cuda/gemm_cuda_mpi.cu)

  target_link_libraries(gemm_cuda_mpi
    PRIVATE
      gemm_backend_cuda
      MPI::MPI_CXX
      CUDA::cudart
  )
  target_include_directories(gemm_cuda_mpi PRIVATE ${CMAKE_SOURCE_DIR}/include)

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

  set_source_files_properties(
               backends/alpaka/gemm_mpi.cpp
                   PROPERTIES LANGUAGE CUDA
                     )

  add_executable(benchmark_mpi backends/alpaka/gemm_mpi.cpp)

  target_link_libraries(benchmark_mpi
                                 PRIVATE
                                       MPI::MPI_CXX
                                             gemm_backend_alpaka
                                                   alpaka
                                                         CUDA::cudart
                                                           )

  target_include_directories(benchmark_mpi PRIVATE ${CMAKE_SOURCE_DIR}/include)

endif()
