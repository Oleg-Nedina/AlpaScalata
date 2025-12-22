
# src/CMakeLists.txt

# include path per "gemm/gemm.hpp"
set(PROJ_INC_DIR ${PROJECT_SOURCE_DIR}/src)

# ---------------- CUDA backend ----------------
if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
    backends/cuda/gemm_cuda.cu
    backends/cuda/gemm_cuda_naive.cu
  )

  target_include_directories(gemm_backend_cuda PUBLIC ${PROJ_INC_DIR})
  target_link_libraries(gemm_backend_cuda PUBLIC CUDA::cudart)

  # Opzionale: warning/flags
  target_compile_options(gemm_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )
endif()

# ---------------- Alpaka backend ----------------
if(ENABLE_ALPAKA)
  add_library(gemm_backend_alpaka STATIC
    backends/alpaka/gemm_alpaka_naive.cpp
  )

  target_include_directories(gemm_backend_alpaka PUBLIC ${PROJ_INC_DIR})
  target_link_libraries(gemm_backend_alpaka PUBLIC alpaka CUDA::cudart)

  # >>> QUESTO È IL PUNTO CHIAVE <<<
  # Compila il .cpp con nvcc (linguaggio CUDA) senza rinominare il file in .cu
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/backends/alpaka/gemm_alpaka_naive.cpp
    PROPERTIES LANGUAGE CUDA
  )

  target_compile_options(gemm_backend_alpaka PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )
endif()
