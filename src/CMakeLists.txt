# src/CMakeLists.txt

set(PROJ_INC_DIRS
  ${PROJECT_SOURCE_DIR}          # se hai gemm/ direttamente in root
  ${PROJECT_SOURCE_DIR}/src      # se hai src/gemm/...
  ${PROJECT_SOURCE_DIR}/include  # se hai include/gemm/...
  ${PROJECT_SOURCE_DIR}/src/include # se hai src/include/gemm/...
)

if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
    backends/cuda/gemm_cuda.cu
    backends/cuda/gemm_cuda_naive.cu
  )
  target_include_directories(gemm_backend_cuda PUBLIC ${PROJ_INC_DIRS})
  target_link_libraries(gemm_backend_cuda PUBLIC CUDA::cudart)
  target_compile_options(gemm_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )
endif()

if(ENABLE_ALPAKA)
  add_library(gemm_backend_alpaka STATIC
    backends/alpaka/gemm_alpaka_naive.cpp
  )
  target_include_directories(gemm_backend_alpaka PUBLIC ${PROJ_INC_DIRS})
  target_link_libraries(gemm_backend_alpaka PUBLIC alpaka CUDA::cudart)

  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/backends/alpaka/gemm_alpaka_naive.cpp
    PROPERTIES LANGUAGE CUDA
  )

  target_compile_options(gemm_backend_alpaka PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
  )
endif()

