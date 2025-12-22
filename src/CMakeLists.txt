# CUDA backend library
if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/gemm_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/gemm_cuda_naive.cu
  )
  target_link_libraries(gemm_backend_cuda PUBLIC gemm_headers CUDA::cudart)
  target_compile_features(gemm_backend_cuda PUBLIC cxx_std_20)
  target_add_cuda_runpath(gemm_backend_cuda)
endif()

# Alpaka backend library
if(ENABLE_ALPAKA)
  add_library(gemm_backend_alpaka STATIC
  backends/alpaka/gemm_alpaka_naive.cpp
)
  target_link_libraries(gemm_backend_alpaka PUBLIC alpaka::alpaka)
  target_include_directories(gemm_backend_alpaka PUBLIC ${PROJECT_SOURCE_DIR}/include)

  # Se Alpaka CUDA è abilitato, compila questo TU con nvcc (LANGUAGE CUDA)
  if(alpaka_ACC_GPU_CUDA_ENABLE)
    set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/alpaka/gemm_alpaka_naive.cpp
    PROPERTIES LANGUAGE CUDA
  )
    # opzionale ma utile per evitare ambiguità
    set_target_properties(gemm_backend_alpaka PROPERTIES LINKER_LANGUAGE CUDA)
  endif()
endif()



add_library(gemm_backend_alpaka
  gemm_alpaka_naive.cpp
)

target_link_libraries(gemm_backend_alpaka PUBLIC gemm_headers alpaka::alpaka)

if(alpaka_ACC_GPU_CUDA_ENABLE)
  set_source_files_properties(gemm_alpaka_naive.cpp PROPERTIES LANGUAGE CUDA)
endif()


