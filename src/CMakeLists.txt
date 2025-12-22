
# src/CMakeLists.txt

# CUDA backend library
if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
    backends/cuda/gemm_cuda.cu
    backends/cuda/gemm_cuda_naive.cu
  )
  target_link_libraries(gemm_backend_cuda PUBLIC gemm_headers CUDA::cudart)
  target_compile_features(gemm_backend_cuda PUBLIC cxx_std_20)
endif()

# Alpaka backend library
if(ENABLE_ALPAKA)
  add_library(gemm_backend_alpaka STATIC
    backends/alpaka/gemm_alpaka_naive.cpp
  )
  target_link_libraries(gemm_backend_alpaka PUBLIC gemm_headers alpaka::alpaka)
  target_compile_features(gemm_backend_alpaka PUBLIC cxx_std_20)

  # Se Alpaka CUDA è abilitato e CUDA è abilitato,
  # questo TU (.cpp) DEVE essere compilato con NVCC.
  if(ENABLE_CUDA)
    if(DEFINED alpaka_ACC_GPU_CUDA_ENABLE AND alpaka_ACC_GPU_CUDA_ENABLE)
      set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/alpaka/gemm_alpaka_naive.cpp
        PROPERTIES LANGUAGE CUDA
      )
      set_target_properties(gemm_backend_alpaka PROPERTIES LINKER_LANGUAGE CUDA)
    endif()
  endif()
endif()
