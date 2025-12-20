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
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/alpaka/gemm_alpaka_naive.cpp
  )
  target_link_libraries(gemm_backend_alpaka PUBLIC gemm_headers alpaka::alpaka)
  target_compile_features(gemm_backend_alpaka PUBLIC cxx_std_20)

  # If Alpaka uses CUDA backend on cluster, link cudart + runpath too
  if(ENABLE_CUDA)
    target_link_libraries(gemm_backend_alpaka PUBLIC CUDA::cudart)
    target_add_cuda_runpath(gemm_backend_alpaka)
  endif()
endif()

