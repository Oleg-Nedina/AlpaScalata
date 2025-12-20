
# bench/micro/CMakeLists.txt

# CUDA benchmark (esiste già: benchmark_test.cu)
if(ENABLE_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_test.cu")
  add_executable(benchmark_cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_test.cu
  )
  target_link_libraries(benchmark_cuda PRIVATE gemm_headers)

  if(TARGET gemm_backend_cuda)
    target_link_libraries(benchmark_cuda PRIVATE gemm_backend_cuda)
  endif()

  target_link_libraries(benchmark_cuda PRIVATE CUDA::cudart)
  target_add_cuda_runpath(benchmark_cuda)
endif()

# Alpaka benchmark (lo aggiungeremo dopo, quando esiste il file)
if(ENABLE_ALPAKA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_alpaka.cpp")
  add_executable(benchmark_alpaka
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_alpaka.cpp
  )
  target_link_libraries(benchmark_alpaka PRIVATE gemm_headers)

  if(TARGET gemm_backend_alpaka)
    target_link_libraries(benchmark_alpaka PRIVATE gemm_backend_alpaka)
  endif()

  # se Alpaka usa CUDA backend sul cluster
  if(ENABLE_CUDA)
    target_link_libraries(benchmark_alpaka PRIVATE CUDA::cudart)
    target_add_cuda_runpath(benchmark_alpaka)
  endif()
endif()
