
# bench/micro/CMakeLists.txt

# CUDA benchmark
if(ENABLE_CUDA)
  add_executable(benchmark_cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_test.cu
  )
  target_link_libraries(benchmark_cuda PRIVATE gemm_backend_cuda CUDA::cudart)
  target_compile_features(benchmark_cuda PRIVATE cxx_std_17)
  target_add_cuda_runpath(benchmark_cuda)
endif()

# Alpaka benchmark (host code!)
if(ENABLE_ALPAKA)
  add_executable(benchmark_alpaka
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_alpaka.cpp
  )
  target_link_libraries(benchmark_alpaka PRIVATE gemm_backend_alpaka CUDA::cudart)
  target_compile_features(benchmark_alpaka PRIVATE cxx_std_17)

  # Do NOT define ALPAKA_ACC_GPU_CUDA_ENABLED for this target if it's compiled with g++ and includes alpaka.hpp.
  # Ideally: benchmark_alpaka.cpp should NOT include alpaka/alpaka.hpp at all.
  target_add_cuda_runpath(benchmark_alpaka)
endif()
