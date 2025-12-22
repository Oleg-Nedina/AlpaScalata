# bench/micro/CMakeLists.txt

# CUDA benchmark exe
if(ENABLE_CUDA)
  add_executable(benchmark_cuda
    benchmark_test.cu
  )
  target_link_libraries(benchmark_cuda PRIVATE gemm_backend_cuda CUDA::cudart)
  target_compile_features(benchmark_cuda PRIVATE cxx_std_20)
endif()

# Alpaka benchmark exe
if(ENABLE_ALPAKA)
  add_executable(benchmark_alpaka
    benchmark_alpaka.cpp
  )
  target_link_libraries(benchmark_alpaka PRIVATE gemm_backend_alpaka alpaka::alpaka)
  target_compile_features(benchmark_alpaka PRIVATE cxx_std_20)

  # Se Alpaka CUDA è abilitato e CUDA è abilitato, benchmark_alpaka.cpp
  # DEVE essere compilato come CUDA (nvcc), anche se è .cpp.
  if(ENABLE_CUDA)
    if(DEFINED alpaka_ACC_GPU_CUDA_ENABLE AND alpaka_ACC_GPU_CUDA_ENABLE)
      set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_alpaka.cpp
        PROPERTIES LANGUAGE CUDA
      )
      set_target_properties(benchmark_alpaka PROPERTIES LINKER_LANGUAGE CUDA)
    endif()
  endif()
endif()

