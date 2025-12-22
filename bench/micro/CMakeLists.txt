
# bench/micro/CMakeLists.txt

set(PROJ_INC_DIR ${PROJECT_SOURCE_DIR}/src)

# ---------- CUDA benchmark ----------
if(ENABLE_CUDA)
  add_executable(benchmark_cuda
    benchmark_test.cu
  )

  target_include_directories(benchmark_cuda PRIVATE ${PROJ_INC_DIR})
  target_link_libraries(benchmark_cuda PRIVATE gemm_backend_cuda CUDA::cudart)
endif()

# ---------- Alpaka benchmark ----------
#
# set_source_files_properties(
#     benchmark_alpaka.cpp
#         PROPERTIES LANGUAGE CUDA
#         )
if(ENABLE_ALPAKA)
  add_executable(benchmark_alpaka
    benchmark_alpaka.cu
  )

  target_include_directories(benchmark_alpaka PRIVATE ${PROJ_INC_DIR})
  target_link_libraries(benchmark_alpaka PRIVATE gemm_backend_alpaka CUDA::cudart)

  if(ENABLE_ALPAKA AND ALPAKA_ACC_GPU_CUDA_ENABLED)
    target_compile_options(benchmark_alpaka PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
  endif()
  # Questo può rimanere CXX “normale”: il device code sta già nella lib Alpaka compilata con nvcc.
  # Se invece in benchmark_alpaka.cpp includi roba Alpaka che richiede __device__/builtins,
  # allora anche questo file va compilato come CUDA language:
  #
  # set_source_files_properties(
  #   ${PROJECT_SOURCE_DIR}/bench/micro/benchmark_alpaka.cpp
  #   PROPERTIES LANGUAGE CUDA
  # )
endif()

