# -----------------------
# CUDA backend library
# -----------------------
if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
       ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/gemm_cuda.cu
           ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/gemm_cuda_naive.cu
             )

  target_link_libraries(gemm_backend_cuda PUBLIC gemm_headers CUDA::cudart)
  target_compile_features(gemm_backend_cuda PUBLIC cxx_std_20)

  # embed RUNPATH for libcudart (cluster)
  target_add_cuda_runpath(gemm_backend_cuda)
endif()

# -----------------------
# Alpaka backend library
# -----------------------
if(ENABLE_ALPAKA)
  set(ALPAKA_GEMM_SRC
       ${CMAKE_CURRENT_SOURCE_DIR}/backends/alpaka/gemm_alpaka_naive.cpp
                             )

  # If Alpaka CUDA accelerator enabled, compile this .cpp with nvcc
  if(alpaka_ACC_GPU_CUDA_ENABLE)
    set_source_files_properties(${ALPAKA_GEMM_SRC} PROPERTIES LANGUAGE CUDA)
  endif()

  add_library(gemm_backend_alpaka STATIC ${ALPAKA_GEMM_SRC})

  target_link_libraries(gemm_backend_alpaka PUBLIC gemm_headers alpaka::alpaka)
  target_compile_features(gemm_backend_alpaka PUBLIC cxx_std_20)

  # Optional: make sure final link uses CUDA toolchain when that TU is CUDA
  if(alpaka_ACC_GPU_CUDA_ENABLE)
    set_target_properties(gemm_backend_alpaka PROPERTIES LINKER_LANGUAGE CUDA)

    # If you want "unset LD_LIBRARY_PATH" to still work:
    target_add_cuda_runpath(gemm_backend_alpaka)
  endif()
endif()

