
# src/CMakeLists.txt

add_library(gemm_interface INTERFACE)
target_include_directories(gemm_interface INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Alpaka header-only in this repo
add_library(alpaka_headers INTERFACE)
target_include_directories(alpaka_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/alpaka/include
)

# ---------------- CUDA backend ----------------
if(ENABLE_CUDA)
  add_library(gemm_backend_cuda STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/gemm_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/gemm_cuda_naive.cu
  )
  target_link_libraries(gemm_backend_cuda PRIVATE gemm_interface CUDA::cudart)
  target_compile_features(gemm_backend_cuda PRIVATE cxx_std_17)
  set_target_properties(gemm_backend_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---------------- Alpaka backend ----------------
if(ENABLE_ALPAKA)
  add_library(gemm_backend_alpaka STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/alpaka/gemm_alpaka_naive.cpp
  )
  target_link_libraries(gemm_backend_alpaka PRIVATE gemm_interface alpaka_headers CUDA::cudart)
  target_compile_features(gemm_backend_alpaka PRIVATE cxx_std_17)

  # Enable Alpaka accelerators (CPU always available; CUDA only when compiled with nvcc)
  target_compile_definitions(gemm_backend_alpaka PRIVATE
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED=1
    ALPAKA_ACC_GPU_CUDA_ENABLED=1
  )

  # IMPORTANT:
  # Keep the file extension .cpp, but compile THIS TU with nvcc so Alpaka can use CUDA builtins.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/alpaka/gemm_alpaka_naive.cpp
    PROPERTIES LANGUAGE CUDA
  )
endif()
