cmake_minimum_required(VERSION 3.25)
project(AlpaScalata LANGUAGES CXX)

option(ENABLE_CUDA   "Build CUDA backend + CUDA benchmark executable" ON)
option(ENABLE_ALPAKA "Build Alpaka backend + Alpaka benchmark executable" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------
# CUDA (optional)
# -----------------------
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# -----------------------
# Helper: embed CUDA runtime RUNPATH (no LD_LIBRARY_PATH)
# Provide -DCUDA_HOME=/software/spack... at configure time on cluster
# -----------------------
function(target_add_cuda_runpath tgt)
  if(DEFINED CUDA_HOME)
    set(_rpath
                 "${CUDA_HOME}/targets/x86_64-linux/lib"
                       "${CUDA_HOME}/lib64"
                           )
    set_target_properties(${tgt} PROPERTIES
                                     BUILD_RPATH   "${_rpath}"
                                           INSTALL_RPATH "${_rpath}"
                                               )
  endif()
endfunction()

# -----------------------
# Alpaka (optional)
# -----------------------
if(ENABLE_ALPAKA)
  add_subdirectory(external/alpaka)
endif()

# Common headers
add_library(gemm_headers INTERFACE)
target_include_directories(gemm_headers INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Source libraries + benchmarks
add_subdirectory(src)
add_subdirectory(bench)

