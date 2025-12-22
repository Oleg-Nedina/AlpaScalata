
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

  # IMPORTANT on clusters: avoid "Failed to detect a default CUDA architecture"
  # You can override at configure time with -DCMAKE_CUDA_ARCHITECTURES=89 (L4) etc.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA archs" FORCE)
  endif()

  # Determine CUDA root for rpath:
  # Prefer CUDA_HOME if given, else fall back to CUDAToolkit_ROOT (what you already pass)
  if(NOT DEFINED CUDA_HOME)
    if(DEFINED CUDAToolkit_ROOT)
      set(CUDA_HOME "${CUDAToolkit_ROOT}" CACHE PATH "CUDA Toolkit root for RPATH" FORCE)
    endif()
  endif()

  # Global RPATH for all targets (executables shared libs)
  # This makes benchmark_cuda work even after `unset LD_LIBRARY_PATH`.
  if(DEFINED CUDA_HOME)
    list(APPEND CMAKE_BUILD_RPATH
      "${CUDA_HOME}/targets/x86_64-linux/lib"
      "${CUDA_HOME}/lib64"
    )
    list(APPEND CMAKE_INSTALL_RPATH
      "${CUDA_HOME}/targets/x86_64-linux/lib"
      "${CUDA_HOME}/lib64"
    )
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
  endif()
endif()

# -----------------------
# Helper: add CUDA RUNPATH to a specific target (optional)
# Use this if you want to enforce rpath only for CUDA targets, not globally.
# With the global CMAKE_BUILD_RPATH above, calling this is usually not necessary.
# -----------------------
function(target_add_cuda_runpath tgt)
  if(ENABLE_CUDA AND DEFINED CUDA_HOME)
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
