cmake_minimum_required(VERSION 3.25)

project(AlpaScalata LANGUAGES CXX)

option(ENABLE_CUDA   "Build CUDA backend/bench" ON)
option(ENABLE_ALPAKA "Build Alpaka backend/bench" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------
# CUDA toolchain
# -----------------------
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # override with -DCMAKE_CUDA_ARCHITECTURES=89 etc.
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA archs" FORCE)
  endif()

  # RPATH per cluster (build + install)
  # (funziona senza funzioni custom)
  list(APPEND CMAKE_BUILD_RPATH
    "${CUDAToolkit_LIBRARY_DIR}"
  )
  list(APPEND CMAKE_INSTALL_RPATH
    "${CUDAToolkit_LIBRARY_DIR}"
  )
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# -----------------------
# Alpaka
# -----------------------
if(ENABLE_ALPAKA)
  # Qui serve che external/alpaka contenga CMakeLists.txt (repo alpaka completo)
  add_subdirectory(external/alpaka)
endif()

# -----------------------
# Common headers (tuo include/)
# -----------------------
add_library(gemm_headers INTERFACE)
target_include_directories(gemm_headers INTERFACE
  ${CMAKE_SOURCE_DIR}/include
)

# -----------------------
# Subdirs
# -----------------------
add_subdirectory(src)

add_subdirectory(bench)
