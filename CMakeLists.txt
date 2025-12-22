cmake_minimum_required(VERSION 3.25)

# Se vuoi che CUDA sia "first class" già dal project(), puoi fare:
# project(AlpaScalata LANGUAGES CXX CUDA)
# ma lo facciamo condizionale con enable_language(CUDA) come prima.
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

  # Arch: imposta default se non specificata a configure time
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA archs" FORCE)
  endif()

  # Root CUDA per RPATH
  if(NOT DEFINED CUDA_HOME)
    if(DEFINED CUDAToolkit_ROOT)
      set(CUDA_HOME "${CUDAToolkit_ROOT}" CACHE PATH "CUDA Toolkit root for RPATH" FORCE)
    endif()
  endif()

  # RPATH globale (aiuta tantissimo su cluster)
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
# Alpaka (optional)
# -----------------------
if(ENABLE_ALPAKA)
  # Alpaka deve avere il suo CMakeLists dentro external/alpaka
  add_subdirectory(external/alpaka)
endif()

# -----------------------
# Common headers
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

# -----------------------
# IMPORTANT: Force specific Alpaka TU to compile with NVCC when CUDA backend is enabled
# -----------------------
if(ENABLE_ALPAKA AND ENABLE_CUDA)
  # Questi due file sono .cpp ma DEVONO essere compilati da nvcc
  # quando alpaka usa TagGpuCudaRt (altrimenti ottieni gli errori "compiler has to support CUDA").
  set(_alpaka_cuda_tus
    "${CMAKE_SOURCE_DIR}/src/backends/alpaka/gemm_alpaka_naive.cpp"
    "${CMAKE_SOURCE_DIR}/bench/micro/benchmark_alpaka.cpp"
  )

  foreach(_tu IN LISTS _alpaka_cuda_tus)
    if(EXISTS "${_tu}")
      set_source_files_properties("${_tu}" PROPERTIES LANGUAGE CUDA)
    endif()
  endforeach()

  # Linker language: evita che il link venga fatto come CXX puro
  # (utile soprattutto per gli executable bench).
  if(TARGET gemm_backend_alpaka)
    set_target_properties(gemm_backend_alpaka PROPERTIES LINKER_LANGUAGE CUDA)
  endif()

  if(TARGET benchmark_alpaka)
    set_target_properties(benchmark_alpaka PROPERTIES LINKER_LANGUAGE CUDA)
  endif()
endif()

if(ENABLE_CUDA AND DEFINED CUDA_HOME)
  set_target_properties(benchmark_cuda PROPERTIES
    BUILD_RPATH   "${CUDA_HOME}/targets/x86_64-linux/lib;${CUDA_HOME}/lib64"
    INSTALL_RPATH "${CUDA_HOME}/targets/x86_64-linux/lib;${CUDA_HOME}/lib64"
  )
endif()

