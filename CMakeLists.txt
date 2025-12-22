cmake_minimum_required(VERSION 3.24)

project(AlpaScalata LANGUAGES CXX CUDA)

option(ENABLE_CUDA   "Build CUDA backend"   ON)
option(ENABLE_ALPAKA "Build Alpaka backend" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default arch if not set from command line
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# CUDA Toolkit
if(ENABLE_CUDA OR ENABLE_ALPAKA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Small helper: embed CUDA lib path in RUNPATH so LD_LIBRARY_PATH is not needed
function(target_add_cuda_runpath tgt)
  if(NOT TARGET ${tgt})
    message(FATAL_ERROR "target_add_cuda_runpath: target '${tgt}' not found")
  endif()
  if(DEFINED CUDAToolkit_LIBRARY_DIR)
    set(_cuda_libdir "${CUDAToolkit_LIBRARY_DIR}")
  elseif(DEFINED CUDAToolkit_ROOT)
    set(_cuda_libdir "${CUDAToolkit_ROOT}/lib64")
  else()
    set(_cuda_libdir "")
  endif()

  if(_cuda_libdir)
    set_property(TARGET ${tgt} APPEND PROPERTY BUILD_RPATH "${_cuda_libdir}")
    set_property(TARGET ${tgt} APPEND PROPERTY INSTALL_RPATH "${_cuda_libdir}")
  endif()
endfunction()

add_subdirectory(src)
add_subdirectory(bench/micro)

