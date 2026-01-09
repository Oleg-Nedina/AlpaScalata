
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-12")
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "CUDA host compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_FLAGS)
  set(CMAKE_CUDA_FLAGS "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA flags" FORCE)
endif()

project(AlpaScalata LANGUAGES CXX CUDA)

option(ENABLE_CUDA   "Build CUDA backend"   ON)
option(ENABLE_ALPAKA "Build Alpaka backend" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(ENABLE_CUDA OR ENABLE_ALPAKA)
  find_package(CUDAToolkit REQUIRED)
endif()

add_library(alpaka INTERFACE)
target_include_directories(alpaka SYSTEM INTERFACE
  ${PROJECT_SOURCE_DIR}/external/alpaka/include
)

target_compile_definitions(alpaka INTERFACE
  ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED=1
)

if(ENABLE_ALPAKA)
  target_compile_definitions(alpaka INTERFACE
    ALPAKA_ACC_GPU_CUDA_ENABLED=1
  )
endif()

# ---- Subdirs ----
add_subdirectory(src)
add_subdirectory(bench/micro)
#add_subdirectory(include)
