
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Set a CUDA 12+ compatible default to avoid sm_52 during compiler ID.
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-12")
  # nvcc 12.x does not support GCC 13+ as host compiler.
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "CUDA host compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_FLAGS)
  # Force an explicit arch for the CUDA compiler ID step (nvcc defaults to sm_52).
  set(CMAKE_CUDA_FLAGS "-arch=sm_${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA flags" FORCE)
endif()
project(AlpaScalata LANGUAGES CXX CUDA)

option(ENABLE_CUDA   "Build CUDA backend"   ON)
option(ENABLE_ALPAKA "Build Alpaka backend" ON)

# ---- C++ / CUDA standards (Alpaka vuole C++20) ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Evita estensioni strane
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Se vuoi, per build più pulite
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- CUDA Toolkit ----
if(ENABLE_CUDA OR ENABLE_ALPAKA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---- Alpaka as header-only (se NON hai un CMakeLists dentro external/alpaka) ----
# Struttura attesa: external/alpaka/include/alpaka/alpaka.hpp
add_library(alpaka INTERFACE)
target_include_directories(alpaka INTERFACE
  ${PROJECT_SOURCE_DIR}/external/alpaka/include
)

# Se vuoi CPU serial come fallback
target_compile_definitions(alpaka INTERFACE
  ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED=1
)

# Se vuoi abilitare Alpaka su CUDA
if(ENABLE_ALPAKA)
  target_compile_definitions(alpaka INTERFACE
    ALPAKA_ACC_GPU_CUDA_ENABLED=1
  )
endif()

# ---- Subdirs ----
add_subdirectory(src)
add_subdirectory(bench/micro)
#add_subdirectory(include)
