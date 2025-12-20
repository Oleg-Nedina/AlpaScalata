cmake_minimum_required(VERSION 3.25)
project(alpaka_smoke_test CXX)

add_subdirectory(thirdParty/alpaka)

alpaka_add_executable(hello_alpaka src/helloWorld.cpp)
target_link_libraries(hello_alpaka PUBLIC alpaka::alpaka)
set(CMAKE_BUILD_RPATH
    "/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap/targets/x86_64-linux/lib"
    "/software/spack-v1.0/opt/spack/linux-cascadelake/cuda-13.0.2-flsbrpd2nhr3wflionjcydwr5hhttjap/lib64"
)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

