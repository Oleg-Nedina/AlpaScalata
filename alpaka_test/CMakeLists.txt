cmake_minimum_required(VERSION 3.25)
project(alpaka_smoke_test CXX)

add_subdirectory(thirdParty/alpaka)

alpaka_add_executable(hello_alpaka src/helloWorld.cpp)
target_link_libraries(hello_alpaka PUBLIC alpaka::alpaka)

