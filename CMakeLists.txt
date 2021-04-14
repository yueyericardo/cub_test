cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(cub_test LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# default CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures" FORCE)
set(CMAKE_CUDA_FLAGS "--extended-lambda")

find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} "cub_test.cu")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

message(STATUS "build type:  ${CMAKE_BUILD_TYPE}")