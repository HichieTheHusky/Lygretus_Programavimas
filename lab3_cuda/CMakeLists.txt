cmake_minimum_required(VERSION 3.17)
project(lab3_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-9")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/gcc-9")

add_executable(lab3_cuda main.cu)

set_target_properties(
        lab3_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)