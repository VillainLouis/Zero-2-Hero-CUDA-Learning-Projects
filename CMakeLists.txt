cmake_minimum_required(VERSION 3.20.0)
project(cuda_practice VERSION 0.1.0 LANGUAGES CUDA CXX C)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(01_Reduce)