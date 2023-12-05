cmake_minimum_required(VERSION 3.8.2)

project(CUDALAB CXX)

enable_language(CUDA)

add_executable(CUDALAB_ImageConv helper.cpp imagConv.cu )

set_property(TARGET CUDALAB_ImageConv PROPERTY CUDA_STANDARD 17 CXX_STANDARD 17)

