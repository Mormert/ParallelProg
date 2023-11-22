cmake_minimum_required(VERSION 3.8.2)

project(CUDALAB CXX)

enable_language(CUDA)

add_executable(CUDALAB_Simple helper.c simple.cu)
add_executable(CUDALAB_MatrixPow helper.c Matrix_pow.cu)
add_executable(CUDALAB_MatrixMult helper.c Matrix_mult.cu)

set_property(TARGET CUDALAB_Simple PROPERTY CUDA_STANDARD 17 CXX_STANDARD 17)
set_property(TARGET CUDALAB_MatrixPow PROPERTY CUDA_STANDARD 17 CXX_STANDARD 17)
set_property(TARGET CUDALAB_MatrixMult PROPERTY CUDA_STANDARD 17 CXX_STANDARD 17)

