

cmake_minimum_required(VERSION 3.26)

project(ParallelQuicksort)

set(CMAKE_CXX_STANDARD 17)


add_executable(ParallelQuicksort
        qs.cpp
        WickedEngine/wiJobSystem.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ParallelQuicksort PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(ParallelQuicksort PRIVATE "${OpenMP_CXX_FLAGS}")
    target_compile_options(ParallelQuicksort PRIVATE "${OpenMP_CXX_FLAGS}" "-openmp:llvm")
endif()