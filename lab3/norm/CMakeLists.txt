

cmake_minimum_required(VERSION 3.26)

project(normOpenMPDAQ)

set(CMAKE_CXX_STANDARD 17)


add_executable(normOpenMPDAQ
        norm.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(normOpenMPDAQ PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(normOpenMPDAQ PRIVATE "${OpenMP_CXX_FLAGS}")
    target_compile_options(normOpenMPDAQ PRIVATE "${OpenMP_CXX_FLAGS}" "-openmp:llvm")
endif()