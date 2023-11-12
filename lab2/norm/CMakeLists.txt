

cmake_minimum_required(VERSION 3.26)

project(normOpenMP)

set(CMAKE_CXX_STANDARD 17)


add_executable(normOpenMP
        norm.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(normOpenMP PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(normOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
    target_compile_options(normOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
endif()