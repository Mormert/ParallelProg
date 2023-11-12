

cmake_minimum_required(VERSION 3.26)

project(introOpenMP)

add_executable(introOpenMP
        introOpenMP.c)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(introOpenMP PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(introOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
    target_compile_options(introOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
endif()