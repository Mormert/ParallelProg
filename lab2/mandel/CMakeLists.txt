

cmake_minimum_required(VERSION 3.26)

project(mandelOpenMP)

set(CMAKE_CXX_STANDARD 17)


add_executable(mandelOpenMP
        mandel.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mandelOpenMP PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(mandelOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
    target_compile_options(mandelOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
endif()