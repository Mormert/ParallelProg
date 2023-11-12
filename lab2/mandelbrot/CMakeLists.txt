cmake_minimum_required(VERSION 3.26)

project(mandelbrotOpenMP)

set(CMAKE_CXX_STANDARD 17)

add_executable(mandelbrotOpenMP
        mandelbrot.cpp
        stb_image_write.cpp
        WickedEngine/wiJobSystem.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mandelbrotOpenMP PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(mandelbrotOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
    target_compile_options(mandelbrotOpenMP PRIVATE "${OpenMP_CXX_FLAGS}")
endif()