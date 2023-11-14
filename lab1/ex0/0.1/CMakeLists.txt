cmake_minimum_required(VERSION 3.26)

project(Ex01_01)

set(CMAKE_CXX_STANDARD 17)

# Uses Vcpkg for this!
find_package(PThreads4W REQUIRED)

include_directories(${PTHREAD_INCLUDE_DIR})

add_executable(Ex01_01
        main.cpp)

# Windows port of pthreads...
target_link_libraries(Ex01_01 PRIVATE PThreads4W::PThreads4W)
