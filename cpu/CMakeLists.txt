cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cs521_mp1 LANGUAGES CXX)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

if(OPT)
    message(STATUS "Optimizing for performance")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -mavx2 -mfma")
    message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
endif()

add_executable(mp1_cpu gemm_cpu.cpp)
