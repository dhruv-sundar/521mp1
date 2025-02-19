cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cs521_mp1 LANGUAGES CUDA CXX)

# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

add_executable(mp1_cpu ../cpu/gemm_cpu.cpp)
# add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
#set_target_properties(mp1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(mp1 CUDA::cudart)
