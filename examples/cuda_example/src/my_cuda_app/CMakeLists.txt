cmake_minimum_required(VERSION 3.23)
project(cuda_ext CXX CUDA)

SET(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)


find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

include_directories(include)
add_library(cuda_ext
        cudaext.cu
        cudaext_wrapper.cpp
        )
set_target_properties(cuda_ext PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE "CUDA"
        )
target_link_libraries(cuda_ext ${PYTHON_LIBRARIES})

add_executable(cuda_main main.cpp)
target_link_libraries(cuda_main cuda_ext)
set_target_properties(cuda_main PROPERTIES
        LINKER_LANGUAGE "CXX"
        )