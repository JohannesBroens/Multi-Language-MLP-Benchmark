cmake_minimum_required(VERSION 3.10)

option(USE_CUDA "Build with CUDA support" OFF)

if(USE_CUDA)
    project(MLinC LANGUAGES C CUDA)
else()
    project(MLinC LANGUAGES C)
endif()

set(CMAKE_C_STANDARD 99)

# Default to Release build for optimized performance (-O3)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable native arch (AVX2/FMA) and fast math for SIMD auto-vectorization
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native -ffast-math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

# Add subdirectories
add_subdirectory(nn_ops)
add_subdirectory(models/mlp)
add_subdirectory(models/cnn)

# MLP executable
add_executable(main main.c data_loader.c)
target_link_libraries(main PRIVATE mlp nn_ops m)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CNN executable
add_executable(cnn_main cnn_main.c data_loader.c)
target_link_libraries(cnn_main PRIVATE cnn nn_ops m)
target_include_directories(cnn_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(USE_CUDA)
    message(STATUS "Building with CUDA support")
    target_compile_definitions(main PRIVATE USE_CUDA)
    target_compile_definitions(cnn_main PRIVATE USE_CUDA)
else()
    message(STATUS "Building without CUDA support")
endif()
