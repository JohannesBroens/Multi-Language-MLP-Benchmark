cmake_minimum_required(VERSION 3.10)

if(USE_CUDA)
    message(STATUS "Building MLP with CUDA support")
    add_library(mlp SHARED mlp.cu)
    target_compile_definitions(mlp PRIVATE USE_CUDA)
    set_target_properties(mlp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET mlp PROPERTY CUDA_ARCHITECTURES native)

    # Link cuBLAS for batched GEMM operations
    find_library(CUBLAS_LIB cublas HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} /usr/local/cuda/lib64)
    if(CUBLAS_LIB)
        target_link_libraries(mlp PUBLIC ${CUBLAS_LIB})
        message(STATUS "  cuBLAS: ${CUBLAS_LIB}")
    else()
        message(FATAL_ERROR "cuBLAS not found -- required for CUDA build")
    endif()
else()
    message(STATUS "Building MLP without CUDA support")
    add_library(mlp SHARED mlp_cpu.c)
    target_link_libraries(mlp PUBLIC nn_ops)
endif()

target_include_directories(mlp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
