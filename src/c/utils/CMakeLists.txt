cmake_minimum_required(VERSION 3.10)

# --- nn_ops library ---
if(USE_CUDA)
    message(STATUS "Building nn_ops with CUDA support")
    add_library(nn_ops SHARED nn_ops.cu)
    target_compile_definitions(nn_ops PRIVATE USE_CUDA)
    set_target_properties(nn_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET nn_ops PROPERTY CUDA_ARCHITECTURES native)

    find_library(CUBLAS_LIB cublas HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} /usr/local/cuda/lib64)
    if(CUBLAS_LIB)
        target_link_libraries(nn_ops PUBLIC ${CUBLAS_LIB})
    endif()
else()
    message(STATUS "Building nn_ops without CUDA support")
    add_library(nn_ops SHARED nn_ops_cpu.c)

    find_package(OpenMP)
    if(OPENMP_FOUND)
        target_link_libraries(nn_ops PUBLIC OpenMP::OpenMP_C)
    endif()

    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        target_link_libraries(nn_ops PUBLIC ${BLAS_LIBRARIES})
        target_compile_definitions(nn_ops PRIVATE USE_BLAS)
        find_path(CBLAS_INCLUDE_DIR cblas.h PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/local/include)
        if(CBLAS_INCLUDE_DIR)
            target_include_directories(nn_ops PRIVATE ${CBLAS_INCLUDE_DIR})
        endif()
    endif()

    target_link_libraries(nn_ops PUBLIC m)
endif()

target_include_directories(nn_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- data_loader (compiled into executables as object library) ---
add_library(data_loader OBJECT data_loader.c)
target_include_directories(data_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
