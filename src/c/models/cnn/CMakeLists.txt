cmake_minimum_required(VERSION 3.10)

if(USE_CUDA)
    add_library(cnn SHARED cnn.cu)
    target_compile_definitions(cnn PRIVATE USE_CUDA)
    set_target_properties(cnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET cnn PROPERTY CUDA_ARCHITECTURES native)
else()
    add_library(cnn SHARED cnn_cpu.c)
endif()

target_link_libraries(cnn PUBLIC nn_ops)
target_include_directories(cnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
