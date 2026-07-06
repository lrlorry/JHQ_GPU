cmake_minimum_required(VERSION 3.18)
project(JHQ_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Avoid CMake/NVCC combinations that translate "native" into unsupported
# "compute_native". Override with -DCMAKE_CUDA_ARCHITECTURES=<capability>
# for the exact server GPU, e.g. 75, 80, 86, 89.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ── Shared CPU library ────────────────────────────────────────────────────────
# All versions link this. Never duplicated.
add_library(jhq_cpu STATIC
    cpu/codebook.cpp
    cpu/jl_transform.cpp
)
target_include_directories(jhq_cpu PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_cpu PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_compile_options(jhq_cpu PRIVATE -O3 -march=native)

# ── Helper macro: add one GPU version + its demo ─────────────────────────────
# Usage: add_jhq_version(jhq_v1_plain)
#        add_jhq_version(v2_batched_query)
#        ...
# Expects files: <ver>/encode.cu  <ver>/search.cu  <ver>/jhq_gpu_index.cu
#                examples/demo_<ver>.cu
macro(add_jhq_version VER)
    add_library(jhq_${VER} STATIC
        ${VER}/encode.cu
        ${VER}/search.cu
        ${VER}/jhq_gpu_index.cu
    )
    target_include_directories(jhq_${VER} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(jhq_${VER} PUBLIC
        jhq_cpu
        CUDA::cublas
        CUDA::cudart
    )
    target_compile_options(jhq_${VER} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    )

    add_executable(demo_${VER} examples/demo_${VER}.cu)
    target_link_libraries(demo_${VER} PRIVATE jhq_${VER})
endmacro()

# ── Register versions here ────────────────────────────────────────────────────
add_jhq_version(jhq_v1_plain)
add_jhq_version(jhq_v2_topk)
add_jhq_version(jhq_v3_ivf)
add_jhq_version(jhq_v4_batched_query)
add_jhq_version(jhq_v5_cuda_graph)
add_jhq_version(jhq_v6_async_h2d)
add_jhq_version(jhq_v7_spin_sync)
add_jhq_version(jhq_v8_timing)
add_jhq_version(jhq_v9_step_timing)
add_jhq_version(jhq_v10_bytelut)
add_jhq_version(jhq_v11_outerlut)
add_jhq_version(jhq_v12_transposed)
add_jhq_version(jhq_v13_step_timing)
add_jhq_version(hblock_v1)
add_jhq_version(hblock_v2)
add_jhq_version(hblock_v3)
add_jhq_version(hblock_v4)
add_jhq_version(hblock_v5)
add_jhq_version(hblock_v6)
add_jhq_version(hblock_v7)
add_jhq_version(hblock_v8)
add_jhq_version(hblock_v9)
add_jhq_version(hblock_v10)
add_jhq_version(hblock_v11)
add_jhq_version(hblock_v12)
add_jhq_version(hblock_v13)
add_jhq_version(hblock_v14)
