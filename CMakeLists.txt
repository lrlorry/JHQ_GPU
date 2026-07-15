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
add_jhq_version(hblock_v15)
add_jhq_version(hblock_v16)
add_jhq_version(hblock_v17)

# v18: JL coarse scan — no encode.cu, added manually
add_library(jhq_hblock_v18 STATIC
    hblock_v18/search.cu
    hblock_v18/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v18 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v18 PUBLIC
    jhq_hblock_v17   # reuses encode.cu kernels
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v18 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_hblock_v18 examples/demo_hblock_v18.cu)
target_link_libraries(demo_hblock_v18 PRIVATE jhq_hblock_v18)

# v19: split d_proj_route/d_proj_scan + empty-centroid fix + K_MAX=192
add_library(jhq_hblock_v19 STATIC
    hblock_v19/search.cu
    hblock_v19/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v19 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v19 PUBLIC
    jhq_hblock_v17   # reuses encode.cu kernels
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v19 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_hblock_v19 examples/demo_hblock_v19.cu)
target_link_libraries(demo_hblock_v19 PRIVATE jhq_hblock_v19)

# v20: v17 + d_proj=128 routing + empty-centroid norm=1e30f fix
add_library(jhq_hblock_v20 STATIC
    hblock_v20/search.cu
    hblock_v20/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v20 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v20 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v20 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_hblock_v20 examples/demo_hblock_v20.cu)
target_link_libraries(demo_hblock_v20 PRIVATE jhq_hblock_v20)

# v21: v17 + sparse subtree bitmask routing (valid_c2/valid_c3 per parent)
add_library(jhq_hblock_v21 STATIC
    hblock_v21/search.cu
    hblock_v21/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v21 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v21 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v21 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_hblock_v21 examples/demo_hblock_v21.cu)
target_link_libraries(demo_hblock_v21 PRIVATE jhq_hblock_v21)

# v30: v29 + fix visited-before-insert bug (check bmax before atomicOr, unblocks graph paths)
add_library(jhq_hblock_v30 STATIC
    hblock_v30/search.cu
    hblock_v30/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v30 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v30 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v30 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v30 examples/demo_hblock_v30.cu)
target_link_libraries(demo_hblock_v30 PRIVATE jhq_hblock_v30)

# v31 diagnostic: missed-GT classification (routing/connectivity/depth) via BFS
add_executable(demo_hblock_v31_diag examples/demo_hblock_v31_diag.cu)
target_link_libraries(demo_hblock_v31_diag PRIVATE jhq_hblock_v30)

# v32: single ef parameter; v30 semantics internally (ef→depth=ef,beam=min(ef,128))
add_library(jhq_hblock_v32 STATIC
    hblock_v32/search.cu
    hblock_v32/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v32 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v32 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v32 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v32 examples/demo_hblock_v32.cu)
target_link_libraries(demo_hblock_v32 PRIVATE jhq_hblock_v32)

# v33: W=4 batched beam expansion + HNSW early termination, beam=ef (no cap)
add_library(jhq_hblock_v33 STATIC
    hblock_v33/search.cu
    hblock_v33/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v33 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v33 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v33 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v33 examples/demo_hblock_v33.cu)
target_link_libraries(demo_hblock_v33 PRIVATE jhq_hblock_v33)

# v29: v28 + balanced k-means (each cluster ≤ 128 → cluster = block, clean centroid)
add_library(jhq_hblock_v29 STATIC
    hblock_v29/search.cu
    hblock_v29/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v29 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v29 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v29 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
add_executable(demo_hblock_v29 examples/demo_hblock_v29.cu)
target_link_libraries(demo_hblock_v29 PRIVATE jhq_hblock_v29)

# v28: per-block exact rerank (PQ filter → exact L2 within block → merge)
add_library(jhq_hblock_v28 STATIC
    hblock_v28/search.cu
    hblock_v28/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v28 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v28 PUBLIC
    jhq_hblock_v27   # for gpu_build_block_adj_v27 and encode kernels
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v28 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
add_executable(demo_hblock_v28 examples/demo_hblock_v28.cu)
target_link_libraries(demo_hblock_v28 PRIVATE jhq_hblock_v28)

# v27: v26 + bitonic warp sort in leaf_flat → top-32 per block (restore QPS, keep quality)
add_library(jhq_hblock_v27 STATIC
    hblock_v27/search.cu
    hblock_v27/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v27 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v27 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v27 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
add_executable(demo_hblock_v27 examples/demo_hblock_v27.cu)
target_link_libraries(demo_hblock_v27 PRIVATE jhq_hblock_v27)

# v26: v25 + template beam (32/64/128 slots) + global top-rerank_r (no per-block top_p)
add_library(jhq_hblock_v26 STATIC
    hblock_v26/search.cu
    hblock_v26/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v26 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v26 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v26 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
add_executable(demo_hblock_v26 examples/demo_hblock_v26.cu)
target_link_libraries(demo_hblock_v26 PRIVATE jhq_hblock_v26)

# v25: v24 + semantic block construction via mini k-means within each L3 cell
add_library(jhq_hblock_v25 STATIC
    hblock_v25/search.cu
    hblock_v25/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v25 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v25 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v25 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
add_executable(demo_hblock_v25 examples/demo_hblock_v25.cu)
target_link_libraries(demo_hblock_v25 PRIVATE jhq_hblock_v25)

# v24: v23 + oracle diagnostic + TOP_P/beam_size runtime sweep + no route D2H
add_library(jhq_hblock_v24 STATIC
    hblock_v24/search.cu
    hblock_v24/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v24 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v24 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v24 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
add_executable(demo_hblock_v24 examples/demo_hblock_v24.cu)
target_link_libraries(demo_hblock_v24 PRIVATE jhq_hblock_v24)

# v23: block-level graph with true block centroids + hierarchical candidate cells
add_library(jhq_hblock_v23 STATIC
    hblock_v23/search.cu
    hblock_v23/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v23 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v23 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v23 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)
add_executable(demo_hblock_v23 examples/demo_hblock_v23.cu)
target_link_libraries(demo_hblock_v23 PRIVATE jhq_hblock_v23)

# v22: L1+L2 GPU beam + L3 CPU graph traversal over leaf cells
add_library(jhq_hblock_v22 STATIC
    hblock_v22/search.cu
    hblock_v22/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v22 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v22 PUBLIC
    jhq_hblock_v17
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v22 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_hblock_v22 examples/demo_hblock_v22.cu)
target_link_libraries(demo_hblock_v22 PRIVATE jhq_hblock_v22)
