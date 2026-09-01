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
# Codebook training is a loop over independent subspaces and was the whole of
# the index build's 27 s. Optional so a toolchain without it still builds.
find_package(OpenMP)
add_library(jhq_cpu STATIC
    cpu/codebook.cpp
    cpu/pq_codebook.cpp
    cpu/jl_transform.cpp
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(jhq_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

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
add_jhq_version(jhq_v14_streaming_add)
add_jhq_version(jhq_v15_eval_fix)
add_jhq_version(jhq_v16_pq_primary)
add_jhq_version(jhq_v17_parallel_scan)
add_jhq_version(jhq_v18_shared_lut)
add_jhq_version(jhq_v19_tiled_scan)
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

# v34: W=4 batched expansion + HNSW early termination, beam=min(ef,128), expanded-block output
add_library(jhq_hblock_v34 STATIC
    hblock_v34/search.cu
    hblock_v34/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v34 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v34 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v34 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v34 examples/demo_hblock_v34.cu)
target_link_libraries(demo_hblock_v34 PRIVATE jhq_hblock_v34)

# v35: ceiling test — beam=ef (no cap), no early termination, expanded-block output
add_library(jhq_hblock_v35 STATIC
    hblock_v35/search.cu
    hblock_v35/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v35 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v35 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v35 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v35 examples/demo_hblock_v35.cu)
target_link_libraries(demo_hblock_v35 PRIVATE jhq_hblock_v35)

# v36: v35 + replace CPU stable_sort → GPU CUB radix sort during build
add_library(jhq_hblock_v36 STATIC
    hblock_v36/search.cu
    hblock_v36/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v36 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v36 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v36 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v36 examples/demo_hblock_v36.cu)
target_link_libraries(demo_hblock_v36 PRIVATE jhq_hblock_v36)

# v37_prr: v36 + certified PQ racing (PRR) — corrected PQ + L2/U2 epsilon bounds + pruned exact rerank
add_library(jhq_hblock_v37_prr STATIC
    hblock_v37_prr/search.cu
    hblock_v37_prr/diag.cu
    hblock_v37_prr/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v37_prr PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v37_prr PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v37_prr PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v37_prr examples/demo_hblock_v37_prr.cu)
target_link_libraries(demo_hblock_v37_prr PRIVATE jhq_hblock_v37_prr)

# v37_prr seed diagnostic: exact-seed threshold go/no-go (additive; see
# HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md). Same library, separate demo.
add_executable(demo_hblock_v37_prr_seed_diag examples/demo_hblock_v37_prr_seed_diag.cu)
target_link_libraries(demo_hblock_v37_prr_seed_diag PRIVATE jhq_hblock_v37_prr)

# v38: plain SPACEV int8, fully GPU-resident, no region partitioning. Exists
# as the controlled comparison baseline for v39/v42/v43's region work --
# same data/encoding/routing, no region_id -> pool-slot indirection.
add_library(jhq_hblock_v38 STATIC
    hblock_v38/search.cu
    hblock_v38/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v38 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v38 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v38 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v38 examples/demo_hblock_v38.cu)
target_link_libraries(demo_hblock_v38 PRIVATE jhq_hblock_v38)

# v36_1: hblock_v38 with add()'s balanced-kmeans reclustering GPU-
# accelerated (cuBLAS GEMM + existing kmeans kernels) instead of pure
# single-threaded CPU -- see hblock_v36_1/jhq_gpu_index.cuh class comment.
add_library(jhq_hblock_v36_1 STATIC
    hblock_v36_1/search.cu
    hblock_v36_1/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v36_1 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v36_1 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v36_1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v36_1 examples/demo_hblock_v36_1.cu)
target_link_libraries(demo_hblock_v36_1 PRIVATE jhq_hblock_v36_1)

# v36_2: straight fork of v36_1, no algorithm change -- diagnose_missed_gt()
# (routing/graph miss classification: A=routing miss, B=graph unreachable,
# C=depth miss) ported in from hblock_v36 and wired into the demo, purely so
# it can be run on a clean, dedicated version to get a real A/B/C breakdown
# before deciding which recall lever (routing breadth, graph connectivity,
# beam cap, or a structural block-graph redesign) to pull next.
add_library(jhq_hblock_v36_2 STATIC
    hblock_v36_2/search.cu
    hblock_v36_2/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v36_2 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v36_2 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v36_2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v36_2 examples/demo_hblock_v36_2.cu)
target_link_libraries(demo_hblock_v36_2 PRIVATE jhq_hblock_v36_2)

# v39: SPACEV int8 streamed add() + logical region partitioning and a
# bounded GPU region pool (leaf kernel resolves PQ codes/ids and raw
# vectors through region_id -> pool-slot indirection instead of flat
# resident arrays; fetch planning stages only the regions a batch touches).
add_library(jhq_hblock_v39 STATIC
    hblock_v39/search.cu
    hblock_v39/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v39 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v39 PUBLIC
    jhq_hblock_v27   # for gpu_build_block_adj_v27 and encode kernels
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v39 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v39 examples/demo_hblock_v39.cu)
target_link_libraries(demo_hblock_v39 PRIVATE jhq_hblock_v39)

# Same jhq_hblock_v39 library, SIFT100M .bvecs entry point (add(BVecsReader&,
# int) overload) -- for direct comparison against hblock_v40 on the same data.
add_executable(demo_hblock_v39_sift100m examples/demo_hblock_v39_sift100m.cu)
target_link_libraries(demo_hblock_v39_sift100m PRIVATE jhq_hblock_v39)

# v42: v39's region pool, backed by real mmap'd region files on disk
# instead of an in-process host RAM buffer (see hblock_v42/jhq_gpu_index.cuh
# class comment). Kernel-level logic is unchanged from v39.
add_library(jhq_hblock_v42 STATIC
    hblock_v42/search.cu
    hblock_v42/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v42 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v42 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v42 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v42 examples/demo_hblock_v42.cu)
target_link_libraries(demo_hblock_v42 PRIVATE jhq_hblock_v42)

# v43: v42 + graph-aware region packing (BFS over the block adjacency graph
# instead of physical/cell-lexicographic block order) -- see
# hblock_v43/jhq_gpu_index.cuh class comment.
add_library(jhq_hblock_v43 STATIC
    hblock_v43/search.cu
    hblock_v43/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v43 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v43 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v43 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v43 examples/demo_hblock_v43.cu)
target_link_libraries(demo_hblock_v43 PRIVATE jhq_hblock_v43)

# v40: SIFT bvecs baseline with split PQ selection and exact int8 rerank.
# The explicit candidate-ID hand-off is the boundary used by later region I/O.
add_library(jhq_hblock_v40 STATIC
    hblock_v40/search.cu
    hblock_v40/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v40 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v40 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v40 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v40 examples/demo_hblock_v40.cu)
target_link_libraries(demo_hblock_v40 PRIVATE jhq_hblock_v40)

# v41: v38 fused leaf semantics with complete block records in host-resident
# logical regions and a bounded GPU staging pool.
add_library(jhq_hblock_v41 STATIC
    hblock_v41/search.cu
    hblock_v41/jhq_gpu_index.cu
)
target_include_directories(jhq_hblock_v41 PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_hblock_v41 PUBLIC
    jhq_hblock_v27
    jhq_cpu
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(jhq_hblock_v41 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>)
add_executable(demo_hblock_v41_sift100m examples/demo_hblock_v41_sift100m.cu)
target_link_libraries(demo_hblock_v41_sift100m PRIVATE jhq_hblock_v41)
add_executable(demo_hblock_v41_spacev examples/demo_hblock_v41_spacev.cu)
target_link_libraries(demo_hblock_v41_spacev PRIVATE jhq_hblock_v41)

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

# ── Step-timed build of v18 ───────────────────────────────────────────────────
# ncu cannot read performance counters in this container (ERR_NVGPUCTRPERM), so
# stage attribution comes from CUDA events compiled into the search path. The
# timed build skips the graph capture, since events cannot be read back from
# inside one.
add_library(jhq_v18_timed STATIC
    jhq_v18_shared_lut/encode.cu
    jhq_v18_shared_lut/search.cu
    jhq_v18_shared_lut/jhq_gpu_index.cu
)
target_include_directories(jhq_v18_timed PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v18_timed PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v18_timed PRIVATE JHQ_STEP_TIMING=1)
target_compile_options(jhq_v18_timed PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v18_timed examples/demo_jhq_v18_shared_lut.cu)
target_link_libraries(demo_jhq_v18_timed PRIVATE jhq_v18_timed)

# ── In-scan top-k ablation (timed) ────────────────────────────────────────────
# Prices the ck-round block reduction inside scan_ivf. Produces wrong results
# by design; it exists only to measure what that selection costs.
add_library(jhq_v18_notopk STATIC
    jhq_v18_shared_lut/encode.cu
    jhq_v18_shared_lut/search.cu
    jhq_v18_shared_lut/jhq_gpu_index.cu
)
target_include_directories(jhq_v18_notopk PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v18_notopk PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v18_notopk PRIVATE JHQ_STEP_TIMING=1 JHQ_ABLATE_INSCAN_TOPK=1)
target_compile_options(jhq_v18_notopk PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v18_notopk examples/demo_jhq_v18_shared_lut.cu)
target_link_libraries(demo_jhq_v18_notopk PRIVATE jhq_v18_notopk)

# ── Inner-loop ablations (timed, results invalid by design) ───────────────────
add_library(jhq_v18_ab_lut STATIC
    jhq_v18_shared_lut/encode.cu
    jhq_v18_shared_lut/search.cu
    jhq_v18_shared_lut/jhq_gpu_index.cu
)
target_include_directories(jhq_v18_ab_lut PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v18_ab_lut PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v18_ab_lut PRIVATE JHQ_STEP_TIMING=1 JHQ_ABLATE_LUT=1)
target_compile_options(jhq_v18_ab_lut PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v18_ab_lut examples/demo_jhq_v18_shared_lut.cu)
target_link_libraries(demo_jhq_v18_ab_lut PRIVATE jhq_v18_ab_lut)

add_library(jhq_v18_ab_acc STATIC
    jhq_v18_shared_lut/encode.cu
    jhq_v18_shared_lut/search.cu
    jhq_v18_shared_lut/jhq_gpu_index.cu
)
target_include_directories(jhq_v18_ab_acc PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v18_ab_acc PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v18_ab_acc PRIVATE JHQ_STEP_TIMING=1 JHQ_ABLATE_ACC=1)
target_compile_options(jhq_v18_ab_acc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v18_ab_acc examples/demo_jhq_v18_shared_lut.cu)
target_link_libraries(demo_jhq_v18_ab_acc PRIVATE jhq_v18_ab_acc)

add_library(jhq_v18_ab_insert STATIC
    jhq_v18_shared_lut/encode.cu
    jhq_v18_shared_lut/search.cu
    jhq_v18_shared_lut/jhq_gpu_index.cu
)
target_include_directories(jhq_v18_ab_insert PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v18_ab_insert PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v18_ab_insert PRIVATE JHQ_STEP_TIMING=1 JHQ_ABLATE_INSERT=1)
target_compile_options(jhq_v18_ab_insert PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v18_ab_insert examples/demo_jhq_v18_shared_lut.cu)
target_link_libraries(demo_jhq_v18_ab_insert PRIVATE jhq_v18_ab_insert)

add_library(jhq_v18_ab_f32lut STATIC
    jhq_v18_shared_lut/encode.cu
    jhq_v18_shared_lut/search.cu
    jhq_v18_shared_lut/jhq_gpu_index.cu
)
target_include_directories(jhq_v18_ab_f32lut PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v18_ab_f32lut PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v18_ab_f32lut PRIVATE JHQ_STEP_TIMING=1 JHQ_SMEM_LUT_FLOAT=1)
target_compile_options(jhq_v18_ab_f32lut PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v18_ab_f32lut examples/demo_jhq_v18_shared_lut.cu)
target_link_libraries(demo_jhq_v18_ab_f32lut PRIVATE jhq_v18_ab_f32lut)

add_library(jhq_v19_timed STATIC
    jhq_v19_tiled_scan/encode.cu
    jhq_v19_tiled_scan/search.cu
    jhq_v19_tiled_scan/jhq_gpu_index.cu
)
target_include_directories(jhq_v19_timed PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v19_timed PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v19_timed PRIVATE JHQ_STEP_TIMING=1)
target_compile_options(jhq_v19_timed PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v19_timed examples/demo_jhq_v19_tiled_scan.cu)
target_link_libraries(demo_jhq_v19_timed PRIVATE jhq_v19_timed)

# ── v20: tile-shape sweep ────────────────────────────────────────────────────
# v19 fixed TILE_C=4, TILE_M=8 and took scan_ivf 16.56 -> 11.48 ms at nprobe=8.
# TILE_C is the number of independent accumulate chains a thread runs, TILE_M
# how many subspaces it sweeps before moving to the next candidate. Which of
# the two bought the 30% was never separated, so build the grid and measure.
function(add_jhq_tile_variant TC TM)
    set(tgt jhq_v20_c${TC}m${TM})
    add_library(${tgt} STATIC
        jhq_v19_tiled_scan/encode.cu
        jhq_v19_tiled_scan/search.cu
        jhq_v19_tiled_scan/jhq_gpu_index.cu
    )
    target_include_directories(${tgt} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(${tgt} PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
    target_compile_definitions(${tgt} PRIVATE
        JHQ_STEP_TIMING=1 JHQ_TILE_C=${TC} JHQ_TILE_M=${TM})
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    )
    add_executable(demo_${tgt} examples/demo_jhq_v19_tiled_scan.cu)
    target_link_libraries(demo_${tgt} PRIVATE ${tgt})
endfunction()

foreach(TC 1 2 4 8 16)
    foreach(TM 4 8 16 32 96)
        add_jhq_tile_variant(${TC} ${TM})
    endforeach()
endforeach()

# ── v21: cascaded scan ───────────────────────────────────────────────────────
# Pass 1 ranks every candidate on a NUM/DEN prefix of the subspaces, pass 2
# completes only the best KEEP per thread. NUM==DEN is v19 with extra copies,
# which is the control the sweep needs.
function(add_jhq_cascade_variant NUM DEN KEEP)
    set(tgt jhq_v21_k${KEEP})
    add_library(${tgt} STATIC
        jhq_v21_cascade/encode.cu
        jhq_v21_cascade/search.cu
        jhq_v21_cascade/jhq_gpu_index.cu
    )
    target_include_directories(${tgt} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(${tgt} PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
    target_compile_definitions(${tgt} PRIVATE
        JHQ_STEP_TIMING=1
        JHQ_PREFIX_NUM=${NUM} JHQ_PREFIX_DEN=${DEN} JHQ_PREFIX_KEEP=${KEEP})
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    )
    add_executable(demo_${tgt} examples/demo_jhq_v21_cascade.cu)
    target_link_libraries(demo_${tgt} PRIVATE ${tgt})
endfunction()

foreach(KEEP 4 8 16 32)
    add_jhq_cascade_variant(1 4 ${KEEP})
endforeach()

# Same kernel with the step timers off. JHQ_STEP_TIMING inserts a cudaEvent
# between every kernel and so suppresses CUDA Graph capture: at batch 1024 the
# timed v20 build spent 45.7 ms outside the scan against 18.8 ms at batch 256,
# and lost throughput (10197 QPS) even though its scan got faster (75.5 -> 52.4
# ms). v16 is untimed and doubled over the same change, 10550 -> 20154. Any
# number compared against v16 has to come from a build in the same mode.
function(add_jhq_cascade_fast KEEP)
    set(tgt jhq_v21_fast_k${KEEP})
    add_library(${tgt} STATIC
        jhq_v21_cascade/encode.cu
        jhq_v21_cascade/search.cu
        jhq_v21_cascade/jhq_gpu_index.cu
    )
    target_include_directories(${tgt} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(${tgt} PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
    target_compile_definitions(${tgt} PRIVATE
        JHQ_PREFIX_NUM=1 JHQ_PREFIX_DEN=4 JHQ_PREFIX_KEEP=${KEEP})
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    )
    add_executable(demo_${tgt} examples/demo_jhq_v21_cascade.cu)
    target_link_libraries(demo_${tgt} PRIVATE ${tgt})
endfunction()

foreach(KEEP 4 8 16 32)
    add_jhq_cascade_fast(${KEEP})
endforeach()

# In-scan top-ck ablation at the matched point. ck = ceil(alpha*k), so alpha=100
# with k=10 makes it 1000, and the selection is 1000 sequential passes of a
# whole-block min-reduction: 10 barriers each at BLOCK=1024, so 10000 barriers
# per block, none of which scale with nprobe. The earlier ablation that found
# this free was run at alpha=4 (ck=40) and BLOCK=256, 25x fewer passes, so it
# says nothing about this regime.
add_library(jhq_v21_noselect STATIC
    jhq_v21_cascade/encode.cu
    jhq_v21_cascade/search.cu
    jhq_v21_cascade/jhq_gpu_index.cu
)
target_include_directories(jhq_v21_noselect PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v21_noselect PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v21_noselect PRIVATE
    JHQ_PREFIX_NUM=1 JHQ_PREFIX_DEN=4 JHQ_PREFIX_KEEP=16 JHQ_ABLATE_INSCAN_TOPK=1)
target_compile_options(jhq_v21_noselect PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v21_noselect examples/demo_jhq_v21_cascade.cu)
target_link_libraries(demo_jhq_v21_noselect PRIVATE jhq_v21_noselect)

# Same kernel with the ck-pass selection replaced by one bitonic sort.
function(add_jhq_bitonic KEEP)
    set(tgt jhq_v21_bito_k${KEEP})
    add_library(${tgt} STATIC
        jhq_v21_cascade/encode.cu
        jhq_v21_cascade/search.cu
        jhq_v21_cascade/jhq_gpu_index.cu
    )
    target_include_directories(${tgt} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(${tgt} PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
    target_compile_definitions(${tgt} PRIVATE
        JHQ_PREFIX_NUM=1 JHQ_PREFIX_DEN=4 JHQ_PREFIX_KEEP=${KEEP} JHQ_BITONIC_SELECT=1)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    )
    add_executable(demo_${tgt} examples/demo_jhq_v21_cascade.cu)
    target_link_libraries(demo_${tgt} PRIVATE ${tgt})
endfunction()

foreach(KEEP 8 16 32)
    add_jhq_bitonic(${KEEP})
endforeach()

# Strided candidate assignment, with and without the bitonic selection. TILE_C
# chains without TILE_C's recall cost: the v20 grid lost 0.7328 -> 0.6933 going
# to TILE_C=16 because contiguous tiles concentrate adjacent candidates, and
# striding by BLOCK undoes exactly that. KEEP stays at 8; 16 needs more than
# the 64 registers a thread gets when 1024 are resident.
function(add_jhq_strided TC BITO)
    set(tgt jhq_v22_s${TC}b${BITO})
    add_library(${tgt} STATIC
        jhq_v21_cascade/encode.cu
        jhq_v21_cascade/search.cu
        jhq_v21_cascade/jhq_gpu_index.cu
    )
    target_include_directories(${tgt} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(${tgt} PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
    target_compile_definitions(${tgt} PRIVATE
        JHQ_PREFIX_NUM=1 JHQ_PREFIX_DEN=4 JHQ_PREFIX_KEEP=8
        JHQ_TILE_C=${TC} JHQ_TILE_STRIDED=1 JHQ_BITONIC_SELECT=${BITO})
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    )
    add_executable(demo_${tgt} examples/demo_jhq_v21_cascade.cu)
    target_link_libraries(demo_${tgt} PRIVATE ${tgt})
endfunction()

foreach(TC 2 4 8)
    add_jhq_strided(${TC} 0)
    add_jhq_strided(${TC} 1)
endforeach()

# ── Residual-codebook layout ablation ────────────────────────────────────────
# Same sources as jhq_v16_pq_primary with the residual codebook forced global
# (v15's layout) instead of per-subspace (the official layout). Only the
# codebook contents differ -- see JHQ_GLOBAL_RESIDUAL_CB in jhq_gpu_index.cu.
add_library(jhq_v16_globalres STATIC
    jhq_v16_pq_primary/encode.cu
    jhq_v16_pq_primary/search.cu
    jhq_v16_pq_primary/jhq_gpu_index.cu
)
target_include_directories(jhq_v16_globalres PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(jhq_v16_globalres PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
target_compile_definitions(jhq_v16_globalres PRIVATE JHQ_GLOBAL_RESIDUAL_CB=1)
target_compile_options(jhq_v16_globalres PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
)
add_executable(demo_jhq_v16_globalres examples/demo_jhq_v16_pq_primary.cu)
target_link_libraries(demo_jhq_v16_globalres PRIVATE jhq_v16_globalres)

# ── K_LOCAL ablation variants ────────────────────────────────────────────────
# Same sources as jhq_v15_eval_fix; only JHQ_K_LOCAL differs.
# scan_ivf_coalesced_kernel keeps K_LOCAL candidates per thread and drops the
# rest before the block-wide selection, so this measures what that truncation
# costs in recall. 32 would need the shared-memory opt-in (66KB > 48KB default)
# -- search.cu checks and throws rather than failing silently.
foreach(KL 8 16)
    add_library(jhq_v15_kl${KL} STATIC
        jhq_v15_eval_fix/encode.cu
        jhq_v15_eval_fix/search.cu
        jhq_v15_eval_fix/jhq_gpu_index.cu
    )
    target_include_directories(jhq_v15_kl${KL} PUBLIC ${CMAKE_SOURCE_DIR})
    target_link_libraries(jhq_v15_kl${KL} PUBLIC jhq_cpu CUDA::cublas CUDA::cudart)
    target_compile_definitions(jhq_v15_kl${KL} PRIVATE JHQ_K_LOCAL=${KL})
    target_compile_options(jhq_v15_kl${KL} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    )
    add_executable(demo_jhq_v15_kl${KL} examples/demo_jhq_v15_eval_fix.cu)
    target_link_libraries(demo_jhq_v15_kl${KL} PRIVATE jhq_v15_kl${KL})
endforeach()

# ── Tests ─────────────────────────────────────────────────────────────────────
# Pure C++ (common/recall.cuh has no CUDA in it), so no CUDA toolchain needed
# to run these.  ctest, or: ./build/test_recall
enable_testing()
add_executable(test_recall tests/test_recall.cpp)
target_include_directories(test_recall PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(test_recall PRIVATE -O2)
add_test(NAME recall COMMAND test_recall)
