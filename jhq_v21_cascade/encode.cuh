#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace jhq_gpu {

// Primary level is a product quantizer now: d_cent is [M][K][Ds] and a code
// byte is the index of the nearest centroid in that subspace, not a packed
// tuple of per-dimension scalar indices. Ds is therefore unconstrained --
// see cpu/pq_codebook.h for why that matters.
void launch_primary_encode(
    const float*   d_y,
    uint8_t*       d_codes,
    const float*   d_cent,
    int N, int d, int M, int Ds, int K,
    cudaStream_t stream = 0);

// d_res_c1d is [M][Kr]: one scalar residual codebook per subspace, matching
// the official IndexJHQ::get_scalar_codebook_ptr(subspace_idx, level).
void launch_residual_encode(
    const float*   d_y,
    const uint8_t* d_primary,
    uint8_t*       d_res_codes,
    float*         d_corrections,
    const float*   d_cent,
    const float*   d_res_c1d,
    int N, int d, int M, int Ds, int K, int Kr, int Br, int bpv,
    cudaStream_t stream = 0);

} // namespace jhq_gpu

namespace jhq_gpu {

// Primary encode as a GEMM plus an argmin.
//
// ||y - c||^2 = ||y||^2 - 2 y.c + ||c||^2, and ||y||^2 is the same for every
// centroid, so the assignment only needs ||c||^2 - 2 y.c. The dot products for
// all K centroids of a subspace are a matrix product, which cuBLAS runs far
// closer to the machine's peak than a hand-written loop can: the loop version
// held 32 floats of the vector in registers per thread and lost the occupancy
// that would have hidden its shared-memory traffic, reaching 3.6 TFLOP/s of a
// far larger budget.
//
// d_cent_sqnorm is [M][K], ||c||^2 precomputed once per index.
void launch_primary_encode_gemm(cublasHandle_t cublas,
                                const float* d_y, uint8_t* d_codes,
                                const float* d_cent, const float* d_cent_sqnorm,
                                float* d_dots_scratch, int dots_capacity_rows,
                                int N, int d, int M, int Ds, int K,
                                cudaStream_t stream);

// ||c||^2 for every centroid, [M][K].
void launch_centroid_sqnorms(const float* d_cent, float* d_out,
                             int M, int K, int Ds, cudaStream_t stream = 0);

} // namespace jhq_gpu
