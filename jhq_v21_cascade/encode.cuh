#pragma once
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
    // Level 2, quantising what level 1 leaves. Pass Br2 = 0 (and null
    // pointers) to keep the single-level index exactly as it was.
    uint8_t*       d_res2_codes,
    const float*   d_res2_c1d,
    int Kr2, int Br2, int bpv2,
    cudaStream_t stream = 0);

} // namespace jhq_gpu
