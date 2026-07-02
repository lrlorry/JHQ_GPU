#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace hblock_v3 {

// ── Search workspace ──────────────────────────────────────────────────────────
struct SearchWorkspace {
    int batch_cap = 0;
    int K1_cap = 0, K2_cap = 0;
    int ck1_cap = 0, ck2_cap = 0, ck3_cap = 0;
    int k_cap = 0;

    float*   h_q_pinned    = nullptr;  // [B, d] pinned
    float*   d_q_batch     = nullptr;  // [B, d]
    float*   d_z1          = nullptr;  // [B, k1]  P1 @ q
    float*   d_z2          = nullptr;  // [B, k2]  P2 @ r1
    float*   d_q_r1        = nullptr;  // [B, d]   q - C1_full[c1]
    float*   d_q_r2        = nullptr;  // [B, d]   r1 - C2_full[c2]
    float*   d_dots1       = nullptr;  // [B, K1]
    float*   d_dots2       = nullptr;  // [B, K2]
    int*     d_top1_ids    = nullptr;  // [B, ck1]
    int*     d_top2_ids    = nullptr;  // [B, ck2]
    int*     d_leaf_sel    = nullptr;  // [B, ck3]
    int*     d_leaf_cnt    = nullptr;  // [B]
    float*   d_lut_fine    = nullptr;  // [B, d, Kr]
    float*   d_fine_dists  = nullptr;  // [B, ck3 * leaf_size]
    int*     d_fine_ids    = nullptr;  // [B, ck3 * leaf_size]
    float*   d_final_dists = nullptr;  // [B, k]
    int*     d_final_ids   = nullptr;  // [B, k]

    cudaStream_t stream = nullptr;
};

// ── Main search function ──────────────────────────────────────────────────────
// P1 [k1*d], P2 [k2*d]: stored as k1/k2 rows of d elements (row-major).
//   CUBLAS sees them as d×k1 / d×k2 column-major, use OP_T.
// C1_proj [K1*k1], C2_proj [K2*k2]: stored as K centroids × k elements (row-major).
//   CUBLAS sees them as k×K column-major, use OP_T.
// C1_full [K1*d], C2_full [K2*d]: full centroids for residual subtraction.
void search_hblock(
    cublasHandle_t cublas,
    const float*   d_P1,              // [k1, d]
    const float*   d_P2,              // [k2, d]
    const float*   d_C1_proj,         // [K1, k1]
    const float*   d_C1_proj_norms,   // [K1]
    const float*   d_C1_full,         // [K1, d]
    const float*   d_C2_proj,         // [K2, k2]
    const float*   d_C2_proj_norms,   // [K2]
    const float*   d_C2_full,         // [K2, d]
    const float*   d_fine_c1d,        // [Kr]
    const int*     d_pair_blk_start,
    const int*     d_pair_blk_count,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids,
    const int*     d_leaf_sizes,
    const float*   h_queries,
    int nq, int d, int K1, int K2, int k1, int k2,
    int Kr, int Br, int bpv,
    int leaf_size, int ck1, int ck2, int ck3, int k,
    int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace hblock_v3
