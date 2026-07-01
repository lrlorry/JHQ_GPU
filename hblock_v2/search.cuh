#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace hblock {

// ── Search workspace (all GPU/pinned buffers for one batch) ───────────────────
struct SearchWorkspace {
    int batch_cap = 0;
    int K1_cap = 0, K2_cap = 0;
    int ck1_cap = 0, ck2_cap = 0, ck3_cap = 0;
    int k_cap = 0;

    float*   h_q_pinned    = nullptr;  // [B, d] pinned
    float*   d_q_batch     = nullptr;  // [B, d]
    float*   d_q_rot       = nullptr;  // [B, d]
    float*   d_q_r1        = nullptr;  // [B, d]  q_rot - c1[best L1]
    float*   d_q_r2        = nullptr;  // [B, d]  q_r1  - c2[best L2]
    float*   d_dots1       = nullptr;  // [B, K1]
    float*   d_dots2       = nullptr;  // [B, K2]
    int*     d_top1_ids    = nullptr;  // [B, ck1]
    int*     d_top2_ids    = nullptr;  // [B, ck2]
    int*     d_leaf_sel    = nullptr;  // [B, ck3] selected leaf block indices
    int*     d_leaf_cnt    = nullptr;  // [B]      number of valid entries in leaf_sel
    float*   d_lut_fine    = nullptr;  // [B, d, Kr]
    float*   d_fine_dists  = nullptr;  // [B, ck3 * leaf_size]
    int*     d_fine_ids    = nullptr;  // [B, ck3 * leaf_size]
    float*   d_final_dists = nullptr;  // [B, k]
    int*     d_final_ids   = nullptr;  // [B, k]

    cudaStream_t stream = nullptr;
};

// ── Main search function ──────────────────────────────────────────────────────
void search_hblock(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_route1_cents,   // [K1, d]
    const float*   d_route1_norms,   // [K1]
    const float*   d_route2_cents,   // [K2, d]
    const float*   d_route2_norms,   // [K2]
    const float*   d_fine_c1d,       // [Kr]
    const int*     d_pair_blk_start, // [K1 * K2]
    const int*     d_pair_blk_count, // [K1 * K2]
    const uint8_t* d_leaf_codes,     // [n_blocks, leaf_size, bpv]
    const int*     d_leaf_ids,       // [n_blocks, leaf_size]
    const int*     d_leaf_sizes,     // [n_blocks]
    const float*   h_queries,
    int nq, int d, int K1, int K2, int Kr, int Br, int bpv,
    int leaf_size, int ck1, int ck2, int ck3, int k,
    int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace hblock
