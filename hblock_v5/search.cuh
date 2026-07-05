#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v5 {

// ── Search workspace ──────────────────────────────────────────────────────────
struct SearchWorkspace {
    int batch_cap = 0;

    // Query/routing buffers (same as v2)
    float*   h_q_pinned    = nullptr;
    float*   d_q_batch     = nullptr;
    float*   d_q_rot       = nullptr;
    float*   d_q_r1        = nullptr;
    float*   d_q_r2        = nullptr;
    float*   d_dots1       = nullptr;
    float*   d_dots2       = nullptr;
    int*     d_top1_ids    = nullptr;
    int*     d_top2_ids    = nullptr;
    int*     d_leaf_sel    = nullptr;  // [B, ck3]
    int*     d_leaf_cnt    = nullptr;  // [B]
    float*   d_lut_fine    = nullptr;  // [B, d, Kr]
    float*   d_fine_dists  = nullptr;  // [B, ck3 * leaf_size]
    int*     d_fine_ids    = nullptr;  // [B, ck3 * leaf_size]
    float*   d_final_dists = nullptr;  // [B, k]
    int*     d_final_ids   = nullptr;  // [B, k]

    // v5: global sort buffers (replaces per-query bitonic sort)
    int*     d_pair_keys   = nullptr;  // [B*ck3] sort input:  leaf_block_id
    int*     d_pair_vals   = nullptr;  // [B*ck3] sort input:  flat_idx = q*ck3+slot
    int*     d_pair_keys_s = nullptr;  // [B*ck3] sort output: sorted leaf_block_ids
    int*     d_pair_vals_s = nullptr;  // [B*ck3] sort output: sorted flat_idx
    void*    d_cub_tmp     = nullptr;  // CUB temp storage
    size_t   cub_bytes     = 0;

    cudaStream_t stream = nullptr;
};

// ── Main search entry point ────────────────────────────────────────────────────
void search_hblock(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_route1_cents,
    const float*   d_route1_norms,
    const float*   d_route2_cents,
    const float*   d_route2_norms,
    const float*   d_fine_c1d,
    const int*     d_pair_blk_start,
    const int*     d_pair_blk_count,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids,
    const int*     d_leaf_sizes,
    const float*   h_queries,
    int nq, int d, int K1, int K2, int Kr, int Br, int bpv,
    int leaf_size, int ck1, int ck2, int ck3, int k,
    int batch_size, int n_leaf_blocks,
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace hblock_v5
