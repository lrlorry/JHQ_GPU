#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v7 {

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
    int*     d_leaf_sel    = nullptr;
    int*     d_leaf_cnt    = nullptr;
    float*   d_lut_fine    = nullptr;
    float*   d_fine_dists  = nullptr;
    int*     d_fine_ids    = nullptr;
    float*   d_final_dists = nullptr;
    int*     d_final_ids   = nullptr;

    // v5-style sort buffers
    int*     d_pair_keys   = nullptr;
    int*     d_pair_vals   = nullptr;
    int*     d_pair_keys_s = nullptr;
    int*     d_pair_vals_s = nullptr;
    void*    d_cub_tmp     = nullptr;
    size_t   cub_bytes     = 0;

    // v7: RLE buffers for transposed leaf kernel
    int*     d_unique_leaf_ids = nullptr;  // [B*ck3] unique leaf_ids after RLE
    int*     d_run_counts      = nullptr;  // [B*ck3] # (q,slot) pairs per unique leaf
    int*     d_run_starts      = nullptr;  // [B*ck3] exclusive prefix sum of run_counts
    int*     d_num_unique      = nullptr;  // [1] number of unique leaves (device)
    void*    d_cub_tmp2        = nullptr;  // CUB temp for RLE + scan
    size_t   cub_bytes2        = 0;

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

} // namespace hblock_v7
