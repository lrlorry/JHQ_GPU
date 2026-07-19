#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v37_prr {

static constexpr int K_MAX = 256;

// Search workspace — identical layout to v36.
struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;
    int d_proj       = 0;
    int ck1 = 0, ck2 = 0, ck3 = 0;
    int beam_size    = 32;
    int per_block_r  = 16;
    int klocal       = 10;

    // ── Pinned host ──────────────────────────────────────────────────────────
    float* h_q_pinned    = nullptr;
    int*   h_leaf_cnt    = nullptr;
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    int*   h_top1_ids  = nullptr;
    int*   h_top2_beam = nullptr;
    int*   h_top3_beam = nullptr;
    int*   h_block_sel = nullptr;

    // ── GPU block beam search buffers ────────────────────────────────────────
    int*   d_visited    = nullptr;
    int    bitmap_words = 0;

    // ── GPU routing ──────────────────────────────────────────────────────────
    float* d_q_batch   = nullptr;
    float* d_q_proj1   = nullptr;
    float* d_dots1     = nullptr;
    int*   d_top1_ids  = nullptr;
    float* d_r1_beam   = nullptr;
    int*   d_top2_beam = nullptr;
    int*   d_top3_beam = nullptr;
    float* d_q_r3      = nullptr;
    float* d_lut_fine  = nullptr;

    // ── Block selection ──────────────────────────────────────────────────────
    int*   d_leaf_sel  = nullptr;
    int*   d_leaf_cnt  = nullptr;

    // ── Pair build + sort ────────────────────────────────────────────────────
    int*   d_query_offsets = nullptr;
    int*   d_pair_leaf_a   = nullptr;
    int*   d_pair_qid_a    = nullptr;
    int*   d_pair_leaf_b   = nullptr;
    int*   d_pair_qid_b    = nullptr;

    // d_out_dists/ids: per-block exact L2 results [max_pairs × klocal]
    float* d_out_dists = nullptr;
    int*   d_out_ids   = nullptr;

    float* d_final_dists = nullptr;
    int*   d_final_ids   = nullptr;

    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;

    cudaStream_t stream = nullptr;
};

// ── Carried over from v36 (same kernels, new namespace) ──────────────────────

void gpu_block_search_v35(
    int B, int n_blks, int d_proj,
    int K2, int K3, int ck1, int ck2, int ck3,
    int degree, int ef, int max_ls, int entry_per_cell,
    const int*   d_block_adj,
    const float* d_blk_proj,
    const float* d_blk_norm,
    const int*   d_pair_blk_start,
    const int*   d_pair_blk_count,
    SearchWorkspace& ws);

void route_gpu_v29(
    cublasHandle_t cublas,
    const float* d_Pi1, const float* d_Pi2, const float* d_Pi3,
    const float* d_route1_cents_proj, const float* d_route1_cents_full,
    const float* d_route1_norms,
    const float* d_route2_cents_proj, const float* d_route2_cents_full,
    const float* d_route2_norms,
    const float* d_route3_cents_proj, const float* d_route3_cents_full,
    const float* d_route3_norms,
    const float* d_fine_c1d,
    const float* h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int K3, int Kr,
    int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws);

void gpu_build_and_sort_pairs_v29(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws);

// Per-block: PQ distances (LUT) → bitonic sort → top per_block_r → exact L2 → top klocal
void launch_leaf_flat_v29(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,
    const float*   d_base_vecs,
    const float*   d_q_batch,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size,
    int per_block_r, int klocal,
    cudaStream_t stream);

// Merge per-block exact results → global top-k per query
void launch_final_merge_v29(
    int nq, int n_pairs, int klocal, int k,
    SearchWorkspace& ws);

// ── v37_prr new kernels ───────────────────────────────────────────────────────

// CORRECTED_FIXED mode: compute PQ dist using block's own cell centroid
void launch_leaf_corrected_pq(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const int*     d_block_cell_id,
    const float*   d_abs_cents,
    const float*   d_fine_c1d,
    const float*   d_base_vecs,
    const float*   d_q_batch,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size,
    int per_block_r, int klocal,
    cudaStream_t stream);

// CERTIFIED_PRR Pass A: compute L2/U2 intervals for all (pair, vector) candidates
void launch_leaf_prr_interval(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_sizes,
    const int*     d_block_cell_id,
    const float*   d_abs_cents,
    const float*   d_fine_c1d,
    const float*   d_q_batch,
    const float*   d_block_eps,
    int            eps_stride,
    float*         d_prr_l2,
    float*         d_prr_u2topk,
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size,
    int klocal,
    cudaStream_t stream);

// CERTIFIED_PRR Pass B: per-query tau2 = k-th smallest U2 across all candidate blocks
void launch_prr_tau2(
    const int*   d_pair_qids,
    const float* d_prr_u2topk,
    float*       d_prr_tau2,
    int n_pairs, int k, int batch_size,
    cudaStream_t stream);

// CERTIFIED_PRR Pass C+D: exact rerank for survivors (L2 <= tau2)
void launch_prr_exact_rerank(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_prr_l2,
    const float*   d_prr_tau2,
    const float*   d_base_vecs,
    const float*   d_q_batch,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_pairs,
    int d, int leaf_size,
    int per_block_r, int klocal,
    cudaStream_t stream);

} // namespace hblock_v37_prr
