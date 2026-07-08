#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v17 {

static constexpr int TOP_P  = 4;
static constexpr int K_MAX  = 128;  // max k for rerank output (must >= rerank_r)

struct SearchWorkspace {
    int batch_cap = 0;
    int max_pairs = 0;
    int rerank_r  = 0;
    int d_proj    = 0;

    // ── Pinned host buffers ───────────────────────────────────────────────────
    float* h_q_pinned    = nullptr;
    int*   h_leaf_cnt    = nullptr;
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    // ── Routing buffers ───────────────────────────────────────────────────────
    float* d_q_batch   = nullptr;   // [B, d]
    float* d_q_proj1   = nullptr;   // [B, d_proj]  L1 projected
    float* d_q_proj2   = nullptr;   // [B, d_proj]  L2 projected (of r1)
    float* d_q_proj3   = nullptr;   // [B, d_proj]  L3 projected (of r2)
    float* d_q_r1      = nullptr;   // [B, d]  r1 = q  - C1_full[c1]
    float* d_q_r2      = nullptr;   // [B, d]  r2 = r1 - C2_full[c2]
    float* d_q_r3      = nullptr;   // [B, d]  r3 = r2 - C3_full[c3]  → for LUT
    float* d_dots1     = nullptr;   // [B, K1]
    float* d_dots2     = nullptr;   // [B, K2]
    float* d_dots3     = nullptr;   // [B, K3]
    int*   d_top1_ids  = nullptr;   // [B, ck1]
    int*   d_top2_ids  = nullptr;   // [B, ck2]
    int*   d_top3_ids  = nullptr;   // [B, ck3]
    int*   d_leaf_sel  = nullptr;   // [B, ck1*ck2*ck3]
    int*   d_leaf_cnt  = nullptr;   // [B]
    float* d_lut_fine  = nullptr;   // [B, d, Kr]

    // ── Pair build + sort (ping-pong) ─────────────────────────────────────────
    int*   d_query_offsets = nullptr;   // [B+1]
    int*   d_pair_leaf_a   = nullptr;   // [max_pairs]
    int*   d_pair_qid_a    = nullptr;
    int*   d_pair_leaf_b   = nullptr;
    int*   d_pair_qid_b    = nullptr;

    // ── Leaf PQ scan output ───────────────────────────────────────────────────
    float* d_out_dists = nullptr;   // [max_pairs, TOP_P]
    int*   d_out_ids   = nullptr;   // [max_pairs, TOP_P]

    // ── PQ merge → top-rerank_r (intermediate) ───────────────────────────────
    float* d_pq_dists  = nullptr;   // [B, K_MAX]  PQ merge output (rerank_r valid)
    int*   d_pq_ids    = nullptr;   // [B, K_MAX]

    // ── Re-rank buffers ───────────────────────────────────────────────────────
    float* d_cand_vecs   = nullptr;  // [B, rerank_r, d]   gathered original vecs
    float* d_exact_dists = nullptr;  // [B, rerank_r]

    // ── Final top-k ───────────────────────────────────────────────────────────
    float* d_final_dists = nullptr;  // [B, K_MAX]  (k entries valid)
    int*   d_final_ids   = nullptr;  // [B, K_MAX]

    // ── CUB temp ─────────────────────────────────────────────────────────────
    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;

    cudaStream_t stream = nullptr;
};

// ── Function declarations ─────────────────────────────────────────────────────

// 3-level JL routing + LUT construction
void route_queries_v17(
    cublasHandle_t cublas,
    const float*   d_Pi1,
    const float*   d_Pi2,
    const float*   d_Pi3,
    const float*   d_route1_cents_proj, const float* d_route1_cents_full, const float* d_route1_norms,
    const float*   d_route2_cents_proj, const float* d_route2_cents_full, const float* d_route2_norms,
    const float*   d_route3_cents_proj, const float* d_route3_cents_full, const float* d_route3_norms,
    const float*   d_fine_c1d,
    const int*     d_pair_blk_start,
    const int*     d_pair_blk_count,
    const float*   h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int K3, int Kr, int Br,
    int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws);

// Sort (leaf, qid) pairs by leaf_id
void gpu_build_and_sort_pairs_v17(
    int nq, int n_pairs, int n_leaf_blocks,
    int ck3, SearchWorkspace& ws);

// Leaf PQ distance kernel (one GPU block per (query,leaf) pair)
void launch_leaf_flat_v17(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size,
    cudaStream_t   stream);

// GPU merge: PQ output → top-R candidates per query
void gpu_merge_pq_topk_v17(
    int nq, int n_pairs, int rerank_r,
    SearchWorkspace& ws);

// Gather original vectors for top-R candidates
void launch_gather_vecs(
    const float*   d_base_vecs,
    const int*     d_cand_ids,
    float*         d_cand_vecs,
    int B, int R, int d,
    cudaStream_t   s);

// Exact inner-product + top-k rerank
void launch_exact_rerank(
    const float*   d_q_batch,
    const float*   d_cand_vecs,
    const int*     d_cand_ids,
    float*         d_exact_dists,
    float*         d_final_dists,
    int*           d_final_ids,
    int B, int R, int d, int k,
    cudaStream_t   s);

// iota + qid sort + segmented top-k (reused from PQ merge)
void gpu_merge_topk_v17(
    int nq, int n_pairs, int k,
    SearchWorkspace& ws);

} // namespace hblock_v17
