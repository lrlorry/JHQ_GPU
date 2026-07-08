#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v18 {

static constexpr int JL_TOP_P = 8;   // candidates kept per leaf block
static constexpr int K_MAX    = 64;  // max rerank_r (smem: 32*64*8 = 16KB)

struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;
    int rerank_r     = 0;
    int d_proj       = 0;

    // ── Pinned host buffers ───────────────────────────────────────────────────
    float* h_q_pinned    = nullptr;
    int*   h_leaf_cnt    = nullptr;
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    // ── L1 routing buffers ────────────────────────────────────────────────────
    float* d_q_batch   = nullptr;   // [B, d]
    float* d_q_proj1   = nullptr;   // [B, d_proj]  reused as leaf JL query projection
    float* d_dots1     = nullptr;   // [B, K1]
    int*   d_top1_ids  = nullptr;   // [B, ck1]

    // ── Hierarchical beam routing buffers ─────────────────────────────────────
    float* d_r1_beam   = nullptr;   // [B*ck1, d]
    int*   d_top2_beam = nullptr;   // [B*ck1, ck2]
    int*   d_top3_beam = nullptr;   // [B*ck1*ck2, ck3]

    // ── Leaf selection ────────────────────────────────────────────────────────
    int*   d_leaf_sel  = nullptr;   // [B, max_leaf_sel]
    int*   d_leaf_cnt  = nullptr;   // [B]

    // ── Pair build + sort (ping-pong) ─────────────────────────────────────────
    int*   d_query_offsets = nullptr;   // [B+1]
    int*   d_pair_leaf_a   = nullptr;   // [max_pairs]
    int*   d_pair_qid_a    = nullptr;
    int*   d_pair_leaf_b   = nullptr;
    int*   d_pair_qid_b    = nullptr;

    // ── Leaf JL scan output ───────────────────────────────────────────────────
    float* d_out_dists = nullptr;   // [max_pairs, JL_TOP_P]
    int*   d_out_ids   = nullptr;   // [max_pairs, JL_TOP_P]

    // ── JL merge → top-K_MAX per query ───────────────────────────────────────
    float* d_jl_dists  = nullptr;   // [B, K_MAX]
    int*   d_jl_ids    = nullptr;   // [B, K_MAX]

    // ── Re-rank buffers ───────────────────────────────────────────────────────
    float* d_cand_vecs   = nullptr;  // [B, rerank_r, d]
    float* d_exact_dists = nullptr;  // [B, rerank_r]

    // ── Final top-k ───────────────────────────────────────────────────────────
    float* d_final_dists = nullptr;  // [B, K_MAX]
    int*   d_final_ids   = nullptr;  // [B, K_MAX]

    // ── CUB temp ─────────────────────────────────────────────────────────────
    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;

    cudaStream_t stream = nullptr;
};

// ── Function declarations ─────────────────────────────────────────────────────

void route_queries_v18(
    cublasHandle_t cublas,
    const float*   d_Pi1,
    const float*   d_Pi2,
    const float*   d_Pi3,
    const float*   d_route1_cents_proj, const float* d_route1_cents_full, const float* d_route1_norms,
    const float*   d_route2_cents_proj, const float* d_route2_cents_full, const float* d_route2_norms,
    const float*   d_route3_cents_proj, const float* d_route3_cents_full, const float* d_route3_norms,
    const int*     d_pair_blk_start,
    const int*     d_pair_blk_count,
    const float*   h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int K3,
    int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws);

void gpu_build_and_sort_pairs_v18(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws);

void launch_leaf_jl_v18(
    const int*   d_pair_leaf_ids,
    const int*   d_pair_qids,
    const float* d_leaf_proj_vecs,
    const float* d_q_proj_all,
    const int*   d_leaf_ids_data,
    const int*   d_leaf_sizes,
    float*       d_out_dists,
    int*         d_out_ids,
    int n_pairs, int d_proj, int leaf_size,
    cudaStream_t stream);

void gpu_merge_jl_topk_v18(
    int nq, int n_pairs, int rerank_r,
    SearchWorkspace& ws);

void launch_gather_vecs_v18(
    const float*   d_base_vecs,
    const int*     d_cand_ids,
    float*         d_cand_vecs,
    int B, int R, int d,
    cudaStream_t   s);

void launch_exact_rerank_v18(
    const float*   d_q_batch,
    const float*   d_cand_vecs,
    const int*     d_cand_ids,
    float*         d_exact_dists,
    float*         d_final_dists,
    int*           d_final_ids,
    int B, int R, int d, int k,
    cudaStream_t   s);

} // namespace hblock_v18
