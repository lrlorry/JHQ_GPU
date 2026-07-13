#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v27 {

static constexpr int K_MAX = 256;

// v27 workspace — beam_size template + bitonic warp sort in leaf_flat (top-32 per block).
struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;   // == graph_budget
    int rerank_r     = 0;
    int d_proj       = 0;
    int ck1 = 0, ck2 = 0, ck3 = 0;
    int beam_size    = 32;  // beam slots: 32/64/128 via template<SPT> kernel
    int block_topk   = 32;  // per-block output candidates (warp-aligned, default=32)

    // ── Pinned host ──────────────────────────────────────────────────────────
    float* h_q_pinned    = nullptr;
    int*   h_leaf_cnt    = nullptr;
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    int*   h_top1_ids  = nullptr;   // [B, ck1]
    int*   h_top2_beam = nullptr;   // [B*ck1, ck2]
    int*   h_top3_beam = nullptr;   // [B*ck1*ck2, ck3]
    int*   h_block_sel  = nullptr;   // [B, graph_budget]  (unused after GPU beam)
    int*   h_entry_cnt  = nullptr;   // [B] entry count stats

    // ── GPU block beam search buffers ────────────────────────────────────────
    int*   d_visited    = nullptr;   // [B × bitmap_words]  visited bitmap
    int    bitmap_words = 0;         // ceil(n_blks / 32)

    // ── GPU routing (L1+L2+L3 beam) ─────────────────────────────────────────
    float* d_q_batch   = nullptr;
    float* d_q_proj1   = nullptr;
    float* d_dots1     = nullptr;
    int*   d_top1_ids  = nullptr;
    float* d_r1_beam   = nullptr;
    int*   d_top2_beam = nullptr;
    int*   d_top3_beam = nullptr;
    float* d_q_r3      = nullptr;
    float* d_lut_fine  = nullptr;

    // ── Block selection (H2D from CPU) ───────────────────────────────────────
    int*   d_leaf_sel  = nullptr;
    int*   d_leaf_cnt  = nullptr;

    // ── Pair build + sort ────────────────────────────────────────────────────
    int*   d_query_offsets = nullptr;
    int*   d_pair_leaf_a   = nullptr;
    int*   d_pair_qid_a    = nullptr;
    int*   d_pair_leaf_b   = nullptr;
    int*   d_pair_qid_b    = nullptr;

    float* d_out_dists = nullptr;
    int*   d_out_ids   = nullptr;
    float* d_pq_dists  = nullptr;
    int*   d_pq_ids    = nullptr;
    float* d_cand_vecs   = nullptr;
    float* d_exact_dists = nullptr;
    float* d_final_dists = nullptr;
    int*   d_final_ids   = nullptr;

    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;

    cudaStream_t stream = nullptr;
};

// GPU block adjacency build (called during add())
void gpu_build_block_adj_v27(
    const float* d_blk_proj, const float* d_blk_norm,
    const int* d_csr_start, const int* d_csr_list, const int* d_cell_id,
    int* d_adj,
    int n_blks, int d_proj, int degree, cudaStream_t s);

// Fused entry-selection + beam search (one warp per query)
void gpu_block_search_v27(
    int B, int n_blks, int d_proj,
    int K2, int K3, int ck1, int ck2, int ck3,
    int degree, int budget, int max_ls, int entry_per_cell,
    const int*   d_block_adj,
    const float* d_blk_proj,
    const float* d_blk_norm,
    const int*   d_pair_blk_start,
    const int*   d_pair_blk_count,
    SearchWorkspace& ws);

void route_gpu_v27(
    cublasHandle_t cublas,
    const float* d_Pi1,
    const float* d_Pi2,
    const float* d_Pi3,
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

void gpu_build_and_sort_pairs_v27(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws);

// leaf_flat: bitonic warp sort per warp + k-way merge → top block_topk per block
void launch_leaf_flat_v27(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size, int block_topk,
    cudaStream_t   stream);

// n_per_pair == block_topk (32), global top-rerank_r selection
void gpu_merge_pq_topk_v27(
    int nq, int n_pairs, int rerank_r, int n_per_pair,
    SearchWorkspace& ws);

void launch_gather_vecs_v27(
    const float* d_base_vecs,
    const int*   d_cand_ids,
    float*       d_cand_vecs,
    int B, int R, int d,
    cudaStream_t s);

void launch_exact_rerank_v27(
    const float* d_q_batch,
    const float* d_cand_vecs,
    const int*   d_cand_ids,
    float*       d_exact_dists,
    float*       d_final_dists,
    int*         d_final_ids,
    int B, int R, int d, int k,
    cudaStream_t s);

} // namespace hblock_v27
