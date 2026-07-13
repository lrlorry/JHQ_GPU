#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v30 {

static constexpr int K_MAX = 256;

// v29: per-block exact rerank (PQ filter → exact L2 within block → merge)
// Eliminates cross-block PQ comparison error; no global gather/rerank step.
struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;
    int d_proj       = 0;
    int ck1 = 0, ck2 = 0, ck3 = 0;
    int beam_size    = 32;
    int per_block_r  = 16;  // PQ prefilter per block before exact L2
    int klocal       = 10;  // per-block exact top-k output

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

// gpu_build_block_adj_v27 is in hblock_v27 namespace (linked via jhq_hblock_v27).
// Call it as hblock_v27::gpu_build_block_adj_v27 after including hblock_v27/search.cuh.

void gpu_block_search_v27(
    int B, int n_blks, int d_proj,
    int K2, int K3, int ck1, int ck2, int ck3,
    int degree, int depth, int max_ls, int entry_per_cell,
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

// Per-block: PQ distances → bitonic sort → top per_block_r → exact L2 → top klocal
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

} // namespace hblock_v30
