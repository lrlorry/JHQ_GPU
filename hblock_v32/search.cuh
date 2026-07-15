#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v32 {

static constexpr int K_MAX = 256;

// v32: single beam parameter.
// Routing uses global top-ef at each level (not per-parent local topk).
// Block graph uses same ef as itopk.
// d_top3_cells[B*ef*K3] is the exhaustive L3 expansion (all K3 children per selected L2 cell).
struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;
    int d_proj       = 0;
    int ef           = 0;   // single beam width — routing + graph search
    int per_block_r  = 16;
    int klocal       = 10;

    float* h_q_pinned    = nullptr;
    int*   h_leaf_cnt    = nullptr;
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    // GPU routing
    float* d_q_batch       = nullptr;
    float* d_q_proj1       = nullptr;
    float* d_dots1         = nullptr;
    int*   d_top1_ids      = nullptr;  // [B*ef] global c1
    float* d_r1_beam       = nullptr;  // [B*ef*d] L1 residuals
    float* d_dist2_all     = nullptr;  // [B*ef*K2] all L2 distances (temp)
    int*   d_top2_localidx = nullptr;  // [B*ef] l1slot*K2 + c2local
    float* d_dist3_all     = nullptr;  // [B*ef*K3] all L3 distances (temp)
    int*   d_top3_localidx = nullptr;  // [B*1] top-1 L3 localidx per query (for PQ LUT)
    int*   d_pq_best_cell  = nullptr;  // [B] global c123 of nearest L3 cell (for PQ LUT)
    int*   d_top3_cells    = nullptr;  // [B*ef*K3] all expanded L3 cells for block entry
    float* d_q_r3          = nullptr;
    float* d_lut_fine      = nullptr;

    int*   d_visited    = nullptr;
    int    bitmap_words = 0;

    int*   d_leaf_sel  = nullptr;
    int*   d_leaf_cnt  = nullptr;

    int*   d_query_offsets = nullptr;
    int*   d_pair_leaf_a   = nullptr;
    int*   d_pair_qid_a    = nullptr;
    int*   d_pair_leaf_b   = nullptr;
    int*   d_pair_qid_b    = nullptr;

    float* d_out_dists  = nullptr;
    int*   d_out_ids    = nullptr;
    float* d_final_dists= nullptr;
    int*   d_final_ids  = nullptr;
    void*  d_cub_tmp    = nullptr;
    size_t cub_bytes    = 0;

    cudaStream_t stream = nullptr;
};

void route_gpu_v32(
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
    int K1, int K2, int K3, int Kr, int ef,
    int batch_size,
    SearchWorkspace& ws);

void gpu_block_search_v32(
    int B, int n_blks, int d_proj,
    int ef_cells, int degree, int max_ls, int entry_per_cell,
    const int*   d_block_adj,
    const float* d_blk_proj,
    const float* d_blk_norm,
    const int*   d_pair_blk_start,
    const int*   d_pair_blk_count,
    SearchWorkspace& ws);

void gpu_build_and_sort_pairs_v32(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws);

void launch_leaf_flat_v32(
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

void launch_final_merge_v32(
    int nq, int n_pairs, int klocal, int k,
    SearchWorkspace& ws);

} // namespace hblock_v32
