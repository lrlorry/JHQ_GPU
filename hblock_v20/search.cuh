#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v20 {

static constexpr int TOP_P  = 4;
static constexpr int K_MAX  = 128;

struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;
    int rerank_r     = 0;
    int d_proj       = 0;

    float* h_q_pinned    = nullptr;
    int*   h_leaf_cnt    = nullptr;
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    float* d_q_batch   = nullptr;
    float* d_q_proj1   = nullptr;
    float* d_dots1     = nullptr;
    int*   d_top1_ids  = nullptr;

    float* d_r1_beam   = nullptr;
    int*   d_top2_beam = nullptr;
    int*   d_top3_beam = nullptr;

    float* d_q_r3      = nullptr;

    int*   d_leaf_sel  = nullptr;
    int*   d_leaf_cnt  = nullptr;
    float* d_lut_fine  = nullptr;

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

void route_queries_v20(
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

void gpu_build_and_sort_pairs_v20(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws);

void launch_leaf_flat_v20(
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

void gpu_merge_pq_topk_v20(
    int nq, int n_pairs, int rerank_r,
    SearchWorkspace& ws);

void launch_gather_vecs_v20(
    const float*   d_base_vecs,
    const int*     d_cand_ids,
    float*         d_cand_vecs,
    int B, int R, int d,
    cudaStream_t   s);

void launch_exact_rerank_v20(
    const float*   d_q_batch,
    const float*   d_cand_vecs,
    const int*     d_cand_ids,
    float*         d_exact_dists,
    float*         d_final_dists,
    int*           d_final_ids,
    int B, int R, int d, int k,
    cudaStream_t   s);

} // namespace hblock_v20
