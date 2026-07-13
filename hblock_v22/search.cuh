#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v22 {

static constexpr int TOP_P = 4;
static constexpr int K_MAX = 128;

// v22 workspace — identical to v20 EXCEPT:
//   • no gather_leaf_blocks step on GPU
//   • adds h_top1/2/3 (pinned D2H buffers) + h_block_sel (CPU traversal output)
//   • max_leaf_sel = graph_budget  (direct block count; no ck³ × blk/cell)
struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;   // == graph_budget
    int rerank_r     = 0;
    int d_proj       = 0;
    int ck1 = 0, ck2 = 0, ck3 = 0;

    // ── Pinned host ──────────────────────────────────────────────────────────
    float* h_q_pinned    = nullptr;
    int*   h_leaf_cnt    = nullptr;   // block count per query (CPU traversal output)
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    // D2H routing results (filled by route_gpu_v22, read by cpu_block_traverse)
    int*   h_top1_ids  = nullptr;   // [B, ck1]
    int*   h_top2_beam = nullptr;   // [B*ck1, ck2]
    int*   h_top3_beam = nullptr;   // [B*ck1*ck2, ck3]

    // CPU block traversal output → H2D
    int*   h_block_sel = nullptr;   // [B, graph_budget]

    // ── GPU routing (L1+L2+L3 beam, identical to v20) ────────────────────────
    float* d_q_batch   = nullptr;   // [B, d]
    float* d_q_proj1   = nullptr;   // [B, d_proj]
    float* d_dots1     = nullptr;   // [B, K1]
    int*   d_top1_ids  = nullptr;   // [B, ck1]
    float* d_r1_beam   = nullptr;   // [B*ck1, d]
    int*   d_top2_beam = nullptr;   // [B*ck1, ck2]
    int*   d_top3_beam = nullptr;   // [B*ck1*ck2, ck3]
    float* d_q_r3      = nullptr;   // [B, d]   best-path residual for LUT
    float* d_lut_fine  = nullptr;   // [B, d, Kr]

    // ── Leaf block selection (H2D from CPU traversal) ────────────────────────
    int*   d_leaf_sel  = nullptr;   // [B, max_leaf_sel]
    int*   d_leaf_cnt  = nullptr;   // [B]

    // ── Pair build + radix sort ───────────────────────────────────────────────
    int*   d_query_offsets = nullptr;
    int*   d_pair_leaf_a   = nullptr;
    int*   d_pair_qid_a    = nullptr;
    int*   d_pair_leaf_b   = nullptr;
    int*   d_pair_qid_b    = nullptr;

    // ── PQ scan output ────────────────────────────────────────────────────────
    float* d_out_dists = nullptr;
    int*   d_out_ids   = nullptr;

    // ── Merge → top-rerank_r ─────────────────────────────────────────────────
    float* d_pq_dists  = nullptr;
    int*   d_pq_ids    = nullptr;

    // ── Rerank ───────────────────────────────────────────────────────────────
    float* d_cand_vecs   = nullptr;
    float* d_exact_dists = nullptr;
    float* d_final_dists = nullptr;
    int*   d_final_ids   = nullptr;

    // ── CUB temp ─────────────────────────────────────────────────────────────
    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;

    cudaStream_t stream = nullptr;
};

// Full GPU routing: L1+L2+L3 beam + extract_best_r3 + build_fine_lut.
// D2H copies top1/top2/top3 beam results to pinned host (synchronized on return).
// NOTE: does NOT gather leaf blocks — that's done by cpu_block_traverse in the index.
void route_gpu_v22(
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

// Remaining pipeline (identical signature/impl to v20, just renamed)
void gpu_build_and_sort_pairs_v22(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws);

void launch_leaf_flat_v22(
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

void gpu_merge_pq_topk_v22(
    int nq, int n_pairs, int rerank_r,
    SearchWorkspace& ws);

void launch_gather_vecs_v22(
    const float* d_base_vecs,
    const int*   d_cand_ids,
    float*       d_cand_vecs,
    int B, int R, int d,
    cudaStream_t s);

void launch_exact_rerank_v22(
    const float* d_q_batch,
    const float* d_cand_vecs,
    const int*   d_cand_ids,
    float*       d_exact_dists,
    float*       d_final_dists,
    int*         d_final_ids,
    int B, int R, int d, int k,
    cudaStream_t s);

} // namespace hblock_v22
