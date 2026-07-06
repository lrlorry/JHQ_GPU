#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v14 {

static constexpr int TOP_P  = 4;
static constexpr int K_MAX  = 64;   // max supported k for search()

struct SearchWorkspace {
    int batch_cap = 0;
    int max_pairs = 0;

    // ── Routing buffers ───────────────────────────────────────────────────────
    float* h_q_pinned  = nullptr;
    float* d_q_batch   = nullptr;
    float* d_q_rot     = nullptr;
    float* d_q_r1      = nullptr;
    float* d_q_r2      = nullptr;
    float* d_dots1     = nullptr;
    float* d_dots2     = nullptr;
    int*   d_top1_ids  = nullptr;
    int*   d_top2_ids  = nullptr;
    int*   d_leaf_sel  = nullptr;
    int*   d_leaf_cnt  = nullptr;
    float* d_lut_fine  = nullptr;
    int*   h_leaf_cnt  = nullptr;   // D2H'd for n_pairs sum

    // ── GPU pair build + leaf sort (ping-pong) ────────────────────────────────
    // Also reused for qid sort:
    //   leaf sort : key_in=leaf_a, val_in=qid_a → key_out=leaf_b, val_out=qid_b
    //   iota fill : leaf_a ← [0..n_pairs-1]
    //   qid  sort : key_in=qid_b,  val_in=leaf_a → key_out=leaf_b, val_out=qid_a
    // After qid sort: qid_a holds permutation (qid-sorted → leaf-sorted pair index)
    int*   d_query_offsets = nullptr;   // [batch_cap + 1]
    int*   d_pair_leaf_a   = nullptr;   // [max_pairs]
    int*   d_pair_qid_a    = nullptr;   // [max_pairs]
    int*   d_pair_leaf_b   = nullptr;   // [max_pairs]
    int*   d_pair_qid_b    = nullptr;   // [max_pairs]

    // ── Leaf kernel output (device, read by segmented_topk) ──────────────────
    float* d_out_dists   = nullptr;     // [max_pairs × TOP_P]
    int*   d_out_ids     = nullptr;     // [max_pairs × TOP_P]

    // ── Final top-k (GPU → host) ──────────────────────────────────────────────
    float* d_final_dists = nullptr;     // [batch_cap × K_MAX]
    int*   d_final_ids   = nullptr;     // [batch_cap × K_MAX]
    float* h_final_dists = nullptr;     // pinned
    int*   h_final_ids   = nullptr;     // pinned

    // ── CUB temp storage (shared for scan + leaf sort + qid sort) ────────────
    void*  d_cub_tmp   = nullptr;
    size_t cub_bytes   = 0;

    cudaStream_t stream = nullptr;
};

void route_queries(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_route1_cents, const float* d_route1_norms,
    const float*   d_route2_cents, const float* d_route2_norms,
    const float*   d_fine_c1d,
    const int*     d_pair_blk_start, const int* d_pair_blk_count,
    const float*   h_queries,
    int nq, int d, int K1, int K2, int Kr, int Br,
    int leaf_size, int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws);

// Sort pairs by leaf_id, building flat (leaf, qid) pair array on GPU.
void gpu_build_and_sort_pairs(
    int nq, int n_pairs, int n_leaf_blocks,
    int ck3, SearchWorkspace& ws);

// Leaf PQ distance kernel: one block per (query, leaf) pair, ~1KB smem.
void launch_leaf_flat(
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

// GPU-side top-k aggregation:
//   1. iota fill d_pair_leaf_a
//   2. CUB radix sort by qid (10 bits) → permutation in d_pair_qid_a
//   3. segmented_topk_kernel: one block per query, writes d_final_dists/ids
// Writes n_pairs to d_query_offsets[nq] so last query's segment end is valid.
void gpu_merge_topk(
    int nq, int n_pairs, int k,
    SearchWorkspace& ws);

} // namespace hblock_v14
