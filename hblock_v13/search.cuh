#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v13 {

static constexpr int TOP_P = 4;

struct QueryState {
    int   heap_size = 0;
    float dists[64] = {};
    int   ids  [64] = {};

    void push(float dist, int id, int k) {
        if (id < 0) return;
        if (heap_size < k) {
            dists[heap_size] = dist;
            ids  [heap_size] = id;
            heap_size++;
            return;
        }
        int worst = 0;
        for (int i = 1; i < k; ++i)
            if (dists[i] > dists[worst]) worst = i;
        if (dist < dists[worst]) { dists[worst] = dist; ids[worst] = id; }
    }
};

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
    int*   d_leaf_sel  = nullptr;   // stays on GPU, fed to build_pairs_kernel
    int*   d_leaf_cnt  = nullptr;
    float* d_lut_fine  = nullptr;

    // h_leaf_sel removed: no longer D2H'd
    int*   h_leaf_cnt  = nullptr;   // D2H'd for n_pairs computation

    // ── GPU pair building + sort buffers ──────────────────────────────────────
    // CUB exclusive scan: d_leaf_cnt → d_query_offsets
    int*   d_query_offsets = nullptr;   // [batch_cap + 1]

    // Radix sort ping-pong buffers (CUB SortPairs needs separate in/out)
    int*   d_pair_leaf_a = nullptr;     // [max_pairs] sort key   input
    int*   d_pair_qid_a  = nullptr;     // [max_pairs] sort value input
    int*   d_pair_leaf_b = nullptr;     // [max_pairs] sort key   output
    int*   d_pair_qid_b  = nullptr;     // [max_pairs] sort value output

    // ── Output buffers ────────────────────────────────────────────────────────
    int*   h_pair_qids  = nullptr;      // [max_pairs] pinned, CPU merge needs qid order
    float* h_out_dists  = nullptr;      // [max_pairs × TOP_P] pinned
    int*   h_out_ids    = nullptr;      // [max_pairs × TOP_P] pinned
    float* d_out_dists  = nullptr;
    int*   d_out_ids    = nullptr;

    // ── CUB temp storage (shared for scan + sort) ─────────────────────────────
    void*  d_cub_tmp   = nullptr;
    size_t cub_bytes   = 0;

    cudaStream_t stream = nullptr;
};

// route_queries: same GPU pipeline, but does NOT D2H h_leaf_sel.
// Only D2H h_leaf_cnt (tiny: nq×4B). d_leaf_sel stays on GPU for build_pairs.
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

// GPU pipeline: scan d_leaf_cnt → d_query_offsets, then build flat pairs,
// then CUB radix sort by leaf_blk (14 bits). All ops on ws.stream.
// Returns n_pairs (= CPU sum of h_leaf_cnt, computed before calling this).
void gpu_build_and_sort_pairs(
    int nq, int n_pairs, int n_leaf_blocks,
    int ck3, SearchWorkspace& ws);

// One block per pair, ~1KB smem → 100% occupancy.
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

} // namespace hblock_v13
