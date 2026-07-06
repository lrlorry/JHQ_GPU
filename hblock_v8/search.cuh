#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace hblock_v8 {

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int DISPATCH_BATCH = 256;  // leaves per GPU kernel launch
static constexpr int MAX_PAIRS      = 16384; // max (leaf,query) pairs per dispatch
static constexpr int TOP_P          = 4;     // candidates retained per (leaf,query)

// ── Per-query aggregation state (CPU side) ────────────────────────────────────
struct QueryState {
    int   received  = 0;
    int   heap_size = 0;
    float dists[64] = {};  // top-k heap (unsorted, size <= k)
    int   ids  [64] = {};

    void push(float dist, int id, int k) {
        if (id < 0) return;
        if (heap_size < k) {
            dists[heap_size] = dist;
            ids  [heap_size] = id;
            heap_size++;
            return;
        }
        // replace worst if better
        int worst = 0;
        for (int i = 1; i < k; ++i)
            if (dists[i] > dists[worst]) worst = i;
        if (dist < dists[worst]) { dists[worst] = dist; ids[worst] = id; }
    }
};

// ── Search workspace ──────────────────────────────────────────────────────────
struct SearchWorkspace {
    int batch_cap = 0;   // max queries in one routing batch
    int nq_cap    = 0;   // same (currently nq must be <= batch_cap)

    // ── Routing buffers (GPU) ─────────────────────────────────────────────────
    float* h_q_pinned    = nullptr;  // pinned host copy of query batch
    float* d_q_batch     = nullptr;
    float* d_q_rot       = nullptr;
    float* d_q_r1        = nullptr;
    float* d_q_r2        = nullptr;
    float* d_dots1       = nullptr;
    float* d_dots2       = nullptr;
    int*   d_top1_ids    = nullptr;
    int*   d_top2_ids    = nullptr;
    int*   d_leaf_sel    = nullptr;  // [B × ck3] kept alive for dispatch phase
    int*   d_leaf_cnt    = nullptr;
    float* d_lut_fine    = nullptr;  // [B × d × Kr] kept alive for dispatch phase

    // ── Dispatch buffers (GPU, small, reused across dispatches) ───────────────
    int*   d_dispatch_leaf_ids = nullptr;  // [DISPATCH_BATCH]
    int*   d_dispatch_offsets  = nullptr;  // [DISPATCH_BATCH + 1]
    int*   d_dispatch_qids     = nullptr;  // [MAX_PAIRS]
    float* d_out_dists         = nullptr;  // [MAX_PAIRS × TOP_P]
    int*   d_out_ids           = nullptr;  // [MAX_PAIRS × TOP_P]

    // ── Pinned host buffers for fast D2H ──────────────────────────────────────
    int*   h_leaf_sel          = nullptr;  // [B × ck3] pinned, for CPU fan-out
    int*   h_leaf_cnt          = nullptr;  // [B]        pinned, for CPU fan-out
    int*   h_dispatch_leaf_ids = nullptr;  // [DISPATCH_BATCH] pinned
    int*   h_dispatch_offsets  = nullptr;  // [DISPATCH_BATCH + 1] pinned
    int*   h_dispatch_qids     = nullptr;  // [MAX_PAIRS] pinned
    float* h_out_dists         = nullptr;  // [MAX_PAIRS × TOP_P] pinned
    int*   h_out_ids           = nullptr;  // [MAX_PAIRS × TOP_P] pinned

    // ── Routing pipeline temp (GPU, for H2D) ──────────────────────────────────
    void*  d_cub_tmp   = nullptr;  // CUB sort temp (routing phase)
    size_t cub_bytes   = 0;

    cudaStream_t stream = nullptr;
};

// ── Route all nq queries: fills d_leaf_sel and d_lut_fine ─────────────────────
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

// ── Single GPU dispatch: one kernel launch for up to DISPATCH_BATCH leaves ────
void dispatch_leaves(
    const int*     d_dispatch_leaf_ids,  // [n_leaves]
    const int*     d_dispatch_offsets,   // [n_leaves + 1]
    const int*     d_dispatch_qids,      // [total_pairs]
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,           // [nq, d, Kr]
    float*         d_out_dists,          // [total_pairs × TOP_P]
    int*           d_out_ids,            // [total_pairs × TOP_P]
    int n_leaves, int total_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size,
    cudaStream_t stream);

} // namespace hblock_v8
