#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace hblock_v10 {

static constexpr int TOP_P = 4;

// ── Per-query aggregation state (CPU side) ────────────────────────────────────
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

// ── Workspace ─────────────────────────────────────────────────────────────────
struct SearchWorkspace {
    int batch_cap  = 0;
    int max_leaves = 0;   // n_leaf_blocks_ at alloc time
    int max_pairs  = 0;   // batch_size × ck3

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
    float* d_lut_fine  = nullptr;  // stays on GPU for leaf kernel

    int*   h_leaf_sel  = nullptr;  // pinned, CPU fan-out
    int*   h_leaf_cnt  = nullptr;

    // ── Single dispatch buffers (all leaves packed at once) ───────────────────
    int*   h_dispatch_leaf_ids = nullptr;  // [max_leaves]
    int*   h_dispatch_offsets  = nullptr;  // [max_leaves + 1]
    int*   h_dispatch_qids     = nullptr;  // [max_pairs]
    float* h_out_dists         = nullptr;  // [max_pairs × TOP_P]
    int*   h_out_ids           = nullptr;  // [max_pairs × TOP_P]

    int*   d_dispatch_leaf_ids = nullptr;
    int*   d_dispatch_offsets  = nullptr;
    int*   d_dispatch_qids     = nullptr;
    float* d_out_dists         = nullptr;
    int*   d_out_ids           = nullptr;

    cudaStream_t stream = nullptr;

    void*  d_cub_tmp  = nullptr;
    size_t cub_bytes  = 0;
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

void launch_leaf_flink(
    const int*     d_dispatch_leaf_ids,
    const int*     d_dispatch_offsets,
    const int*     d_dispatch_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_leaves,
    int d, int Kr, int Br, int bpv, int leaf_size,
    cudaStream_t   stream);

} // namespace hblock_v10
