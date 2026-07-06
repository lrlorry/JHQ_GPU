#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace hblock_v9 {

static constexpr int DISPATCH_BATCH = 256;
static constexpr int MAX_PAIRS      = 16384;
static constexpr int TOP_P          = 4;

// ── Per-query aggregation state (CPU side, Flink keyBy query_id) ──────────────
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
    int batch_cap = 0;

    // ── Routing buffers (GPU + pinned host) ───────────────────────────────────
    float* h_q_pinned  = nullptr;  // [B × d] pinned
    float* d_q_batch   = nullptr;
    float* d_q_rot     = nullptr;
    float* d_q_r1      = nullptr;
    float* d_q_r2      = nullptr;
    float* d_dots1     = nullptr;
    float* d_dots2     = nullptr;
    int*   d_top1_ids  = nullptr;
    int*   d_top2_ids  = nullptr;
    int*   d_leaf_sel  = nullptr;  // [B × ck3] stays on GPU for dispatch
    int*   d_leaf_cnt  = nullptr;
    float* d_lut_fine  = nullptr;  // [B × d × Kr] stays on GPU for dispatch

    int*   h_leaf_sel  = nullptr;  // [B × ck3] pinned, for CPU fan-out
    int*   h_leaf_cnt  = nullptr;  // [B] pinned, for CPU fan-out

    // ── Double dispatch buffers (ping-pong) ───────────────────────────────────
    // Index 0 and 1 alternate: while stream[0] runs, CPU packs buf[1], etc.
    int*   h_dispatch_leaf_ids[2] = {};   // [DISPATCH_BATCH] pinned × 2
    int*   h_dispatch_offsets [2] = {};   // [DISPATCH_BATCH+1] pinned × 2
    int*   h_dispatch_qids    [2] = {};   // [MAX_PAIRS] pinned × 2
    float* h_out_dists        [2] = {};   // [MAX_PAIRS × TOP_P] pinned × 2
    int*   h_out_ids          [2] = {};   // [MAX_PAIRS × TOP_P] pinned × 2

    int*   d_dispatch_leaf_ids[2] = {};
    int*   d_dispatch_offsets [2] = {};
    int*   d_dispatch_qids    [2] = {};
    float* d_out_dists        [2] = {};
    int*   d_out_ids          [2] = {};

    // ── Two CUDA streams + events ─────────────────────────────────────────────
    // streams[0]: also used for routing phase
    // streams[0..1]: alternating dispatch streams
    // d2h_events[b]: fires when D2H of buf[b] is complete → safe to merge
    cudaStream_t streams    [2] = {};
    cudaEvent_t  d2h_events [2] = {};

    void*  d_cub_tmp  = nullptr;
    size_t cub_bytes  = 0;
};

// ── Routing: GPU pipeline → d_leaf_sel + d_lut_fine resident on GPU ──────────
// Also D2Hs h_leaf_sel + h_leaf_cnt (syncs stream before returning)
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

// ── Kernel-only launch (caller manages all memcpy) ────────────────────────────
// Separation of kernel vs memcpy lets the caller pipeline H2D/D2H across two
// streams for true CPU-GPU overlap.
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

} // namespace hblock_v9
