#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v15 {

static constexpr int TOP_P  = 4;
static constexpr int K_MAX  = 64;
static constexpr int K1_MAX = 128;  // max K1

struct SearchWorkspace {
    int batch_cap = 0;
    int max_pairs = 0;
    int ck3_ci    = 0;   // ck3 / ck1 — leaf budget per query per ci

    // ── Routing buffers (same as v14) ─────────────────────────────────────────
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
    float* d_lut_fine  = nullptr;  // full LUT [batch_cap × d × Kr]
    int*   h_leaf_cnt  = nullptr;

    // ── v15: ci group precomputation ──────────────────────────────────────────
    // d_ci_lists[ci][slot] = global query idx visiting ci (slot < d_ci_counts[ci])
    int*   d_ci_lists  = nullptr;  // [K1_MAX × batch_cap]
    int*   d_ci_counts = nullptr;  // [K1_MAX]
    int*   h_ci_counts = nullptr;  // pinned

    // ── v15: per-ci compact buffers ───────────────────────────────────────────
    // Sized for worst-case n_ci = batch_cap; in practice avg = batch_cap*ck1/K1
    float* d_lut_ci         = nullptr;  // [batch_cap × d × Kr]  (L2-resident per ci)
    int*   d_leaf_sel_ci    = nullptr;  // [batch_cap × ck3_ci]
    int*   d_pair_leaf_ci_a = nullptr;  // [batch_cap × ck3_ci]
    int*   d_pair_qid_ci_a  = nullptr;
    int*   d_pair_leaf_ci_b = nullptr;
    int*   d_pair_qid_ci_b  = nullptr;
    float* d_out_dists_ci   = nullptr;  // [batch_cap × ck3_ci × TOP_P]
    int*   d_out_ids_ci     = nullptr;
    int*   d_q_offsets_ci   = nullptr;  // [batch_cap + 1]

    // ── v15: global accumulator heap + per-ci topk ────────────────────────────
    float* d_heap_vals    = nullptr;  // [batch_cap × K_MAX]
    int*   d_heap_ids     = nullptr;
    float* d_ci_topk_vals = nullptr;  // [batch_cap × K_MAX]
    int*   d_ci_topk_ids  = nullptr;

    // ── Final output (D2H) ────────────────────────────────────────────────────
    float* h_final_dists = nullptr;
    int*   h_final_ids   = nullptr;

    // ── CUB temp storage ──────────────────────────────────────────────────────
    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;

    cudaStream_t stream = nullptr;
};

void init_heap(float* vals, int* ids, int n, cudaStream_t s);

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

void precompute_ci_groups(int nq, int ck1, int K1, SearchWorkspace& ws);

void process_ci(
    int ci, int n_ci, int K2, int n_leaf_blocks,
    int ck2, int ck3_ci, int leaf_cap_ci, int k,
    const int*     d_pair_blk_start,
    const int*     d_pair_blk_count,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    int d, int Kr, int Br, int bpv, int leaf_size,
    SearchWorkspace& ws);

} // namespace hblock_v15
