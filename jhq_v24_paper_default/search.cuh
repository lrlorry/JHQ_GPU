#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v12_transposed: transpose list_primary from [N, M] to [M, N] so that
// the scan inner loop accesses list_primary_t[m * N + abs_pos].
//
// With the original [N, M] layout, 32 warp threads each read the m-th byte
// of their own candidate: addresses differ by N_stride × M = 24 576 bytes →
// 32 separate cache lines loaded, 4 bytes used per line (98% waste).
//
// With [M, N] layout, 32 warp threads read 32 consecutive abs_pos values at
// the same m: 32 consecutive bytes = 1 cache line, 100% utilisation.
// This reduces list_primary L2/HBM traffic by 32× per warp transaction.
//
// All other changes from v10_bytelut are preserved:
//   - byte_lut [B, M, 256] (no bank conflict, v10)
//   - CUDA graph (v5+)
//   - spin-wait sync (v7)

// Element type of the primary lookup table, and the size the workspace
// allocates for it.
//
// This has to be in the header. search.cu writes the table and
// jhq_gpu_index.cu allocates it, and a default only one of them can see gives
// a buffer sized for halves and filled with floats: at M=96 the overrun landed
// on other allocations and the run still reported plausible recall, and at
// M=192 it faulted.
//
// Float is the default. Equation 6 sums exact per-subspace distances, and
// storing each as a half rounds it to eleven mantissa bits before the sum ever
// happens. JHQ_LUT32=0 restores the half table for measurement.
#ifndef JHQ_LUT32
#define JHQ_LUT32 1
#endif
#if JHQ_LUT32
typedef float jhq_lut_t;
#else
typedef __half jhq_lut_t;
#endif

struct SearchWorkspace {
    int batch_cap = 0, ck_cap = 0, k_cap = 0;

#if JHQ_STEP_TIMING
    // Per-stage timing bypasses the captured graph: events cannot be read back
    // from inside one, and the question the timing answers -- which stage owns
    // the time -- is worth losing graph launch overhead for. Enabled with
    // -DJHQ_STEP_TIMING=1; the default build is untouched.
    static constexpr int N_STEP_EVENTS = 9;
    cudaEvent_t ev_step[N_STEP_EVENTS] = {};
    cudaEvent_t ev_h2d_start = nullptr, ev_h2d_done = nullptr, ev_d2h_done = nullptr;
    int         timing_count = 0;
    double      acc_ms[N_STEP_EVENTS] = {};
#endif

    float*    h_q_pinned      = nullptr;
    float*    d_q_batch       = nullptr;
    float*    d_q_rot         = nullptr;
    float*    d_dots          = nullptr;
    jhq_lut_t* d_byte_lut     = nullptr;   // [batch_cap, M, 256]
    int*      d_probe_ids     = nullptr;
    int*      d_probe_offsets = nullptr;
    int*      d_query_total   = nullptr;
    int*      d_topck_pos     = nullptr;
    float*    d_topck_primary = nullptr;
    float*    d_lut_r         = nullptr;
    float*    d_comp_dists    = nullptr;
    int*      d_final_ids     = nullptr;
    float*    d_final_dists   = nullptr;

    cudaStream_t    stream     = nullptr;
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_ck = 0, graph_nprobe = 0;
};

void search_gpu(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_cent,      // [M][K][Ds] product-quantiser centroids
    const float*   d_res_c1d,   // [M][Kr] per-subspace residual codebooks
    const float*   d_centroids,
    const float*   d_cent_norms,
    const int*     d_list_offsets,
    const int*     d_list_ids,
    const uint8_t* d_list_primary_t,   // [M, N] transposed
    const uint8_t* d_list_res,
    const float*   d_list_corr,
    const float*   h_queries,
    int nq, int d, int M, int Ds, int K, int Kr,
    int nlist, int nprobe,
    int Br, int bpv,
    float alpha, int k,
    int batch_size,
    int ntotal,                         // N — needed for [M,N] index
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
