#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v9_step_timing: no CUDA Graph; 9 CUDA events bracket every pipeline step
// so we can see exactly which kernel consumes the 77 ms observed in v8.
//
// Steps (event indices):
//   [0]→[1]  GEMM rotate  (Pi × Q_batch)
//   [1]→[2]  GEMM centroid (C^T × Q_rot)
//   [2]→[3]  select_probes_kernel
//   [3]→[4]  build_primary_lut_kernel
//   [4]→[5]  scan_ivf_batched_topk_kernel
//   [5]→[6]  build_residual_lut_kernel
//   [6]→[7]  residual_refine_kernel
//   [7]→[8]  batched_topk_final_kernel

static constexpr int N_STEP_EVENTS = 9;
static constexpr const char* STEP_NAMES[] = {
    "gemm_rotate", "gemm_centroid", "select_probes",
    "build_lut1",  "scan_ivf",      "build_lut_r",
    "residual_refine", "topk_final"
};

struct SearchWorkspace {
    int batch_cap = 0, ck_cap = 0, k_cap = 0;

    float*    h_q_pinned      = nullptr;
    float*    d_q_batch       = nullptr;
    float*    d_q_rot         = nullptr;
    float*    d_dots          = nullptr;
    float*    d_lut           = nullptr;
    int*      d_probe_ids     = nullptr;
    int*      d_probe_offsets = nullptr;
    int*      d_query_total   = nullptr;
    int*      d_topck_pos     = nullptr;
    float*    d_topck_primary = nullptr;
    float*    d_lut_r         = nullptr;
    float*    d_comp_dists    = nullptr;
    int*      d_final_ids     = nullptr;
    float*    d_final_dists   = nullptr;

    cudaStream_t stream = nullptr;

    // Per-step timing events (created in alloc_workspace)
    cudaEvent_t ev_step[N_STEP_EVENTS] = {};
    int timing_count = 0;
};

void search_gpu(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_c1d,
    const float*   d_res_c1d,
    const float*   d_centroids,
    const float*   d_cent_norms,
    const int*     d_list_offsets,
    const int*     d_list_ids,
    const uint8_t* d_list_primary,
    const uint8_t* d_list_res,
    const float*   d_list_corr,
    const float*   h_queries,
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
