#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v11_outerlut: restructure scan to outer-codeword / inner-candidate.
//
// Key change vs v10_bytelut:
//   * scan_ivf_outerlut_kernel loops over m=0..95 in the OUTER loop.
//     At each m, 256 threads cooperatively load the 256-entry sub-table
//     (1 KB) into shared memory; then all candidates look up from shared.
//   * Shared lookup: zero bank conflict (32 threads → 32 different entries
//     in a 256-entry table → each in a different bank), and ~1-cycle latency
//     vs ~30-cycle L2 latency in v10.
//   * Per-thread state: abs_pos[MAX_CANDS] + partial[MAX_CANDS] live in
//     registers (not shared), keeping shared memory at 10 KB (same as v10).
//   * Fallback: when nprobe is large (n_cands_per_thread > MAX_CANDS), the
//     kernel falls back to v10's inner-m approach to avoid register overflow.

struct SearchWorkspace {
    int batch_cap = 0, ck_cap = 0, k_cap = 0;

    float*    h_q_pinned      = nullptr;
    float*    d_q_batch       = nullptr;
    float*    d_q_rot         = nullptr;
    float*    d_dots          = nullptr;
    float*    d_byte_lut      = nullptr;  // [batch_cap, M, 256]
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
