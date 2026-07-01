#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// hblock: three-stage cascade
//   1. IVF primary scan (coalesced [M,N], same as v12) → top-ck1
//   2. Coarse cascade (byte_lut built with c1d_coarse, coalesced [M,N]) → top-ck2
//   3. Fine residual refine (same as v12) → top-k
//
// Coarse byte_lut reuses the same d_byte_lut buffer as primary:
//   after scan_ivf writes topck1, byte_lut is overwritten with coarse LUT,
//   then coarse_cascade reads it. No additional 94 MB buffer needed.
//
// ck1 = max(k, ceil(alpha1 * k))   — controls primary scan breadth
// ck2 = max(k, ceil(alpha2 * k))   — controls coarse cascade survivors (ck2 <= ck1)
//
// Memory vs v12:
//   + d_coarse_t_  [M,N]  — coarse codes in IVF order (same size as d_primary_t_)
//   + d_topck1_pos/primary — second topck buffer
//   Total extra: ~1× primary_t size ≈ M*N bytes

struct SearchWorkspace {
    int batch_cap = 0, ck1_cap = 0, ck2_cap = 0, k_cap = 0;

    float*    h_q_pinned      = nullptr;  // pinned host buffer
    float*    d_q_batch       = nullptr;  // [batch_cap, d]
    float*    d_q_rot         = nullptr;  // [batch_cap, d]
    float*    d_dots          = nullptr;  // [batch_cap, nlist]
    float*    d_byte_lut      = nullptr;  // [batch_cap, M, 256] — reused for primary/coarse

    int*      d_probe_ids     = nullptr;  // [batch_cap, nprobe]
    int*      d_probe_offsets = nullptr;  // [batch_cap, nprobe+1]
    int*      d_query_total   = nullptr;  // [batch_cap]

    // Primary scan output → ck1 candidates per query
    int*      d_topck1_pos     = nullptr;  // [batch_cap, ck1]
    float*    d_topck1_primary = nullptr;  // [batch_cap, ck1]

    // Coarse cascade output → ck2 survivors per query
    int*      d_topck2_pos     = nullptr;  // [batch_cap, ck2]
    float*    d_topck2_primary = nullptr;  // [batch_cap, ck2] (primary_dist of survivors)

    float*    d_lut_r         = nullptr;  // [batch_cap, d, Kr]
    float*    d_comp_dists    = nullptr;  // [batch_cap, ck2]
    int*      d_final_ids     = nullptr;  // [batch_cap, k]
    float*    d_final_dists   = nullptr;  // [batch_cap, k]

    cudaStream_t    stream     = nullptr;
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_ck1 = 0, graph_ck2 = 0, graph_nprobe = 0;
};

void search_hblock(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_c1d,
    const float*   d_c1d_coarse,
    const float*   d_res_c1d,
    const float*   d_centroids,
    const float*   d_cent_norms,
    const int*     d_list_offsets,
    const int*     d_list_ids,
    const uint8_t* d_primary_t,    // [M, N] primary codes in IVF order (transposed)
    const uint8_t* d_coarse_t,     // [M, N] coarse codes in IVF order (transposed)
    const uint8_t* d_list_res,     // [N, bpv] fine residual codes
    const float*   d_list_corr,    // [N] fine corrections
    const float*   h_queries,
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha1, float alpha2, int k,
    int batch_size, int ntotal,
    SearchWorkspace& ws,
    float* h_out_dists, int* h_out_ids);

} // namespace jhq_gpu
