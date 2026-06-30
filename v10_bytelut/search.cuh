#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v10_bytelut: replace [M][Ds][K1D] shared-mem LUT with [M][256] byte LUT.
//
// Key changes vs v7_spin_sync:
//   * build_byte_lut_kernel precomputes byte_lut[B][M][256]: one entry per
//     (query, codeword, byte_value).  Cost: B*M*256*Ds MADs — negligible.
//   * scan_ivf_bytelut_kernel inner loop reduces from M*Ds=768 ops to M=96 ops
//     per candidate, eliminating the 16-way shared-mem bank conflict from K1D=2.
//   * list_primary read uses uint32 (4-byte aligned) to halve global load count.
//   * Shared mem drops from 16KB to 10KB → SM occupancy 75% → 100%.

struct SearchWorkspace {
    int batch_cap = 0, ck_cap = 0, k_cap = 0;

    float*    h_q_pinned      = nullptr;  // [batch_cap, d]  pinned host staging
    float*    d_q_batch       = nullptr;  // [batch_cap, d]  fixed-addr device input
    float*    d_q_rot         = nullptr;
    float*    d_dots          = nullptr;
    float*    d_byte_lut      = nullptr;  // [batch_cap, M, 256]  pre-expanded LUT
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
