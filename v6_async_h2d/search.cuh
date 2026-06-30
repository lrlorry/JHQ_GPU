#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v6: v5 (CUDA Graph) + async H2D via pinned host buffer.
//
// Key changes vs v5_cuda_graph:
//   * Pre-allocated pinned host buffer (h_q_pinned) replaces the per-call
//     cudaMalloc + blocking cudaMemcpy(H2D) + cudaFree in search().
//   * search_gpu() accepts the raw host query pointer; it memcpy's each batch
//     to h_q_pinned then fires an async H2D into d_q_batch on ws.stream,
//     chaining directly into the graph launch on the same stream.
//   * No device-side intermediate buffer (d_queries) needed at all.

struct SearchWorkspace {
    int batch_cap = 0, ck_cap = 0, k_cap = 0;

    float*    h_q_pinned    = nullptr;  // [batch_cap, d]  pinned host staging
    float*    d_q_batch     = nullptr;  // [batch_cap, d]  fixed-addr device input
    float*    d_q_rot       = nullptr;
    float*    d_dots        = nullptr;
    float*    d_lut         = nullptr;
    int*      d_probe_ids   = nullptr;
    int*      d_probe_offsets = nullptr;
    int*      d_query_total = nullptr;
    int*      d_topck_pos   = nullptr;
    float*    d_topck_primary = nullptr;
    float*    d_lut_r       = nullptr;
    float*    d_comp_dists  = nullptr;
    int*      d_final_ids   = nullptr;
    float*    d_final_dists = nullptr;

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
    const float*   h_queries,   // host pointer — no device intermediate needed
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
