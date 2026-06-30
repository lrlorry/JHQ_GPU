#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v5: v4 architecture + CUDA Graph.
//
// Key changes vs v4_batched_query:
//   * All kernels run on a dedicated per-workspace CUDA stream.
//   * The pipeline is captured into a cudaGraph on the first search call,
//     then every subsequent batch executes via a single cudaGraphLaunch.
//   * thrust::sort replaced by batched_topk_final_kernel (one block per query,
//     k rounds of block-wide argmin), which is fully graph-capturable.
//   * d_q_batch holds the current batch at a fixed device address so the
//     graph's pointers never change between launches.
//   * D→H copies are async on the same stream; a single cudaStreamSynchronize
//     at the end of search() replaces one blocking cudaMemcpy per batch.

struct SearchWorkspace {
    int batch_cap = 0, ck_cap = 0, k_cap = 0;

    float*    d_q_batch       = nullptr;  // [batch_cap, d]  fixed-addr query input
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
    const float*   d_queries,
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
