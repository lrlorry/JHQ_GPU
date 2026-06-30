#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v4: batched query search -- redesigned for correctness and QPS.
//
// Pipeline per batch of B queries:
//   1. GEMM: rotate all B queries at once.
//   2. GEMM: compute B × nlist centroid dot products at once.
//   3. select_probes_kernel: one block per query, thread-0 sequential argmin
//      selects top-nprobe centroids and builds per-query prefix-sum offsets.
//      All on GPU, no host round-trip.
//   4. build_primary_lut_batched_kernel: B × lut_size LUT entries.
//   5. scan_ivf_batched_topk_kernel: one block per query.
//        - Each thread scans its stride of the scanned IVF lists while
//          keeping a local register top-K_LOCAL buffer (insertion sort).
//        - After the scan, per-thread candidates are gathered into shared
//          memory, and the block selects top-ck via iterative argmin.
//        - Directly writes top-ck (primary dist + abs_pos) into compact
//          B×ck output buffers -- no huge intermediate sort needed.
//   6. build_residual_lut_batched_kernel.
//   7. residual_refine_batched_kernel.
//   8. pack_keys2_kernel + thrust::sort_by_key (only B×ck elements).
//   9. gather_final_kernel.
//  10. Single cudaMemcpy(D→H) per batch.
//
// Workspace buffers (all modest in size):
//   d_q_rot          [batch_cap × d]
//   d_dots           [batch_cap × nlist]
//   d_probe_ids      [batch_cap × nprobe]
//   d_probe_offsets  [batch_cap × (nprobe+1)]
//   d_query_total    [batch_cap]
//   d_lut            [batch_cap × M×Ds×K1D]
//   d_topck_pos      [batch_cap × ck_cap]
//   d_topck_primary  [batch_cap × ck_cap]
//   d_lut_r          [batch_cap × d×Kr]
//   d_comp_dists     [batch_cap × ck_cap]
//   d_keys2          [batch_cap × ck_cap]   uint64 composite key for final sort
//   d_final_ids      [batch_cap × k_cap]
//   d_final_dists    [batch_cap × k_cap]

struct SearchWorkspace {
    int batch_cap = 0;
    int ck_cap    = 0;
    int k_cap     = 0;

    float*    d_q_rot          = nullptr;
    float*    d_dots           = nullptr;
    int*      d_probe_ids      = nullptr;
    int*      d_probe_offsets  = nullptr;
    int*      d_query_total    = nullptr;
    float*    d_lut            = nullptr;
    int*      d_topck_pos      = nullptr;
    float*    d_topck_primary  = nullptr;
    float*    d_lut_r          = nullptr;
    float*    d_comp_dists     = nullptr;
    uint64_t* d_keys2          = nullptr;
    int*      d_final_ids      = nullptr;
    float*    d_final_dists    = nullptr;
};

void search_gpu(
    cublasHandle_t        cublas,
    const float*          d_Pi,
    const float*          d_c1d,
    const float*          d_res_c1d,
    const float*          d_centroids,
    const float*          d_cent_norms,
    const int*            d_list_offsets,
    const int*            d_list_ids,
    const uint8_t*        d_list_primary,
    const uint8_t*        d_list_res,
    const float*          d_list_corr,
    const float*          d_queries,      // [nq, d] already on GPU
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    int batch_size,
    SearchWorkspace&      ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
