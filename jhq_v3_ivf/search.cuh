#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v3: IVF-JHQ search.
// The index stores vectors in concatenated inverted lists. A query first
// selects nprobe centroid lists, scans only those lists with primary ADC,
// then refines the top alpha*k candidates with residual codes.

struct SearchWorkspace {
    float*    d_q_rot      = nullptr;
    float*    d_lut        = nullptr;
    float*    d_dists      = nullptr;
    int*      d_indices    = nullptr;
    float*    d_lut_r      = nullptr;
    float*    d_comp_dists = nullptr;
    uint64_t* d_packed     = nullptr;
    float*    d_cent_dists = nullptr;
    int*      d_cent_ids   = nullptr;
    int*      d_probe_ids  = nullptr;
    int*      d_probe_offsets = nullptr;
    int*      d_final_ids  = nullptr;
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
    const float*          d_queries,
    int nq, int N, int d, int M, int Ds, int K1D, int Kr, int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    SearchWorkspace&      ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
