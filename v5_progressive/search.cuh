#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v5: batched progressive search over locality-sorted, layer-major codes.
//
// Key design changes vs v4_batched_query:
//   * Every kernel covers a full batch of B queries simultaneously
//     (one CUDA block per query for selection / scan steps).
//   * Storage is layer-major: primary[m, pos] and residual[byte, pos], so
//     progressive stages can read only the code layers they need.
//   * Search first scores a short primary prefix, keeps stage1 survivors,
//     then reads more primary layers, keeps stage2 survivors, then reads
//     the full primary code, and finally refines alpha*k candidates with
//     the residual layer.
//
// Buffer layout (every "batch-sized" buffer uses stride cap_per_query
// or ck so query bqi occupies [bqi*stride, (bqi+1)*stride)):
//   d_q_rot       [batch_cap, d]
//   d_dots        [batch_cap, nlist]  centroid dot products (col-major [nlist,batch])
//   d_probe_ids   [batch_cap, nprobe]
//   d_probe_offsets [batch_cap, nprobe+1]
//   d_query_total [batch_cap]
//   d_lut         [batch_cap, M*Ds*K1D]
//   d_cand_dist   [batch_cap, cap_per_query]
//   d_cand_pos    [batch_cap, cap_per_query]
//   d_keys        [batch_cap, cap_per_query]  uint64 composite sort key
//   d_topck_pos   [batch_cap, ck_cap]
//   d_topck_primary [batch_cap, ck_cap]
//   d_lut_r       [batch_cap, d*Kr]
//   d_comp_dists  [batch_cap, ck_cap]
//   d_keys2       [batch_cap, ck_cap]
//   d_final_ids   [batch_cap, k_cap]
//   d_final_dists [batch_cap, k_cap]

struct SearchWorkspace {
    int       batch_cap     = 0;
    long long cap_per_query = 0;
    int       ck_cap        = 0;
    int       k_cap         = 0;

    float*    d_q_rot           = nullptr;
    float*    d_dots            = nullptr;
    int*      d_probe_ids       = nullptr;
    int*      d_probe_offsets   = nullptr;
    int*      d_query_total     = nullptr;
    float*    d_lut             = nullptr;
    float*    d_cand_dist       = nullptr;
    int*      d_cand_pos        = nullptr;
    uint64_t* d_keys            = nullptr;
    int*      d_topck_pos       = nullptr;
    float*    d_topck_primary   = nullptr;
    float*    d_lut_r           = nullptr;
    float*    d_comp_dists      = nullptr;
    uint64_t* d_keys2           = nullptr;
    int*      d_final_ids       = nullptr;
    float*    d_final_dists     = nullptr;
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
    const uint8_t*        d_list_primary_lm,
    const uint8_t*        d_list_res_lm,
    const float*          d_list_corr,
    const float*          d_queries,      // [nq, d] already on GPU
    int nq, int N, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    int batch_size,
    int short_M, int mid_M, int stage1, int stage2,
    SearchWorkspace&      ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
