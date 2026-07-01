#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// v2: top-ck via packed uint64_t (dist_bits<<32 | index).
// Non-negative floats sort correctly as uint32_t bit patterns, so uint64_t
// comparison gives dist-major ordering without zip_iterator.

struct SearchWorkspace {
    float*    d_q_rot      = nullptr;
    float*    d_lut        = nullptr;
    float*    d_dists      = nullptr;
    int*      d_indices    = nullptr;
    float*    d_lut_r      = nullptr;
    float*    d_comp_dists = nullptr;
    uint64_t* d_packed     = nullptr;  // [N]  dist_bits<<32 | index
};

void search_gpu(
    cublasHandle_t        cublas,
    const float*          d_Pi,
    const float*          d_c1d,
    const float*          d_res_c1d,
    const uint8_t*        d_primary_codes,
    const uint8_t*        d_res_codes,
    const float*          d_corrections,
    const float*          d_queries,
    int nq, int N, int d, int M, int Ds, int K1D, int Kr,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    SearchWorkspace&      ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
