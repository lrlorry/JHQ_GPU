#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

namespace jhq_gpu {

// Preallocated search workspace — owned by JHQGpuIndex, passed into search_gpu
// so that benchmarks don't measure malloc overhead.
//
// Sizes (N = ntotal):
//   d_q_rot      [d]
//   d_lut        [M * Ds * K1D]
//   d_dists      [N]
//   d_indices    [N]
//   d_lut_r      [d * Kr]
//   d_comp_dists [N]   (only ck ≤ N entries used per query)
struct SearchWorkspace {
    float* d_q_rot      = nullptr;
    float* d_lut        = nullptr;
    float* d_dists      = nullptr;
    int*   d_indices    = nullptr;
    float* d_lut_r      = nullptr;
    float* d_comp_dists = nullptr;
};

// Full search for nq queries.
// ws must be pre-allocated to the sizes above (JHQGpuIndex::alloc_workspace).
void search_gpu(
    cublasHandle_t        cublas,
    // permanent index data
    const float*          d_Pi,            // [d × d]
    const float*          d_c1d,           // [K1D]
    const float*          d_res_c1d,       // [Kr]
    const uint8_t*        d_primary_codes, // [N × M]
    const uint8_t*        d_res_codes,     // [N × bpv]
    const float*          d_corrections,   // [N]
    // queries (device)
    const float*          d_queries,       // [nq × d]
    // dimensions
    int nq, int N, int d, int M, int Ds, int K1D, int Kr,
    int Br, int bpv, int bits_per_dim,
    // search params
    float alpha, int k,
    // preallocated workspace
    SearchWorkspace&      ws,
    // outputs (host)
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
