#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace jhq_gpu {

void launch_primary_encode(
    const float*   d_y,
    uint8_t*       d_codes,
    const float*   d_c1d,
    int N, int d, int M, int Ds, int K1D, int bits_per_dim,
    cudaStream_t stream = 0);

void launch_residual_encode(
    const float*   d_y,
    const uint8_t* d_primary,
    uint8_t*       d_res_codes,
    float*         d_corrections,
    const float*   d_c1d,
    const float*   d_res_c1d,
    int N, int d, int M, int Ds, int K1D, int Kr, int Br, int bpv, int bits_per_dim,
    cudaStream_t stream = 0);

} // namespace jhq_gpu
