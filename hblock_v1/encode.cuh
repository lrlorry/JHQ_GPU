#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace jhq_gpu {

// Primary JHQ encoding: d_y [N,d] → d_codes [N,M]
void launch_primary_encode(
    const float*   d_y,
    uint8_t*       d_codes,
    const float*   d_c1d,
    int N, int d, int M, int Ds, int K1D, int bits_per_dim,
    cudaStream_t stream = 0);

// Compute primary residuals: r[vid][j] = y[vid][j] - c1d[decode(primary[vid][j])]
void launch_compute_primary_residual(
    const float*   d_y,         // [N, d]
    const uint8_t* d_primary,   // [N, M]
    float*         d_residual,  // [N, d] output
    const float*   d_c1d,       // [K1D]
    int N, int d, int M, int Ds, int K1D, int bits_per_dim,
    cudaStream_t stream = 0);

// Coarse residual encode: same sub-quantizer structure as primary, applied to residuals.
// d_residual [N,d] + c1d_coarse [K1D] → d_coarse_codes [N,M]
void launch_coarse_residual_encode(
    const float*   d_residual,
    uint8_t*       d_coarse_codes,
    const float*   d_c1d_coarse,
    int N, int d, int M, int Ds, int K1D, int bits_per_dim,
    cudaStream_t stream = 0);

// Fine residual encode: d_y [N,d] + d_primary [N,M] → d_res_codes [N,bpv] + d_corrections [N]
void launch_fine_residual_encode(
    const float*   d_y,
    const uint8_t* d_primary,
    uint8_t*       d_res_codes,
    float*         d_corrections,
    const float*   d_c1d,
    const float*   d_res_c1d,
    int N, int d, int M, int Ds, int K1D, int Kr, int Br, int bpv, int bits_per_dim,
    cudaStream_t stream = 0);

} // namespace jhq_gpu
