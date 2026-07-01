#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace jhq_gpu {

// Primary encode: rotated vectors → M-byte codes.
// One thread per vector.
void launch_primary_encode(
    const float*   d_y,       // [N × d]  rotated vectors
    uint8_t*       d_codes,   // [N × M]  output primary codes
    const float*   d_c1d,     // [K1D]    1D Lloyd-Max codewords
    int N, int d, int M, int Ds, int K1D, int bits_per_dim,
    cudaStream_t stream = 0);

// Residual encode + correction: combine into one pass to avoid re-reading d_y.
// Inputs:  d_y (rotated), d_primary (already encoded)
// Outputs: d_res_codes (nibble-packed Br=4 or byte-packed Br=8), d_corrections (2<yhat,rhat>)
void launch_residual_encode(
    const float*   d_y,           // [N × d]
    const uint8_t* d_primary,     // [N × M]
    uint8_t*       d_res_codes,   // [N × bpv]
    float*         d_corrections, // [N]
    const float*   d_c1d,         // [K1D]
    const float*   d_res_c1d,     // [Kr]
    int N, int d, int M, int Ds, int K1D, int Kr, int Br, int bpv, int bits_per_dim,
    cudaStream_t stream = 0);

} // namespace jhq_gpu
