#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace hblock {

// ── GPU kernels ───────────────────────────────────────────────────────────────
// Assign via pre-computed dot products: argmin { ||c||² - 2·dot(q,c) }
//   dots [nb, K], cent_norms [K] → assigns [nb]
void launch_assign_from_dots(const float* d_dots, const float* d_cent_norms,
                              int* d_assigns, int K, int nb, cudaStream_t s);

// Subtract routing centroid from each vector:
//   out[i] = in[i] - cents[codes[i]]
void launch_subtract_centroid(const float* d_in, const int* d_codes,
                               const float* d_cents,
                               float* d_out, int n, int d, cudaStream_t s);

// Encode fine residual → bpv-byte packed code per vector (nibble packing, Br=4).
//   d_r2   [n, d]: fine residual
//   d_c1d  [Kr]:   sorted scalar fine centroids
//   d_out  [n, bpv]: output bit-packed codes
void launch_fine_encode(const float* d_r2, const float* d_c1d,
                        uint8_t* d_out,
                        int n, int d, int Kr, int Br, int bpv,
                        cudaStream_t s);

} // namespace hblock
