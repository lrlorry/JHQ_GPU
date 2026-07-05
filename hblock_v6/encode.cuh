#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace hblock_v6 {

void launch_assign_from_dots(const float* d_dots, const float* d_cent_norms,
                              int* d_assigns, int K, int nb, cudaStream_t s);

void launch_subtract_centroid(const float* d_in, const int* d_codes,
                               const float* d_cents,
                               float* d_out, int n, int d, cudaStream_t s);

void launch_fine_encode(const float* d_r2, const float* d_c1d,
                        uint8_t* d_out,
                        int n, int d, int Kr, int Br, int bpv,
                        cudaStream_t s);

} // namespace hblock_v6
