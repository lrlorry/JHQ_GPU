#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace hblock_v17 {

void launch_assign_from_dots(const float* d_dots, const float* d_cent_norms,
                              int* d_assigns, int K, int nb, cudaStream_t s);

void launch_subtract_centroid(const float* d_in, const int* d_codes,
                               const float* d_cents,
                               float* d_out, int n, int d, cudaStream_t s);

void launch_fine_encode(const float* d_r2, const float* d_c1d,
                        uint8_t* d_out,
                        int n, int d, int Kr, int Br, int bpv,
                        cudaStream_t s);

// k-means assignment: d_x_proj [n, d_proj], d_cents [K, d_proj] -> d_assigns [n]
// d_norms_cent [K] = ||cent||^2 (precomputed)
void launch_kmeans_assign(const float* d_dots, const float* d_cent_norms,
                          int* d_assigns, int K, int n, cudaStream_t s);

// k-means center update: accumulate sums; called after assign
void launch_kmeans_update(const float* d_x_proj, const int* d_assigns,
                          float* d_new_cents, int* d_counts,
                          int n, int d_proj, int K, cudaStream_t s);

} // namespace hblock_v17
