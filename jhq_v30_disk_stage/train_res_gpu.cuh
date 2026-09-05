#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace jhq_gpu {

// The residual level's scalar codebooks, trained on the device.
//
// This is the largest phase of the index build: 3.8 s of 8.3 s even with all
// 208 cores on it, and almost all of that is a sort -- one per subspace over
// n_train*Ds values, 96 sorts of 800k at the usual settings.
//
// Two things fall out of doing it here. The residual never needs the gather the
// host version does: it is written straight into [M][n*Ds] segment-major
// layout, which is what a segmented sort wants. And the rotated training set
// stays on the device, so the 2.9 GB copy back to the host that exists only to
// feed host-side training can go.
//
// Matches cpu/codebook.cpp: quantile initialisation from the sorted values, the
// binary search on midpoints of adjacent centroids, an empty cluster falling
// back to its quantile seed, and the centroids re-sorted each iteration to keep
// that search valid. Sums accumulate in double so the atomics' order does not
// move the result. The host loop stops early once nothing moves; this runs the
// full count, which reaches the same fixed point.
void launch_residual_codebook(const float*   d_y,       // [n, d] rotated training set
                              const uint8_t* d_codes,   // [n, M] primary codes
                              const float*   d_cent,    // [M][K][Ds]
                              int n, int d, int M, int Ds, int K,
                              int Kr, int max_iter,
                              float* d_res_c1d,         // [M][Kr] out
                              cudaStream_t stream = 0);

} // namespace jhq_gpu

namespace jhq_gpu {

// The residual codebooks again, but from a histogram instead of a sort.
//
// The sorted values were only ever wanted for two things: the quantiles that
// seed the codebook, and the boundary lookups each iteration. A histogram of
// the residuals answers both -- quantiles are an inverse lookup on the
// cumulative counts, and a bin's count and sum come from adding the buckets it
// covers. That drops the O(n log n) sort and the prefix scan, leaving one O(n)
// pass, and every iteration afterwards touches only the buckets:
//
//     sort + scan + 25 boundary searches   ->   one histogram + 25 bucket sweeps
//
// It is an approximation where the sort was exact: a value is placed by its
// bucket rather than its own magnitude, so the centroids move by up to half a
// bucket width. With 2^16 buckets over the observed range that is far below
// what Br=4 or 8 resolves, but it is a real difference and the recall it costs
// is measured rather than assumed.
// The estimator on residuals the caller already computed: same sort, same
// prefix sum, same O(Kr log n) iterations, without needing the rotated set
// resident. d_values is [M_chunk][seg], segment-major.
void launch_residual_codebook_from_values(const float* d_values, long long seg,
                                          int M_chunk, int Kr, int max_iter,
                                          float* d_res_c1d,
                                          cudaStream_t stream = 0);

void launch_residual_codebook_hist(const float* d_y, const uint8_t* d_codes,
                                   const float* d_cent,
                                   int n, int d, int M, int Ds, int K,
                                   int Kr, int max_iter, int nbuckets,
                                   float* d_res_c1d, cudaStream_t stream = 0);

} // namespace jhq_gpu
