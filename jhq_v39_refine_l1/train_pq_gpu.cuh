#pragma once
#include <cuda_runtime.h>

namespace jhq_gpu {

// Lloyd iterations for the primary product quantiser, on the device.
//
// The analytical initialisation stays on the host: it is the paper's Gaussian
// quantile placement, it costs microseconds, and keeping it there means the
// starting point is identical to the reference. Only the iterations move.
//
// Matches cpu/pq_codebook.cpp exactly on the two points where a k-means can
// legitimately differ: a tie in the assignment goes to the lowest centroid
// index, and a centroid that draws no points keeps its analytical position
// rather than being reseeded. Accumulation is in double so that the order the
// atomics happen to run in does not change the float result.
//
// d_y is the rotated training set, [n, d] row-major -- subspace m of point i is
// the Ds-wide slice at y[i*d + m*Ds], so no gather is needed.
void launch_pq_kmeans(const float* d_y, int n, int d, int M, int Ds, int K,
                      float* d_cent,          // [M][K][Ds], analytically seeded
                      int iters, cudaStream_t stream = 0);

} // namespace jhq_gpu

namespace jhq_gpu {

// IVF centroid refinement, on the device.
//
// The host loop reads the assignment back and accumulates on one core, once per
// iteration: eight round trips and eight single-threaded passes over the
// training set, 397 ms of the build. Only the assignment was already on the
// device.
//
// Matches the host rule for a list that draws nothing: it is reseeded from the
// training vector at (c*1103515245 + iter*12345) % n, not left where it was.
void launch_ivf_accumulate(const float* d_y,      // [n, d] rotated training set
                           const int*   d_assign, // [n]
                           float*       d_cent,   // [nlist][d], updated in place
                           int n, int d, int nlist, int iter,
                           cudaStream_t stream = 0);

} // namespace jhq_gpu

namespace jhq_gpu {

// Per-subspace first and second moments of the rotated training set, [M][Ds]
// each. These are the only things the analytical initialisation reads from the
// data, so computing them here replaces a copy of the whole rotated training
// set to the host -- 2.9 GB on Vogue -- with M*Ds*2 floats, about 6 KB.
void launch_subspace_stats(const float* d_y, int n, int d, int M, int Ds,
                           float* d_mean, float* d_var, cudaStream_t stream = 0);

} // namespace jhq_gpu
