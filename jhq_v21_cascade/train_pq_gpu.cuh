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
