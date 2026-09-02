#include "jhq_v21_cascade/train_pq_gpu.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace jhq_gpu {

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" + \
        std::to_string(__LINE__) + "  " + cudaGetErrorString(_e)); } } while (0)
#endif

namespace {

// One thread per (subspace, point). The subspace's K centroids are staged in
// shared memory: at K=256 and Ds=8 that is 8 KB, read K times per point, so
// the staging pays for itself many times over.
__global__ void pq_assign_kernel(
    const float* __restrict__ d_y,      // [n, d]
    const float* __restrict__ d_cent,   // [M][K][Ds]
    int*                      d_assign, // [M, n]
    int n, int d, int Ds, int K)
{
    extern __shared__ float s_cent[];   // K * Ds
    const int m = blockIdx.y;
    const float* cent_m = d_cent + (long long)m * K * Ds;
    for (int i = threadIdx.x; i < K * Ds; i += blockDim.x) s_cent[i] = cent_m[i];
    __syncthreads();

    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x) {
        const float* xi = d_y + i * d + (long long)m * Ds;
        float x[16];                       // Ds <= 16 covers every configuration here
        for (int j = 0; j < Ds; ++j) x[j] = xi[j];

        int   best = 0;
        float bd   = 3.402823466e+38f;
        for (int c = 0; c < K; ++c) {
            const float* cc = s_cent + (long long)c * Ds;
            float dd = 0.f;
            for (int j = 0; j < Ds; ++j) { float t = x[j] - cc[j]; dd += t * t; }
            // strict < so a tie takes the lowest index, as on the host
            if (dd < bd) { bd = dd; best = c; }
        }
        d_assign[(long long)m * n + i] = best;
    }
}

// Sums in double: the atomics land in whatever order the scheduler chooses, and
// float accumulation would make the trained codebook depend on that order.
__global__ void pq_accum_kernel(
    const float* __restrict__ d_y,
    const int*   __restrict__ d_assign,
    double*                   d_sums,    // [M][K][Ds]
    int*                      d_counts,  // [M][K]
    int n, int d, int Ds, int K)
{
    const int m = blockIdx.y;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x) {
        const int c = d_assign[(long long)m * n + i];
        const float* xi = d_y + i * d + (long long)m * Ds;
        double* s = d_sums + ((long long)m * K + c) * Ds;
        for (int j = 0; j < Ds; ++j) atomicAdd(s + j, (double)xi[j]);
        atomicAdd(d_counts + (long long)m * K + c, 1);
    }
}

// A centroid with no points keeps the position analytical_init gave it.
__global__ void pq_update_kernel(
    const double* __restrict__ d_sums,
    const int*    __restrict__ d_counts,
    float*                     d_cent,
    int M, int K, int Ds)
{
    long long total = (long long)M * K;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long long)gridDim.x * blockDim.x) {
        const int cnt = d_counts[idx];
        if (cnt == 0) continue;
        const double inv = 1.0 / (double)cnt;
        const double* s = d_sums + idx * Ds;
        float* cc = d_cent + idx * Ds;
        for (int j = 0; j < Ds; ++j) cc[j] = (float)(s[j] * inv);
    }
}

} // namespace

void launch_pq_kmeans(const float* d_y, int n, int d, int M, int Ds, int K,
                      float* d_cent, int iters, cudaStream_t stream)
{
    if (Ds > 16)
        throw std::runtime_error("launch_pq_kmeans: Ds > 16 exceeds the per-thread "
                                 "register buffer; raise it or fall back to the host path");

    int*    d_assign = nullptr;
    double* d_sums   = nullptr;
    int*    d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_assign, (size_t)M * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums,   (size_t)M * K * Ds * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)M * K * sizeof(int)));

    const int BLOCK   = 256;
    const int gx      = (int)std::min((long long)((n + BLOCK - 1) / BLOCK), 4096LL);
    const size_t smem = (size_t)K * Ds * sizeof(float);
    dim3 grid(gx, M);

    for (int it = 0; it < iters; ++it) {
        pq_assign_kernel<<<grid, BLOCK, smem, stream>>>(
            d_y, d_cent, d_assign, n, d, Ds, K);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(d_sums, 0, (size_t)M * K * Ds * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_counts, 0, (size_t)M * K * sizeof(int), stream));
        pq_accum_kernel<<<grid, BLOCK, 0, stream>>>(
            d_y, d_assign, d_sums, d_counts, n, d, Ds, K);
        CUDA_CHECK(cudaGetLastError());

        const int upd_grid = (int)(((long long)M * K + BLOCK - 1) / BLOCK);
        pq_update_kernel<<<upd_grid, BLOCK, 0, stream>>>(
            d_sums, d_counts, d_cent, M, K, Ds);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_assign); cudaFree(d_sums); cudaFree(d_counts);
}

} // namespace jhq_gpu
