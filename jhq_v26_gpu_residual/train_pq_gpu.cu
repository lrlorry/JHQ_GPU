#include "jhq_v26_gpu_residual/train_pq_gpu.cuh"

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
        float x[32];                       // Ds = d/M is 32 at d=3072, M=96
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
    if (Ds > 32)
        throw std::runtime_error("launch_pq_kmeans: Ds = " + std::to_string(Ds) +
                                 " exceeds the 32-float per-thread buffer; raise it or "
                                 "use the host path");

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

namespace jhq_gpu {
namespace {

// One warp per vector: the lanes read consecutive floats and add to
// consecutive doubles, so each step is one coalesced read and one warp-wide
// atomic to a 256-byte run. The thread-per-vector version had the lanes d
// floats apart on both sides, every element its own transaction.
__global__ void ivf_accum_kernel(
    const float* __restrict__ d_y, const int* __restrict__ d_assign,
    double* d_sums, int* d_counts, int n, int d)
{
    const int lane = threadIdx.x & 31;
    const long long nwarps = (long long)gridDim.x * (blockDim.x >> 5);
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
         i < n; i += nwarps) {
        const int c = d_assign[i];
        const float* yi = d_y + i * d;
        double* s = d_sums + (long long)c * d;
        for (int j = lane; j < d; j += 32) atomicAdd(s + j, (double)yi[j]);
        if (lane == 0) atomicAdd(d_counts + c, 1);
    }
}

// A list that drew nothing is reseeded from a training vector, the same index
// the host picks so the two paths diverge only through float accumulation.
__global__ void ivf_update_kernel(
    const double* __restrict__ d_sums, const int* __restrict__ d_counts,
    const float* __restrict__ d_y, float* d_cent, int n, int d, int nlist, int iter)
{
    for (long long c = (long long)blockIdx.x; c < nlist; c += gridDim.x) {
        const int cnt = d_counts[c];
        float* cc = d_cent + c * d;
        if (cnt == 0) {
            const int src = (int)(((long long)c * 1103515245LL + (long long)iter * 12345LL) % n);
            for (int j = threadIdx.x; j < d; j += blockDim.x) cc[j] = d_y[(long long)src * d + j];
        } else {
            const double inv = 1.0 / (double)cnt;
            const double* s = d_sums + c * d;
            for (int j = threadIdx.x; j < d; j += blockDim.x) cc[j] = (float)(s[j] * inv);
        }
    }
}

} // namespace

void launch_ivf_accumulate(const float* d_y, const int* d_assign, float* d_cent,
                           int n, int d, int nlist, int iter, cudaStream_t stream)
{
    double* d_sums = nullptr; int* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums, (size_t)nlist * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)nlist * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_sums, 0, (size_t)nlist * d * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(d_counts, 0, (size_t)nlist * sizeof(int), stream));

    const int BLOCK = 256;
    const int gx = (int)std::min(((long long)n * 32 + BLOCK - 1) / BLOCK, 8192LL);
    ivf_accum_kernel<<<gx, BLOCK, 0, stream>>>(d_y, d_assign, d_sums, d_counts, n, d);
    CUDA_CHECK(cudaGetLastError());
    ivf_update_kernel<<<std::min(nlist, 4096), BLOCK, 0, stream>>>(
        d_sums, d_counts, d_y, d_cent, n, d, nlist, iter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_sums); cudaFree(d_counts);
}

} // namespace jhq_gpu

namespace jhq_gpu {
namespace {

// One block per (subspace, dimension); the grid is M*Ds blocks, each reducing
// its own column of the training set. Sums in double so the result does not
// depend on the order the blocks' partial sums happen to land in.
__global__ void subspace_moments_kernel(
    const float* __restrict__ d_y, double* d_sum, double* d_sqsum,
    int n, int d, int Ds)
{
    const int col = blockIdx.x;              // (m * Ds + j)
    const int m   = col / Ds, j = col - m * Ds;
    const long long off = (long long)m * Ds + j;

    __shared__ double s1[256], s2[256];
    double a = 0.0, b = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = d_y[i * d + off];
        a += v; b += v * v;
    }
    s1[threadIdx.x] = a; s2[threadIdx.x] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { s1[threadIdx.x] += s1[threadIdx.x + s];
                               s2[threadIdx.x] += s2[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { d_sum[col] = s1[0]; d_sqsum[col] = s2[0]; }
}

__global__ void mean_finish_kernel(
    const double* __restrict__ d_sum, float* d_mean, int cols, int n)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += gridDim.x * blockDim.x)
        d_mean[c] = (float)(d_sum[c] / (double)n);
}

// Second pass, over deviations from the mean.
//
// The one-pass form -- (sum(x^2) - n*mean^2)/(n-1) -- is algebraically the same
// and reads the data once, but the two terms are close and cancel, and the
// variance it returns is off by enough to move the analytical radii: recall
// went 0.9454 -> 0.9444 on Vogue with it. The host computes deviations
// directly, so this does too.
__global__ void variance_kernel(
    const float* __restrict__ d_y, const float* __restrict__ d_mean,
    double* d_ss, int n, int d, int Ds)
{
    const int col = blockIdx.x;
    const int m   = col / Ds, j = col - m * Ds;
    const long long off = (long long)m * Ds + j;
    const double mu = (double)d_mean[col];

    __shared__ double s[256];
    double a = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        const double t = (double)d_y[i * d + off] - mu;
        a += t * t;
    }
    s[threadIdx.x] = a;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_ss[col] = s[0];
}

__global__ void var_finish_kernel(
    const double* __restrict__ d_ss, float* d_var, int cols, int n)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += gridDim.x * blockDim.x)
        d_var[c] = n > 1 ? (float)(d_ss[c] / (double)(n - 1)) : 0.f;
}

} // namespace

void launch_subspace_stats(const float* d_y, int n, int d, int M, int Ds,
                           float* d_mean, float* d_var, cudaStream_t stream)
{
    const int cols = M * Ds;
    double *d_sum = nullptr, *d_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, (size_t)cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sq,  (size_t)cols * sizeof(double)));
    subspace_moments_kernel<<<cols, 256, 0, stream>>>(d_y, d_sum, d_sq, n, d, Ds);
    CUDA_CHECK(cudaGetLastError());
    mean_finish_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(d_sum, d_mean, cols, n);
    CUDA_CHECK(cudaGetLastError());
    variance_kernel<<<cols, 256, 0, stream>>>(d_y, d_mean, d_sq, n, d, Ds);
    CUDA_CHECK(cudaGetLastError());
    var_finish_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(d_sq, d_var, cols, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_sum); cudaFree(d_sq);
}

} // namespace jhq_gpu
