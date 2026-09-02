#include "jhq_v21_cascade/train_res_gpu.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace jhq_gpu {

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" + \
        std::to_string(__LINE__) + "  " + cudaGetErrorString(_e)); } } while (0)
#endif

namespace {

// Residual straight into segment-major [M][n*Ds]: subspace m owns one
// contiguous run, which is what the segmented sort and every later pass want.
// The host version builds [n][d] and then gathers per subspace.
__global__ void residual_segments_kernel(
    const float*   __restrict__ d_y,      // [n, d]
    const uint8_t* __restrict__ d_codes,  // [n, M]
    const float*   __restrict__ d_cent,   // [M][K][Ds]
    float*                      d_seg,    // [M][n*Ds]
    int n, int d, int M, int Ds, int K)
{
    const int m = blockIdx.y;
    const long long seg = (long long)n * Ds;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < seg; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / Ds;
        const int       j = (int)(t - i * Ds);
        const uint8_t   c = d_codes[i * M + m];
        const float yhat  = d_cent[((long long)m * K + c) * Ds + j];
        d_seg[(long long)m * seg + t] = d_y[i * d + (long long)m * Ds + j] - yhat;
    }
}

// c[i] = sorted[min((i + 0.5)/Kr * n, n-1)] -- the host's quantile placement.
__global__ void quantile_init_kernel(
    const float* __restrict__ d_sorted, float* d_c, long long seg, int M, int Kr)
{
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < (long long)M * Kr; t += (long long)gridDim.x * blockDim.x) {
        const int m = (int)(t / Kr), i = (int)(t - (long long)m * Kr);
        long long idx = (long long)(((float)i + 0.5f) / (float)Kr * (float)seg);
        if (idx > seg - 1) idx = seg - 1;
        d_c[t] = d_sorted[(long long)m * seg + idx];
    }
}

// Binary search on midpoints of adjacent centroids, exactly the host's loop.
//
// The bins are private to the block: 25 iterations over n*Ds values with two
// global atomics apiece is 3.8 billion of them at the usual settings, and the
// atomics, not the sort, are what the phase costs. Accumulating in shared and
// flushing once per block per bin divides the global traffic by roughly the
// block size. Kr <= 256 so the private copy is 2 KB of doubles plus 1 KB of
// counts, on top of the Kr floats of centroids.
__global__ void assign_accum_kernel(
    const float* __restrict__ d_seg, const float* __restrict__ d_c,
    double* d_sum, int* d_cnt, long long seg, int Kr)
{
    // doubles first so their alignment does not depend on Kr
    extern __shared__ char s_raw[];
    double* s_sum = (double*)s_raw;                // Kr
    int*    s_cnt = (int*)(s_sum + Kr);            // Kr
    float*  s_c   = (float*)(s_cnt + Kr);          // Kr

    const int m = blockIdx.y;
    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        s_c[i]   = d_c[(long long)m * Kr + i];
        s_sum[i] = 0.0;
        s_cnt[i] = 0;
    }
    __syncthreads();

    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < seg; t += (long long)gridDim.x * blockDim.x) {
        const float v = d_seg[(long long)m * seg + t];
        int lo = 0, hi = Kr - 1;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (v < 0.5f * (s_c[mid] + s_c[mid + 1])) hi = mid; else lo = mid + 1;
        }
        atomicAdd(s_sum + lo, (double)v);
        atomicAdd(s_cnt + lo, 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        if (s_cnt[i] == 0) continue;
        atomicAdd(d_sum + (long long)m * Kr + i, s_sum[i]);
        atomicAdd(d_cnt + (long long)m * Kr + i, s_cnt[i]);
    }
}

// Mean, or the quantile seed again when a cluster drew nothing; then the
// centroids are re-sorted so the next binary search stays valid. One block per
// subspace, Kr <= 256, so an insertion sort in shared is ample.
__global__ void update_sort_kernel(
    const double* __restrict__ d_sum, const int* __restrict__ d_cnt,
    const float*  __restrict__ d_sorted, float* d_c, long long seg, int Kr)
{
    extern __shared__ float s[];
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        const long long k = (long long)m * Kr + i;
        if (d_cnt[k] == 0) {
            long long idx = (long long)(((float)i + 0.5f) / (float)Kr * (float)seg);
            if (idx > seg - 1) idx = seg - 1;
            s[i] = d_sorted[(long long)m * seg + idx];
        } else {
            s[i] = (float)(d_sum[k] / (double)d_cnt[k]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int i = 1; i < Kr; ++i) {
            const float v = s[i];
            int j = i - 1;
            while (j >= 0 && s[j] > v) { s[j + 1] = s[j]; --j; }
            s[j + 1] = v;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < Kr; i += blockDim.x)
        d_c[(long long)m * Kr + i] = s[i];
}

} // namespace

void launch_residual_codebook(const float* d_y, const uint8_t* d_codes,
                              const float* d_cent, int n, int d, int M, int Ds,
                              int K, int Kr, int max_iter, float* d_res_c1d,
                              cudaStream_t stream)
{
    if (Kr > 256)
        throw std::runtime_error("launch_residual_codebook: Kr > 256 exceeds the "
                                 "single-block sort in update_sort_kernel");
    const long long seg   = (long long)n * Ds;
    const long long total = seg * M;

    float* d_seg    = nullptr;
    float* d_sorted = nullptr;
    double* d_sum   = nullptr;
    int*   d_cnt    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seg,    total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sorted, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum,    (size_t)M * Kr * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnt,    (size_t)M * Kr * sizeof(int)));

    const int BLOCK = 256;
    const int gx = (int)std::min((seg + BLOCK - 1) / BLOCK, 4096LL);
    residual_segments_kernel<<<dim3(gx, M), BLOCK, 0, stream>>>(
        d_y, d_codes, d_cent, d_seg, n, d, M, Ds, K);
    CUDA_CHECK(cudaGetLastError());

    // One segmented radix sort replaces M host sorts of n*Ds values each --
    // 96 sorts of 800k at the usual settings, which is where the 3.8 s went.
    std::vector<int> h_off(M + 1);
    for (int m = 0; m <= M; ++m) h_off[m] = (int)std::min((long long)m * seg, (long long)INT32_MAX);
    if ((long long)M * seg > INT32_MAX)
        throw std::runtime_error("launch_residual_codebook: M*n*Ds exceeds the 32-bit "
                                 "offsets cub's segmented sort takes; lower n_train");
    int* d_off = nullptr;
    CUDA_CHECK(cudaMalloc(&d_off, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_off, h_off.data(), (M + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    size_t tmp_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, tmp_bytes, d_seg, d_sorted, (int)total, M, d_off, d_off + 1, 0, 32, stream));
    void* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
        d_tmp, tmp_bytes, d_seg, d_sorted, (int)total, M, d_off, d_off + 1, 0, 32, stream));

    quantile_init_kernel<<<(M * Kr + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_sorted, d_res_c1d, seg, M, Kr);
    CUDA_CHECK(cudaGetLastError());

    for (int it = 0; it < max_iter; ++it) {
        CUDA_CHECK(cudaMemsetAsync(d_sum, 0, (size_t)M * Kr * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, (size_t)M * Kr * sizeof(int), stream));
        const size_t acc_smem = (size_t)Kr * (sizeof(float) + sizeof(double) + sizeof(int));
        assign_accum_kernel<<<dim3(gx, M), BLOCK, acc_smem, stream>>>(
            d_seg, d_res_c1d, d_sum, d_cnt, seg, Kr);
        CUDA_CHECK(cudaGetLastError());
        update_sort_kernel<<<M, BLOCK, Kr * sizeof(float), stream>>>(
            d_sum, d_cnt, d_sorted, d_res_c1d, seg, Kr);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_seg); cudaFree(d_sorted); cudaFree(d_sum); cudaFree(d_cnt);
    cudaFree(d_off); cudaFree(d_tmp);
}

} // namespace jhq_gpu
