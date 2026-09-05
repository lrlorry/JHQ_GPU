#include "jhq_v30_disk_stage/train_res_gpu.cuh"

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
    float*                      d_seg,    // [mc][n*Ds], subspaces m0 .. m0+mc
    int n, int d, int M, int Ds, int K, int m0 = 0)
{
    // Local index into the scratch, global index into the codes and centroids:
    // the scratch holds one chunk of subspaces, the index holds all of them.
    const int ml = blockIdx.y;
    const int m  = m0 + ml;
    const long long seg = (long long)n * Ds;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < seg; t += (long long)gridDim.x * blockDim.x) {
        const long long i = t / Ds;
        const int       j = (int)(t - i * Ds);
        const uint8_t   c = d_codes[i * M + m];
        const float yhat  = d_cent[((long long)m * K + c) * Ds + j];
        d_seg[(long long)ml * seg + t] = d_y[i * d + (long long)m * Ds + j] - yhat;
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

// Assignment straight off the sorted values.
//
// The host walks every value and binary-searches its bin, which is O(n log Kr)
// an iteration. But the values are already sorted here -- the quantile
// initialisation needed that -- and a sorted sequence makes each bin one
// contiguous run, so the whole assignment is just the Kr-1 boundaries between
// them. With an inclusive prefix sum of the sorted values taken once, a bin's
// count is a difference of indices and its sum a difference of prefixes:
//
//     O(n log Kr) per iteration   ->   O(Kr log n) per iteration
//     800k * 8 = 6.4M compares    ->   256 * 20 = 5120
//
// The result is identical: sums and counts do not depend on the order the
// values are visited. The kernel below is what that replaces, kept because the
// boundary form needs the prefix sum and the prefix sum needs the sort.
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

    // Bitonic sort across the block. The insertion sort this replaces ran on
    // thread 0 alone: 128 comparisons at Kr=16, but 32768 at Kr=256, times the
    // iteration count, with the other 255 threads idle. Bitonic is log^2(Kr)
    // stages with every thread comparing one pair, so 256 keys take 36 steps
    // instead of 32768. Kr is a power of two, which is what lets the padding be
    // skipped.
    for (int k = 2; k <= Kr; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
                const int p = i ^ j;
                if (p > i) {
                    const bool up = ((i & k) == 0);
                    if ((s[i] > s[p]) == up) {
                        const float t = s[i]; s[i] = s[p]; s[p] = t;
                    }
                }
            }
            __syncthreads();
        }
    for (int i = threadIdx.x; i < Kr; i += blockDim.x)
        d_c[(long long)m * Kr + i] = s[i];
}


// One thread per bin: find where its lower boundary falls in the sorted values,
// then read count and sum off the index and prefix differences.
__global__ void assign_from_sorted_kernel(
    const float*  __restrict__ d_sorted,   // [M][seg], ascending
    const double* __restrict__ d_prefix,   // [M][seg], inclusive sum of d_sorted
    const float*  __restrict__ d_c,        // [M][Kr], ascending
    double* d_sum, int* d_cnt, long long seg, int Kr)
{
    const int m = blockIdx.x;
    const float*  seg_v = d_sorted + (long long)m * seg;
    const double* seg_p = d_prefix + (long long)m * seg;

    extern __shared__ long long s_bound[];   // Kr + 1 boundary indices

    for (int i = threadIdx.x; i <= Kr; i += blockDim.x) {
        if (i == 0)      { s_bound[0] = 0;   continue; }
        if (i == Kr)     { s_bound[Kr] = seg; continue; }
        // the host sends v to bin i when v >= midpoint(c[i-1], c[i])
        const float mid = 0.5f * (d_c[(long long)m * Kr + i - 1] +
                                  d_c[(long long)m * Kr + i]);
        long long lo = 0, hi = seg;
        while (lo < hi) {                       // first index with value >= mid
            const long long md = (lo + hi) >> 1;
            if (seg_v[md] < mid) lo = md + 1; else hi = md;
        }
        s_bound[i] = lo;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        const long long a = s_bound[i], b = s_bound[i + 1];
        const long long k = (long long)m * Kr + i;
        d_cnt[k] = (int)(b - a);
        d_sum[k] = (b > a) ? (seg_p[b - 1] - (a > 0 ? seg_p[a - 1] : 0.0)) : 0.0;
    }
}

// Inclusive prefix sum of the sorted values, per segment, taken once. Done as
// one scan over the whole array with the segment's own base subtracted, since
// the segments are contiguous and equal in length.
__global__ void gather_segment_bases_kernel(
    const double* __restrict__ d_prefix, double* d_base, long long seg, int M)
{
    for (int m = blockIdx.x * blockDim.x + threadIdx.x; m < M;
         m += gridDim.x * blockDim.x)
        d_base[m] = (m == 0) ? 0.0 : d_prefix[(long long)m * seg - 1];
}

// The bases have to be read before any of them is overwritten: subtracting in
// place while reading segment m-1's last entry is a race, and it cost 0.9444 ->
// 0.9309 on Vogue before it was caught.
__global__ void subtract_segment_base_kernel(
    double* d_prefix, const double* __restrict__ d_base, long long seg, int M)
{
    const int m = blockIdx.y;
    const double base = d_base[m];
    if (base == 0.0 && m == 0) return;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < seg; i += (long long)gridDim.x * blockDim.x)
        d_prefix[(long long)m * seg + i] -= base;
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
    const long long seg = (long long)n * Ds;

    // Equation 5 collects the residual of every vector in every dimension, so
    // the scratch is n*d values three times over -- the segments, their sorted
    // copy, and a double prefix sum -- which is 33 GB at n=2M, d=1024 and does
    // not fit. Nothing here couples one subspace to another, though: each is an
    // independent 1-D k-means. So the subspaces are done in chunks, and the
    // scratch is sized by the chunk rather than by d. That also keeps the
    // segmented sort inside the 32-bit offsets cub takes, which n*d alone
    // exceeds past 2.1M vectors at 1024 dimensions.
    const size_t per_sub = (size_t)seg * (sizeof(float) * 2 + sizeof(double));
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    int mc = (int)std::min<size_t>((size_t)M,
                 std::max<size_t>(1, (size_t)(free_b * 0.55) / std::max<size_t>(per_sub, 1)));
    while (mc > 1 && (long long)mc * seg > (long long)INT32_MAX) mc /= 2;
    if ((long long)seg > (long long)INT32_MAX)
        throw std::runtime_error("launch_residual_codebook: n*Ds exceeds the 32-bit "
                                 "offsets cub's segmented sort takes; lower n_train");

    float*  d_seg    = nullptr;
    float*  d_sorted = nullptr;
    double* d_prefix = nullptr;
    double* d_sum    = nullptr;
    int*    d_cnt    = nullptr;
    int*    d_off    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seg,    (size_t)mc * seg * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sorted, (size_t)mc * seg * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prefix, (size_t)mc * seg * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum,    (size_t)mc * Kr * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnt,    (size_t)mc * Kr * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_off,    (size_t)(mc + 1) * sizeof(int)));

    const int BLOCK = 256;
    const int gx = (int)std::min((seg + BLOCK - 1) / BLOCK, 4096LL);

    // cub's temporaries are sized for the largest chunk and reused by the rest.
    size_t sort_bytes = 0, scan_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, sort_bytes, d_seg, d_sorted, (int)((long long)mc * seg), mc,
        d_off, d_off + 1, 0, 32, stream));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_bytes, d_sorted, d_prefix, (int)((long long)mc * seg), stream));
    void* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, std::max(sort_bytes, scan_bytes)));
    size_t tmp_bytes = std::max(sort_bytes, scan_bytes);

    for (int m0 = 0; m0 < M; m0 += mc) {
        const int mb = std::min(mc, M - m0);
        const long long tot = (long long)mb * seg;

        std::vector<int> h_off(mb + 1);
        for (int m = 0; m <= mb; ++m) h_off[m] = (int)((long long)m * seg);
        CUDA_CHECK(cudaMemcpyAsync(d_off, h_off.data(), (size_t)(mb + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));

        residual_segments_kernel<<<dim3(gx, mb), BLOCK, 0, stream>>>(
            d_y, d_codes, d_cent, d_seg, n, d, M, Ds, K, m0);
        CUDA_CHECK(cudaGetLastError());

        size_t sb = tmp_bytes;
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
            d_tmp, sb, d_seg, d_sorted, (int)tot, mb, d_off, d_off + 1, 0, 32, stream));

        float* c_out = d_res_c1d + (long long)m0 * Kr;
        quantile_init_kernel<<<(mb * Kr + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
            d_sorted, c_out, seg, mb, Kr);
        CUDA_CHECK(cudaGetLastError());

        sb = tmp_bytes;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            d_tmp, sb, d_sorted, d_prefix, (int)tot, stream));
        {
            double* d_base = nullptr;
            CUDA_CHECK(cudaMalloc(&d_base, (size_t)mb * sizeof(double)));
            gather_segment_bases_kernel<<<(mb + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
                d_prefix, d_base, seg, mb);
            CUDA_CHECK(cudaGetLastError());
            subtract_segment_base_kernel<<<dim3(gx, mb), BLOCK, 0, stream>>>(
                d_prefix, d_base, seg, mb);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaFree(d_base);
        }

        for (int it = 0; it < max_iter; ++it) {
            assign_from_sorted_kernel<<<mb, BLOCK, (size_t)(Kr + 1) * sizeof(long long), stream>>>(
                d_sorted, d_prefix, c_out, d_sum, d_cnt, seg, Kr);
            CUDA_CHECK(cudaGetLastError());
            update_sort_kernel<<<mb, BLOCK, Kr * sizeof(float), stream>>>(
                d_sum, d_cnt, d_sorted, c_out, seg, Kr);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    cudaFree(d_seg); cudaFree(d_sorted); cudaFree(d_prefix);
    cudaFree(d_sum); cudaFree(d_cnt); cudaFree(d_off); cudaFree(d_tmp);
}

// The same estimator, on residuals the caller already has.
//
// launch_residual_codebook derives its segments from the rotated set, which has
// to be resident for it -- 38.5 GB at bge-m3 and 67.8 at stella. Everything
// after residual_segments_kernel touches only d_seg, so a caller that computed
// the residuals itself, a batch at a time, can hand them straight in. The sort,
// the prefix sum and the O(Kr log n) Lloyd iterations are unchanged, and so is
// the result.
//
// d_values is [M_chunk][seg], segment-major, which is the layout the segmented
// sort wants.
void launch_residual_codebook_from_values(const float* d_values, long long seg,
                                          int M_chunk, int Kr, int max_iter,
                                          float* d_res_c1d, cudaStream_t stream)
{
    if (Kr > 256)
        throw std::runtime_error("launch_residual_codebook_from_values: Kr > 256 "
                                 "exceeds the single-block sort in update_sort_kernel");
    const long long tot = (long long)M_chunk * seg;
    if (tot > 2147483647LL)
        throw std::runtime_error("launch_residual_codebook_from_values: M_chunk*seg "
                                 "exceeds the 32-bit offsets cub's segmented sort "
                                 "takes; lower JHQ_RES_CHUNK");

    float*  d_sorted = nullptr;
    double* d_prefix = nullptr;
    double* d_sum    = nullptr;
    int*    d_cnt    = nullptr;
    int*    d_off    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sorted, (size_t)tot * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prefix, (size_t)tot * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum,    (size_t)M_chunk * Kr * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnt,    (size_t)M_chunk * Kr * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_off,    (size_t)(M_chunk + 1) * sizeof(int)));

    const int BLOCK = 256;
    const int gx = (int)std::min((seg + BLOCK - 1) / BLOCK, 4096LL);
    std::vector<int> h_off(M_chunk + 1);
    for (int m = 0; m <= M_chunk; ++m) h_off[m] = (int)((long long)m * seg);
    CUDA_CHECK(cudaMemcpyAsync(d_off, h_off.data(), (size_t)(M_chunk + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    size_t sort_bytes = 0, scan_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, sort_bytes, d_values, d_sorted, (int)tot, M_chunk,
        d_off, d_off + 1, 0, 32, stream));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_bytes, d_sorted, d_prefix, (int)tot, stream));
    void* d_tmp = nullptr;
    size_t tmp_bytes = std::max(sort_bytes, scan_bytes);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));

    size_t sb = tmp_bytes;
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortKeys(
        d_tmp, sb, d_values, d_sorted, (int)tot, M_chunk, d_off, d_off + 1, 0, 32, stream));

    quantile_init_kernel<<<(M_chunk * Kr + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_sorted, d_res_c1d, seg, M_chunk, Kr);
    CUDA_CHECK(cudaGetLastError());

    sb = tmp_bytes;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        d_tmp, sb, d_sorted, d_prefix, (int)tot, stream));
    {
        double* d_base = nullptr;
        CUDA_CHECK(cudaMalloc(&d_base, (size_t)M_chunk * sizeof(double)));
        gather_segment_bases_kernel<<<(M_chunk + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
            d_prefix, d_base, seg, M_chunk);
        subtract_segment_base_kernel<<<dim3(gx, M_chunk), BLOCK, 0, stream>>>(
            d_prefix, d_base, seg, M_chunk);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_base);
    }

    for (int it = 0; it < max_iter; ++it) {
        assign_from_sorted_kernel<<<M_chunk, BLOCK,
            (size_t)(Kr + 1) * sizeof(long long), stream>>>(
            d_sorted, d_prefix, d_res_c1d, d_sum, d_cnt, seg, Kr);
        update_sort_kernel<<<M_chunk, BLOCK, Kr * sizeof(float), stream>>>(
            d_sum, d_cnt, d_sorted, d_res_c1d, seg, Kr);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_sorted); cudaFree(d_prefix);
    cudaFree(d_sum); cudaFree(d_cnt); cudaFree(d_off); cudaFree(d_tmp);
}

} // namespace jhq_gpu

namespace jhq_gpu {
namespace {

__global__ void seg_minmax_kernel(const float* __restrict__ d_seg,
                                  float* d_lo, float* d_hi, long long seg)
{
    const int m = blockIdx.x;
    __shared__ float s_lo[256], s_hi[256];
    float lo = 3.402823466e+38f, hi = -3.402823466e+38f;
    for (long long i = threadIdx.x; i < seg; i += blockDim.x) {
        const float v = d_seg[(long long)m * seg + i];
        lo = fminf(lo, v); hi = fmaxf(hi, v);
    }
    s_lo[threadIdx.x] = lo; s_hi[threadIdx.x] = hi;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            s_lo[threadIdx.x] = fminf(s_lo[threadIdx.x], s_lo[threadIdx.x + k]);
            s_hi[threadIdx.x] = fmaxf(s_hi[threadIdx.x], s_hi[threadIdx.x + k]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { d_lo[m] = s_lo[0]; d_hi[m] = s_hi[0]; }
}

// Counts and sums per bucket, privatised in shared so the global atomics are
// one per bucket per block rather than one per value.
__global__ void histogram_kernel(const float* __restrict__ d_seg,
                                 const float* __restrict__ d_lo,
                                 const float* __restrict__ d_hi,
                                 int* d_cnt, double* d_sum,
                                 long long seg, int nb)
{
    extern __shared__ char h_raw[];
    int*    s_cnt = (int*)h_raw;
    double* s_sum = (double*)(s_cnt + nb);

    const int m = blockIdx.y;
    const float lo = d_lo[m], hi = d_hi[m];
    const float inv = (hi > lo) ? (float)nb / (hi - lo) : 0.f;

    for (int i = threadIdx.x; i < nb; i += blockDim.x) { s_cnt[i] = 0; s_sum[i] = 0.0; }
    __syncthreads();

    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < seg; t += (long long)gridDim.x * blockDim.x) {
        const float v = d_seg[(long long)m * seg + t];
        int b = (int)((v - lo) * inv);
        b = b < 0 ? 0 : (b >= nb ? nb - 1 : b);
        atomicAdd(s_cnt + b, 1);
        atomicAdd(s_sum + b, (double)v);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
        if (s_cnt[i]) {
            atomicAdd(d_cnt + (long long)m * nb + i, s_cnt[i]);
            atomicAdd(d_sum + (long long)m * nb + i, s_sum[i]);
        }
    }
}

// One block per subspace: prefix the buckets once, seed by quantile, then run
// the Lloyd iterations entirely on the buckets.
__global__ void hist_lloyd_kernel(const int* __restrict__ d_cnt,
                                  const double* __restrict__ d_sum,
                                  const float* __restrict__ d_lo,
                                  const float* __restrict__ d_hi,
                                  float* d_c, int nb, int Kr, int max_iter)
{
    extern __shared__ char l_raw[];
    long long* s_ccnt = (long long*)l_raw;          // nb + 1 cumulative counts
    double*    s_csum = (double*)(s_ccnt + nb + 1); // nb + 1 cumulative sums
    float*     s_c    = (float*)(s_csum + nb + 1);  // Kr centroids

    const int m = blockIdx.x;
    const float lo = d_lo[m], hi = d_hi[m];
    const float w  = (hi - lo) / (float)nb;

    // Serial prefix on one thread: nb is 4096 here, and it runs once.
    if (threadIdx.x == 0) {
        s_ccnt[0] = 0; s_csum[0] = 0.0;
        for (int i = 0; i < nb; ++i) {
            s_ccnt[i + 1] = s_ccnt[i] + d_cnt[(long long)m * nb + i];
            s_csum[i + 1] = s_csum[i] + d_sum[(long long)m * nb + i];
        }
    }
    __syncthreads();
    const long long total = s_ccnt[nb];

    // Quantile seed, the same (i - 0.5)/Kr the sorted path uses.
    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        const long long target = (long long)(((double)i + 0.5) / (double)Kr * (double)total);
        int a = 0, b = nb;
        while (a < b) { const int md = (a + b) >> 1;
                        if (s_ccnt[md + 1] <= target) a = md + 1; else b = md; }
        s_c[i] = lo + ((float)a + 0.5f) * w;
    }
    __syncthreads();

    for (int it = 0; it < max_iter; ++it) {
        // Each centroid owns the buckets between its neighbours' midpoints.
        for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
            const float bl = (i == 0)      ? lo : 0.5f * (s_c[i - 1] + s_c[i]);
            const float br = (i == Kr - 1) ? hi : 0.5f * (s_c[i] + s_c[i + 1]);
            int a = (int)((bl - lo) / w); a = a < 0 ? 0 : (a > nb ? nb : a);
            int b = (int)((br - lo) / w); b = b < 0 ? 0 : (b > nb ? nb : b);
            if (b < a) b = a;
            const long long c = s_ccnt[b] - s_ccnt[a];
            s_c[i] = c ? (float)((s_csum[b] - s_csum[a]) / (double)c)
                       : lo + ((float)a + 0.5f) * w;
        }
        __syncthreads();
        // Keep them ascending so the next midpoint split stays valid.
        for (int k = 2; k <= Kr; k <<= 1)
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
                    const int p = i ^ j;
                    if (p > i) {
                        const bool up = ((i & k) == 0);
                        if ((s_c[i] > s_c[p]) == up) {
                            const float t = s_c[i]; s_c[i] = s_c[p]; s_c[p] = t;
                        }
                    }
                }
                __syncthreads();
            }
    }
    for (int i = threadIdx.x; i < Kr; i += blockDim.x)
        d_c[(long long)m * Kr + i] = s_c[i];
}

} // namespace

void launch_residual_codebook_hist(const float* d_y, const uint8_t* d_codes,
                                   const float* d_cent, int n, int d, int M,
                                   int Ds, int K, int Kr, int max_iter,
                                   int nbuckets, float* d_res_c1d,
                                   cudaStream_t stream)
{
    const long long seg = (long long)n * Ds;
    float*  d_seg = nullptr; float* d_lo = nullptr; float* d_hi = nullptr;
    int*    d_cnt = nullptr; double* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seg, (size_t)seg * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lo, (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hi, (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt, (size_t)M * nbuckets * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, (size_t)M * nbuckets * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(d_cnt, 0, (size_t)M * nbuckets * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, (size_t)M * nbuckets * sizeof(double), stream));

    const int BLOCK = 256;
    const int gx = (int)std::min((seg + BLOCK - 1) / BLOCK, 4096LL);
    residual_segments_kernel<<<dim3(gx, M), BLOCK, 0, stream>>>(
        d_y, d_codes, d_cent, d_seg, n, d, M, Ds, K);
    CUDA_CHECK(cudaGetLastError());
    seg_minmax_kernel<<<M, BLOCK, 0, stream>>>(d_seg, d_lo, d_hi, seg);
    CUDA_CHECK(cudaGetLastError());
    // Both ask for more than the 48 KB a kernel gets unasked: the histogram
    // wants nbuckets*(4+8) bytes and the Lloyd pass (nbuckets+1)*16 plus the
    // centroids. Opting in reaches ~227 KB, so 8192 buckets (96 and 131 KB)
    // fit where 2048 was chosen for the 48 KB default. The bucket count is
    // what bounds the error -- at most Kr-1 buckets straddle a cell boundary
    // and each carries 1/nbuckets of the mass -- so it is worth spending.
    const size_t hist_smem  = (size_t)nbuckets * (sizeof(int) + sizeof(double));
    const size_t lloyd_smem = (size_t)(nbuckets + 1) * (sizeof(long long) + sizeof(double))
                            + (size_t)Kr * sizeof(float);
    int optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (hist_smem > (size_t)optin || lloyd_smem > (size_t)optin)
        throw std::runtime_error("launch_residual_codebook_hist: " +
            std::to_string(nbuckets) + " buckets need more shared memory than the "
            "device allows; lower nbuckets");
    if (hist_smem > 48u * 1024u)
        CUDA_CHECK(cudaFuncSetAttribute(histogram_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)hist_smem));
    if (lloyd_smem > 48u * 1024u)
        CUDA_CHECK(cudaFuncSetAttribute(hist_lloyd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)lloyd_smem));

    histogram_kernel<<<dim3(gx, M), BLOCK, hist_smem, stream>>>(
        d_seg, d_lo, d_hi, d_cnt, d_sum, seg, nbuckets);
    CUDA_CHECK(cudaGetLastError());
    hist_lloyd_kernel<<<M, BLOCK, lloyd_smem, stream>>>(
        d_cnt, d_sum, d_lo, d_hi, d_res_c1d, nbuckets, Kr, max_iter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_seg); cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_cnt); cudaFree(d_sum);
}

} // namespace jhq_gpu
