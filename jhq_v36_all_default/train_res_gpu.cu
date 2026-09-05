#include "jhq_v36_all_default/train_res_gpu.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace jhq_gpu {

// Lloyd leaves two things to whoever implements it, and the paper fixes
// neither: where the Kr seeds start, and how many iterations run. Both are
// exposed here so the spread they produce can be measured against the spread
// an estimator produces -- an estimator whose codebook sits inside Lloyd's own
// arbitrariness is not distinguishable from Lloyd. Defaults reproduce every
// earlier version exactly: seed 0 is the quantile placement, 25 iterations.
// Measured both ways in results/v31_init_spread/.
static unsigned res_seed() {
    const char* s = std::getenv("JHQ_RES_SEED");
    return s ? (unsigned)std::strtoul(s, nullptr, 10) : 0u;
}
// Lloyd runs to its fixed point now, so what was an iteration count is a
// safety cap. It is still exposed, because a cap that is hit is a result the
// run record has to carry, and because setting it low reproduces the fixed
// counts v31 swept.
static int res_max_iters(int def) {
    const char* s = std::getenv("JHQ_RES_MAX_ITER");
    return s ? std::atoi(s) : def;
}

// v32 tested the fixed point exactly, on the bin boundaries, and never reached
// it: 1000 iterations hit the cap on all three datasets. That is what an exact
// test does at this size. vogue alone has 96 * 257 = 24,672 boundaries over
// 800k sorted values a subspace, and one boundary sitting between two values a
// float apart flips forever while the codebook it describes stands still.
//
// So the test is on the centroids and relative: converged when no centroid
// moves by more than JHQ_RES_TOL of the largest one. 1e-6 is below what float32
// resolves (2^-23 = 1.2e-7 relative), so a codebook that passes it is the same
// codebook to the precision it is stored in. The count of still-moving
// boundaries is reported alongside, because stopping with three of them moving
// is a different claim from stopping with three thousand.
static float res_tol() {
    const char* s = std::getenv("JHQ_RES_TOL");
    return s ? (float)std::atof(s) : 1e-6f;
}
static const int CHECK_EVERY = 8;

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

// The Kr seeds for one subspace, one block each.
//
// seed 0 is the quantile placement c[i] = sorted[(i + 0.5)/Kr * n], which is
// what the host did and what every earlier version used. Any other seed draws
// Kr positions uniformly at random instead -- Forgy initialisation, expressed
// as positions in the sorted array so the draw is uniform over the data by
// construction -- and sorts them, because assign_from_sorted_kernel finds its
// bin boundaries by binary search and needs c ascending.
//
// Drawing with replacement can repeat a position. That leaves two equal
// centroids whose midpoint is that value, so one of them takes an empty bin
// and update_sort_kernel reseeds it. That is a real outcome of random
// initialisation, not a defect to guard against.
__global__ void seed_init_kernel(
    const float* __restrict__ d_sorted, float* d_c, long long seg, int M,
    int Kr, unsigned seed)
{
    extern __shared__ float s_c[];              // Kr
    const int m = blockIdx.x;
    if (m >= M) return;

    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        long long idx;
        if (seed == 0u) {
            idx = (long long)(((float)i + 0.5f) / (float)Kr * (float)seg);
        } else {
            unsigned h = seed + 0x9E3779B9u * (unsigned)(m * Kr + i);
            h ^= h >> 16; h *= 0x21F0AAADu;     // splitmix32
            h ^= h >> 15; h *= 0x735A2D97u;
            h ^= h >> 15;
            idx = (long long)(h % (unsigned)seg);
        }
        if (idx > seg - 1) idx = seg - 1;
        s_c[i] = d_sorted[(long long)m * seg + idx];
    }
    __syncthreads();

    // Already ascending when seed == 0, since idx rises with i; the sort is
    // what the random draw needs. Same bitonic as update_sort_kernel.
    if (seed != 0u) {
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

// Mean, or the centroid it already had when a cluster drew nothing; then the
// centroids are re-sorted so the next binary search stays valid. One block per
// subspace, Kr <= 256, so an insertion sort in shared is ample.
//
// Holding an empty cell in place is what makes this converge. v33 reseeded it
// to the quantile position sorted[(i+0.5)/Kr * n] instead, which is a dense
// part of the data: the cell would take points from a neighbour, empty some
// other cell, and be thrown back again, forever. 20,000 iterations still ended
// with 1563 of vogue's 24,672 boundaries moving and max |dc| at 2.5e-5 of the
// largest centroid, 250x what float32 jitter can account for. The reseed also
// put c out of order, so the bitonic sort permuted it and the convergence test
// -- which compares position by position -- saw a permutation as motion.
//
// Held in place, an empty cell's centroid still lies between its neighbours'
// (the mean of bin i-1 is below the midpoint that bounds it, and the mean of
// bin i+1 above), so c stays ordered, the sort becomes a no-op, and distortion
// cannot increase. Lloyd then converges, which is the whole point of running it
// to a fixed point. The cost is that a cell which empties stays a wasted
// codeword; the count is reported so that cost is in the record.
__global__ void update_sort_kernel(
    const double* __restrict__ d_sum, const int* __restrict__ d_cnt,
    const float*  __restrict__ d_sorted, float* d_c, long long seg, int Kr,
    int* d_maxdelta, int* d_maxabs,   // float bits, atomicMax; both non-negative
    int* d_empty)                     // cells that drew nothing this iteration
{
    extern __shared__ float s[];
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        const long long k = (long long)m * Kr + i;
        if (d_cnt[k] == 0) {
            s[i] = d_c[k];
            atomicAdd(d_empty, 1);
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
    for (int i = threadIdx.x; i < Kr; i += blockDim.x) {
        const long long k = (long long)m * Kr + i;
        const float prev = d_c[k];
        d_c[k] = s[i];
        // Non-negative floats order the same as their bit patterns as ints, so
        // an integer atomicMax is a float atomicMax here.
        atomicMax(d_maxdelta, __float_as_int(fabsf(s[i] - prev)));
        atomicMax(d_maxabs,   __float_as_int(fabsf(s[i])));
    }
}


// One thread per bin: find where its lower boundary falls in the sorted values,
// then read count and sum off the index and prefix differences.
__global__ void assign_from_sorted_kernel(
    const float*  __restrict__ d_sorted,   // [M][seg], ascending
    const double* __restrict__ d_prefix,   // [M][seg], inclusive sum of d_sorted
    const float*  __restrict__ d_c,        // [M][Kr], ascending
    double* d_sum, int* d_cnt, long long seg, int Kr,
    long long* d_bound,                    // [M][Kr+1], last iteration's
    int* d_moved)                          // += every boundary that moved
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

    // Diagnostic, not the test: how much of the partition is still moving.
    // d_bound starts at -1, so the first pass counts every boundary.
    for (int i = threadIdx.x; i <= Kr; i += blockDim.x) {
        const long long k = (long long)m * (Kr + 1) + i;
        if (d_bound[k] != s_bound[i]) { d_bound[k] = s_bound[i]; atomicAdd(d_moved, 1); }
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
    long long* d_bound   = nullptr;
    int*       d_stat    = nullptr;      // moved, max|dc| bits, max|c| bits
    CUDA_CHECK(cudaMalloc(&d_bound, (size_t)mc * (Kr + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_stat,  4 * sizeof(int)));

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
        seed_init_kernel<<<mb, BLOCK, Kr * sizeof(float), stream>>>(
            d_sorted, c_out, seg, mb, Kr, res_seed());
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

        const int cap = res_max_iters(max_iter);
        const float tol = res_tol();
        CUDA_CHECK(cudaMemsetAsync(d_bound, 0xFF,
                                   (size_t)mb * (Kr + 1) * sizeof(long long), stream));
        int it = 0, h_stat[4] = {0, 0, 0, 0};
        bool conv = false;
        for (; it < cap; ++it) {
            CUDA_CHECK(cudaMemsetAsync(d_stat, 0, 4 * sizeof(int), stream));
            assign_from_sorted_kernel<<<mb, BLOCK, (size_t)(Kr + 1) * sizeof(long long), stream>>>(
                d_sorted, d_prefix, c_out, d_sum, d_cnt, seg, Kr, d_bound, d_stat);
            CUDA_CHECK(cudaGetLastError());
            update_sort_kernel<<<mb, BLOCK, Kr * sizeof(float), stream>>>(
                d_sum, d_cnt, d_sorted, c_out, seg, Kr, d_stat + 1, d_stat + 2, d_stat + 3);
            CUDA_CHECK(cudaGetLastError());
            if ((it + 1) % CHECK_EVERY == 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_stat, d_stat, 4 * sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                float dc, ca;
                std::memcpy(&dc, &h_stat[1], 4); std::memcpy(&ca, &h_stat[2], 4);
                if (ca > 0.f && dc <= tol * ca) { conv = true; ++it; break; }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        {
            float dc, ca;
            std::memcpy(&dc, &h_stat[1], 4); std::memcpy(&ca, &h_stat[2], 4);
            std::fprintf(stderr, "[res-train] subspaces %d-%d: %s at %d iterations, "
                         "rel %.3e, %d boundaries moving, %d cells empty\n",
                         m0, m0 + mb - 1, conv ? "converged" : "CAP", it,
                         ca > 0.f ? dc / ca : 0.f, h_stat[0], h_stat[3]);
        }
    }

    cudaFree(d_seg); cudaFree(d_sorted); cudaFree(d_prefix);
    cudaFree(d_sum); cudaFree(d_cnt); cudaFree(d_off); cudaFree(d_tmp);
    cudaFree(d_bound); cudaFree(d_stat);
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
    long long* d_bound   = nullptr;
    int*       d_stat    = nullptr;      // moved, max|dc| bits, max|c| bits
    CUDA_CHECK(cudaMalloc(&d_bound, (size_t)M_chunk * (Kr + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_stat,  4 * sizeof(int)));

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

    seed_init_kernel<<<M_chunk, BLOCK, Kr * sizeof(float), stream>>>(
        d_sorted, d_res_c1d, seg, M_chunk, Kr, res_seed());
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

    const int cap = res_max_iters(max_iter);
    const float tol = res_tol();
    CUDA_CHECK(cudaMemsetAsync(d_bound, 0xFF,
                               (size_t)M_chunk * (Kr + 1) * sizeof(long long), stream));
    int it = 0, h_stat[4] = {0, 0, 0, 0};
    bool conv = false;
    for (; it < cap; ++it) {
        CUDA_CHECK(cudaMemsetAsync(d_stat, 0, 4 * sizeof(int), stream));
        assign_from_sorted_kernel<<<M_chunk, BLOCK, (size_t)(Kr + 1) * sizeof(long long), stream>>>(
            d_sorted, d_prefix, d_res_c1d, d_sum, d_cnt, seg, Kr, d_bound, d_stat);
        CUDA_CHECK(cudaGetLastError());
        update_sort_kernel<<<M_chunk, BLOCK, Kr * sizeof(float), stream>>>(
            d_sum, d_cnt, d_sorted, d_res_c1d, seg, Kr, d_stat + 1, d_stat + 2, d_stat + 3);
        CUDA_CHECK(cudaGetLastError());
        if ((it + 1) % CHECK_EVERY == 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_stat, d_stat, 4 * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float dc, ca;
            std::memcpy(&dc, &h_stat[1], 4); std::memcpy(&ca, &h_stat[2], 4);
            if (ca > 0.f && dc <= tol * ca) { conv = true; ++it; break; }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    {
        float dc, ca;
        std::memcpy(&dc, &h_stat[1], 4); std::memcpy(&ca, &h_stat[2], 4);
        std::fprintf(stderr, "[res-train] %d subspaces: %s at %d iterations, "
                     "rel %.3e, %d boundaries moving, %d cells empty\n",
                     M_chunk, conv ? "converged" : "CAP", it,
                     ca > 0.f ? dc / ca : 0.f, h_stat[0], h_stat[3]);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_sorted); cudaFree(d_prefix);
    cudaFree(d_sum); cudaFree(d_cnt); cudaFree(d_off); cudaFree(d_tmp);
    cudaFree(d_bound); cudaFree(d_stat);
}

} // namespace jhq_gpu
