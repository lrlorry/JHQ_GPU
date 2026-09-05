#include "jhq_v33_lloyd_tol/encode.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d  %d\n", __FILE__, __LINE__, (int)_s); \
    abort(); } } while (0)
#endif
#include "common/cuda_utils.cuh"

namespace jhq_gpu {

// The float boundaries put a value in the right cell except when it sits
// closer to a midpoint than a float can resolve. Checking the neighbours in
// double settles it: the codewords and the value are floats and convert
// exactly, so |v-a| and |v-b| are computed without error and the nearer one is
// determined. Costs two comparisons per dimension and no shared memory, where
// storing the boundaries in double would have cost up to 16 KB in the grouped
// kernel.
__device__ __forceinline__
int settle_nearest(const float* cb, int n, float v, int i) {
    const double dv = (double)v;
    double best = fabs(dv - (double)cb[i]);
    if (i > 0) {
        const double dl = fabs(dv - (double)cb[i - 1]);
        if (dl < best) { best = dl; i = i - 1; }
    }
    if (i + 1 < n) {
        const double dr = fabs(dv - (double)cb[i + 1]);
        if (dr < best) { i = i + 1; }
    }
    return i;
}

__device__ __forceinline__
int nearest_sorted_dev(const float* arr, int n, float v) {
    if (n == 1) return 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        // The midpoint is formed in double. Both codewords and v are floats and
        // convert exactly, so the sum and the halving are exact, and the
        // comparison decides even when v sits closer to the midpoint than one
        // float ULP -- which happens: on vogue at M=384, Br=8 a residual landed
        // 1.17e-10 above a midpoint of 2.297826926e-03, where the float ULP is
        // about 1.4e-10, and the float comparison went the other way from the
        // reference. The two codewords were 2.9e-05 apart in relative squared
        // distance, so nothing about the reconstruction turned on it, but the
        // code it produces is a discrete output and the specification says
        // nearest. Deciding it in float leaves that to rounding.
        if ((double)v < 0.5 * ((double)arr[mid] + (double)arr[mid + 1]))
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

// One thread per vector, scanning K centroids of Ds dims per subspace. That is
// K*d work per vector against the old K1D-binary-search's d*log(K1D), so
// encoding is the step that pays for the product quantizer -- but it happens
// once per vector at add() time, while the accuracy it buys is paid back on
// every query.
// One block per (vector tile, subspace), with that subspace's centroids staged
// in shared memory.
//
// The version this replaces gave each thread a whole vector and looped every
// subspace and centroid from global memory. The vector itself is d floats --
// 3 KB at d=768 -- but the centroids it reads are M*K*Ds, 786 KB, and every
// vector read all of them again. Centroid traffic dominated by two orders of
// magnitude and scaled with N; staged per block it scales with M instead.
__global__ void primary_encode_kernel(
    const float* __restrict__ d_y,
    uint8_t*                  d_codes,
    const float* __restrict__ d_cent,     // [M][K][Ds]
    int N, int d, int M, int Ds, int K)
{
    extern __shared__ float s_cent[];     // K * Ds
    const int m = blockIdx.y;
    const float* cm = d_cent + (long long)m * K * Ds;
    for (int i = threadIdx.x; i < K * Ds; i += blockDim.x) s_cent[i] = cm[i];
    __syncthreads();

    for (long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         vid < N; vid += (long long)gridDim.x * blockDim.x) {
        const float* ym = d_y + vid * d + (long long)m * Ds;
        float x[32];                      // Ds <= 32 covers d/M for every setting used
        for (int j = 0; j < Ds; ++j) x[j] = ym[j];

        int   best = 0;
        float bd   = __int_as_float(0x7F800000);
        for (int c = 0; c < K; ++c) {
            const float* cc = s_cent + (long long)c * Ds;
            float acc = 0.f;
            for (int j = 0; j < Ds; ++j) { float t = x[j] - cc[j]; acc += t * t; }
            if (acc < bd) { bd = acc; best = c; }
        }
        d_codes[vid * M + m] = (uint8_t)best;
    }
}

// Kept for the configurations the staged version cannot take: Ds above 32 does
// not fit the per-thread buffer, and K*Ds floats above the shared limit do not
// fit the stage.
__global__ void primary_encode_kernel_global(
    const float* __restrict__ d_y,
    uint8_t*                  d_codes,
    const float* __restrict__ d_cent,
    int N, int d, int M, int Ds, int K)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= N) return;
    const float* y    = d_y     + (long long)vid * d;
    uint8_t*     code = d_codes + (long long)vid * M;
    for (int m = 0; m < M; m++) {
        const float* ym = y + (long long)m * Ds;
        const float* cm = d_cent + (long long)m * K * Ds;
        int   best = 0;
        float bd   = __int_as_float(0x7F800000);
        for (int c = 0; c < K; ++c) {
            const float* cc = cm + (long long)c * Ds;
            float acc = 0.f;
            for (int j = 0; j < Ds; ++j) { float t = ym[j] - cc[j]; acc += t * t; }
            if (acc < bd) { bd = acc; best = c; }
        }
        code[m] = (uint8_t)best;
    }
}

__global__ void residual_encode_kernel(
    const float*   __restrict__ d_y,
    const uint8_t* __restrict__ d_primary,
    uint8_t*                    d_res_codes,
    float*                      d_corrections,
    const float*   __restrict__ d_cent,     // [M][K][Ds]
    const float*   __restrict__ d_res_c1d,  // [M][Kr]
    int N, int d, int M, int Ds, int K, int Kr, int Br, int bpv)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= N) return;

    const float*   y     = d_y         + (long long)vid * d;
    const uint8_t* pcode = d_primary   + (long long)vid * M;
    uint8_t*       rcode = d_res_codes + (long long)vid * bpv;

    for (int i = 0; i < bpv; i++) rcode[i] = 0;

    float dot = 0.f;
    for (int j = 0; j < d; j++) {
        const int m = j / Ds;
        const int k = j - m * Ds;

        // yhat now reads straight out of the subspace centroid rather than
        // unpacking per-dimension indices out of the code byte.
        float yhat_j = d_cent[((long long)m * K + pcode[m]) * Ds + k];

        const float* rcb = d_res_c1d + (long long)m * Kr;
        float resid  = y[j] - yhat_j;
        int   ri     = nearest_sorted_dev(rcb, Kr, resid);
        float rhat_j = rcb[ri];

        if (Br == 4) {
            if (j % 2 == 0) rcode[j / 2]  = (uint8_t)(ri & 0x0F);
            else            rcode[j / 2] |= (uint8_t)((ri & 0x0F) << 4);
        } else {
            rcode[j] = (uint8_t)ri;
        }
        dot += yhat_j * rhat_j;
    }
    d_corrections[vid] = 2.f * dot;
}

void launch_primary_encode(
    const float* d_y, uint8_t* d_codes, const float* d_cent,
    int N, int d, int M, int Ds, int K, cudaStream_t stream)
{
    const int    BLOCK = 256;
    const size_t smem  = (size_t)K * Ds * sizeof(float);
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    if (Ds <= 32 && smem <= (size_t)optin) {
        if (smem > 48u * 1024u)
            CUDA_CHECK(cudaFuncSetAttribute(primary_encode_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        const int gx = (int)std::min((long long)((N + BLOCK - 1) / BLOCK), 4096LL);
        primary_encode_kernel<<<dim3(gx, M), BLOCK, smem, stream>>>(
            d_y, d_codes, d_cent, N, d, M, Ds, K);
    } else {
        primary_encode_kernel_global<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
            d_y, d_codes, d_cent, N, d, M, Ds, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_residual_encode(
    const float* d_y, const uint8_t* d_primary,
    uint8_t* d_res_codes, float* d_corrections,
    const float* d_cent, const float* d_res_c1d,
    int N, int d, int M, int Ds, int K, int Kr, int Br, int bpv,
    cudaStream_t stream)
{
    const int BLOCK = 128;
    residual_encode_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_y, d_primary, d_res_codes, d_corrections,
        d_cent, d_res_c1d, N, d, M, Ds, K, Kr, Br, bpv);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace jhq_gpu

namespace jhq_gpu {
namespace {

__global__ void centroid_sqnorm_kernel(const float* __restrict__ d_cent,
                                       float* d_out, int M, int K, int Ds)
{
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < (long long)M * K; t += (long long)gridDim.x * blockDim.x) {
        const float* c = d_cent + t * Ds;
        float s = 0.f;
        for (int j = 0; j < Ds; ++j) s += c[j] * c[j];
        d_out[t] = s;
    }
}

// argmin over ||c||^2 - 2 y.c, one thread per (vector, subspace-in-chunk).
// The tie goes to the lowest index, as the loop version and the host both do.
__global__ void argmin_from_dots_kernel(
    const float* __restrict__ d_dots,     // [chunk][nb][K], row-major per subspace
    const float* __restrict__ d_sqnorm,   // [M][K]
    uint8_t*                  d_codes,    // [N][M]
    int nb, int K, int M, int m0, int chunk, long long row_offset)
{
    const long long total = (long long)chunk * nb;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += (long long)gridDim.x * blockDim.x) {
        const int mi = (int)(t / nb);          // subspace within the chunk
        const int i  = (int)(t - (long long)mi * nb);
        const float* dots = d_dots + ((long long)mi * nb + i) * K;
        const float* sq   = d_sqnorm + (long long)(m0 + mi) * K;

        int   best = 0;
        float bd   = sq[0] - 2.f * dots[0];
        for (int c = 1; c < K; ++c) {
            const float v = sq[c] - 2.f * dots[c];
            if (v < bd) { bd = v; best = c; }
        }
        d_codes[(row_offset + i) * M + (m0 + mi)] = (uint8_t)best;
    }
}

} // namespace

void launch_centroid_sqnorms(const float* d_cent, float* d_out,
                             int M, int K, int Ds, cudaStream_t stream)
{
    const int BLOCK = 256;
    centroid_sqnorm_kernel<<<((long long)M * K + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_cent, d_out, M, K, Ds);
    CUDA_CHECK(cudaGetLastError());
}

void launch_primary_encode_gemm(cublasHandle_t cublas,
                                const float* d_y, uint8_t* d_codes,
                                const float* d_cent, const float* d_cent_sqnorm,
                                float* d_dots, int dots_capacity_rows,
                                int N, int d, int M, int Ds, int K,
                                cudaStream_t stream)
{
    // The scratch holds chunk*N*K floats; pick the largest chunk it can take so
    // one strided-batched call covers as many subspaces as possible.
    int chunk = dots_capacity_rows / (N > 0 ? N : 1);
    if (chunk < 1) chunk = 1;
    if (chunk > M) chunk = M;

    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    const int BLOCK = 256;

    for (int m0 = 0; m0 < M; m0 += chunk) {
        const int c = (m0 + chunk <= M) ? chunk : (M - m0);
        // C[mi] (K x N, column-major) = A[mi]^T (K x Ds) * B[mi] (Ds x N)
        //   A[mi] = centroids of subspace m0+mi, [K][Ds] row-major = Ds x K col-major
        //   B[mi] = y's slice for that subspace: leading dimension d, stride Ds
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            K, N, Ds,
            &one,
            d_cent + (long long)m0 * K * Ds, Ds, (long long)K * Ds,
            d_y + (long long)m0 * Ds,        d,  (long long)Ds,
            &zero,
            d_dots, K, (long long)N * K,
            c));
        const long long total = (long long)c * N;
        argmin_from_dots_kernel<<<(int)std::min((total + BLOCK - 1) / BLOCK, 4096LL),
                                  BLOCK, 0, stream>>>(
            d_dots, d_cent_sqnorm, d_codes, N, K, M, m0, c, 0);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace jhq_gpu

namespace jhq_gpu {
namespace {

// One thread per (vector, subspace). The level table is tiny -- 2 to 256 floats
// -- so it sits in shared and every dimension reads the same copy.
__global__ void primary_encode_cartesian_kernel(
    const float* __restrict__ d_y,
    uint8_t*                  d_codes,
    const float* __restrict__ d_levels,   // L ascending values
    int L, int N, int d, int M, int Ds)
{
    extern __shared__ float s_lv[];
    for (int i = threadIdx.x; i < L; i += blockDim.x) s_lv[i] = d_levels[i];
    __syncthreads();

    const int m = blockIdx.y;
    for (long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         vid < N; vid += (long long)gridDim.x * blockDim.x) {
        const float* ym = d_y + vid * d + (long long)m * Ds;

        // Digit j of the code in base L is the nearest level to y_j. Ties go to
        // the lower level, matching the generic argmin's strict <.
        int code = 0;
        for (int j = 0; j < Ds; ++j) {
            const float v = ym[j];
            int   best = 0;
            float bd   = (v - s_lv[0]) * (v - s_lv[0]);
            for (int l = 1; l < L; ++l) {
                const float t = v - s_lv[l];
                const float dd = t * t;
                if (dd < bd) { bd = dd; best = l; }
            }
            code = code * L + best;
        }
        d_codes[vid * M + m] = (uint8_t)code;
    }
}

} // namespace

void launch_primary_encode_cartesian(const float* d_y, uint8_t* d_codes,
                                     const float* d_levels, int L,
                                     int N, int d, int M, int Ds,
                                     cudaStream_t stream)
{
    const int BLOCK = 256;
    const int gx = (int)std::min((long long)((N + BLOCK - 1) / BLOCK), 4096LL);
    primary_encode_cartesian_kernel<<<dim3(gx, M), BLOCK,
                                      (size_t)L * sizeof(float), stream>>>(
        d_y, d_codes, d_levels, L, N, d, M, Ds);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace jhq_gpu

namespace jhq_gpu {
namespace {

__global__ void encode_fused_cartesian_kernel(
    const float* __restrict__ d_y,
    const float* __restrict__ d_levels,
    const float* __restrict__ d_res_c1d,
    uint8_t* d_codes, uint8_t* d_res_codes, float* d_corrections,
    int L, int N, int d, int M, int Ds, int Kr, int Br, int bpv, int y_transposed)
{
    extern __shared__ float s_all[];
    float* s_lv  = s_all;              // L levels of the primary product
    float* s_rcb = s_all + L;          // Kr residual codewords for this subspace
    float* s_lb  = s_rcb + Kr;         // L-1 boundaries between primary levels
    float* s_rb  = s_lb + L;           // Kr-1 boundaries between residual codewords

    // Both codebooks are sorted (the levels by construction, the residual
    // codewords by train()), so the nearest codeword is the cell the value
    // falls in, and the cells are delimited by the midpoints. That turns the
    // Kr-way scan into log2(Kr) compares: 8 in place of 256 at Br=8.
    const int m = blockIdx.y;
    for (int i = threadIdx.x; i < L; i += blockDim.x) s_lv[i] = d_levels[i];
    for (int i = threadIdx.x; i < Kr; i += blockDim.x)
        s_rcb[i] = d_res_c1d[(long long)m * Kr + i];
    __syncthreads();
    for (int i = threadIdx.x; i + 1 < L;  i += blockDim.x) s_lb[i] = 0.5f * (s_lv[i]  + s_lv[i + 1]);
    for (int i = threadIdx.x; i + 1 < Kr; i += blockDim.x) s_rb[i] = 0.5f * (s_rcb[i] + s_rcb[i + 1]);
    __syncthreads();

    for (long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         vid < N; vid += (long long)gridDim.x * blockDim.x) {
        uint8_t* rcode = d_res_codes + vid * bpv;
        // [N][d]: this vector's Ds values are contiguous, but the warp's
        // threads are d floats apart. [d][N]: each dimension is one coalesced
        // read across the warp, which is the point of the layout.
        const float* ym = y_transposed ? nullptr : d_y + vid * d + (long long)m * Ds;

        int   code = 0;
        float dot  = 0.f;              // this subspace's share of yhat . rhat
        for (int j = 0; j < Ds; ++j) {
            const float v = y_transposed
                          ? d_y[((long long)m * Ds + j) * N + vid]
                          : ym[j];

            // primary digit: the level whose cell holds v. L and Kr are
            // powers of two, so this is a fixed number of branch-free steps;
            // a value on a boundary goes to the lower index, as the scan did.
            int best = 0;
            for (int w = L >> 1; w > 0; w >>= 1)
                best += (v > s_lb[best + w - 1]) ? w : 0;
            code = code * L + best;
            const float yhat = s_lv[best];

            // residual against the scalar codebook, same rule
            const float r = v - yhat;
            int rb = 0;
            for (int w = Kr >> 1; w > 0; w >>= 1)
                rb += (r > s_rb[rb + w - 1]) ? w : 0;
            rb = settle_nearest(s_rcb, Kr, r, rb);
            const float rhat = s_rcb[rb];
            dot += yhat * rhat;

            const int gj = m * Ds + j;         // dimension index in the full vector
            if (Br == 8) rcode[gj] = (uint8_t)rb;
            else {
                // two 4-bit codes share a byte, and the two dimensions that
                // share one always belong to the same subspace when Ds is even,
                // so this thread owns the whole byte and no atomic is needed
                if ((gj & 1) == 0) rcode[gj >> 1] = (uint8_t)(rb & 0x0F);
                else               rcode[gj >> 1] |= (uint8_t)((rb & 0x0F) << 4);
            }
        }
        d_codes[vid * M + m] = (uint8_t)code;
        atomicAdd(d_corrections + vid, 2.f * dot);
    }
}


// The same computation with each thread owning G consecutive subspaces of one
// vector, for the dimension-major layout. The thread-per-(vector, subspace)
// kernel above stores every code byte, every residual byte and every
// correction share on its own: a warp's 32 lanes are 32 vectors, so each of
// those stores touches 32 sectors for 32 bytes, and there are Ds + 2 of them
// per subspace -- 1152 per vector at M=128. That, not the 73 GB it reads,
// is what held it at 465 ms on stella-trec24. Here the G code bytes leave as
// one 8-byte store, the G*Ds residual codes as 16-byte stores, and the
// correction as one atomic per group: 80 stores per vector. The residual
// codebooks of the G subspaces sit in shared memory as before.
template <int G, int DS, int BR>
__global__ void encode_fused_grouped_kernel(
    const float* __restrict__ d_y,
    const float* __restrict__ d_levels,
    const float* __restrict__ d_res_c1d,
    uint8_t* d_codes, uint8_t* d_res_codes, float* d_corrections,
    int L, int N, int M, int Kr, int bpv)
{
    extern __shared__ float s_all[];
    float* s_lv  = s_all;
    float* s_lb  = s_lv + L;
    float* s_rcb = s_lb + L;
    float* s_rb  = s_rcb + G * Kr;

    const int m0 = blockIdx.y * G;
    for (int i = threadIdx.x; i < L; i += blockDim.x) s_lv[i] = d_levels[i];
    for (int i = threadIdx.x; i < G * Kr; i += blockDim.x)
        s_rcb[i] = d_res_c1d[(long long)m0 * Kr + i];
    __syncthreads();
    for (int i = threadIdx.x; i + 1 < L; i += blockDim.x) s_lb[i] = 0.5f * (s_lv[i] + s_lv[i + 1]);
    for (int t = threadIdx.x; t < G * Kr; t += blockDim.x) {
        const int i = t % Kr;
        if (i + 1 < Kr) s_rb[t] = 0.5f * (s_rcb[t] + s_rcb[t + 1]);
    }
    __syncthreads();

    constexpr int RBYTES = G * DS * BR / 8;      // residual bytes per thread
    constexpr int RWORDS = RBYTES / 4;
    static_assert(RBYTES % 16 == 0, "residual codes of a group must fill whole 16-byte stores");

    for (long long vid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         vid < N; vid += (long long)gridDim.x * blockDim.x) {
        uint32_t rw[RWORDS];
#pragma unroll
        for (int w = 0; w < RWORDS; ++w) rw[w] = 0u;
        unsigned long long cw = 0ull;
        float dot = 0.f;

#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float* rcb = s_rcb + g * Kr;
            const float* rb  = s_rb  + g * Kr;
            const float* col = d_y + ((long long)(m0 + g) * DS) * N + vid;
            int code = 0;
#pragma unroll
            for (int j = 0; j < DS; ++j) {
                const float v = col[(long long)j * N];
                int best = 0;
                for (int w = L >> 1; w > 0; w >>= 1)
                    best += (v > s_lb[best + w - 1]) ? w : 0;
                code = code * L + best;
                const float yhat = s_lv[best];
                const float r = v - yhat;
                int rbi = 0;
                for (int w = Kr >> 1; w > 0; w >>= 1)
                    rbi += (r > rb[rbi + w - 1]) ? w : 0;
                rbi = settle_nearest(rcb, Kr, r, rbi);
                dot += yhat * rcb[rbi];
                // residual code of dimension idx of the group, packed as the
                // per-subspace kernel lays it out: a byte per dimension at
                // Br=8, the even dimension in the low nibble at Br=4
                const int idx = g * DS + j;
                if (BR == 8) rw[idx >> 2] |= (uint32_t)rbi << ((idx & 3) * 8);
                else         rw[idx >> 3] |= (uint32_t)(rbi & 0xF) << ((idx & 7) * 4);
            }
            cw |= (unsigned long long)(code & 0xFF) << (8 * g);
        }

        *reinterpret_cast<uint2*>(d_codes + vid * M + m0) =
            make_uint2((uint32_t)cw, (uint32_t)(cw >> 32));
        uint4* rdst = reinterpret_cast<uint4*>(d_res_codes + vid * bpv + (m0 * DS * BR) / 8);
#pragma unroll
        for (int w = 0; w < RWORDS / 4; ++w)
            rdst[w] = make_uint4(rw[4 * w], rw[4 * w + 1], rw[4 * w + 2], rw[4 * w + 3]);
        atomicAdd(d_corrections + vid, 2.f * dot);
    }
}

template <int DS, int BR>
static void launch_grouped(const float* d_y, const float* d_levels, int L,
                           const float* d_res_c1d, uint8_t* d_codes,
                           uint8_t* d_res_codes, float* d_corrections,
                           int N, int M, int Kr, int bpv, cudaStream_t stream)
{
    constexpr int G = 8;
    const int BLOCK = 256;
    const int gx = (int)std::min((long long)((N + BLOCK - 1) / BLOCK), 4096LL);
    encode_fused_grouped_kernel<G, DS, BR><<<dim3(gx, M / G), BLOCK,
        (size_t)(2 * L + 2 * G * Kr) * sizeof(float), stream>>>(
        d_y, d_levels, d_res_c1d, d_codes, d_res_codes, d_corrections,
        L, N, M, Kr, bpv);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_encode_fused_cartesian(
    const float* d_y, const float* d_levels, int L, const float* d_res_c1d,
    uint8_t* d_codes, uint8_t* d_res_codes, float* d_corrections,
    int N, int d, int M, int Ds, int Kr, int Br, int bpv, int y_transposed,
    cudaStream_t stream)
{
    if (Br == 4 && (Ds & 1))
        throw std::runtime_error("launch_encode_fused_cartesian: Br=4 packs two "
                                 "dimensions per byte, so Ds must be even for a "
                                 "thread to own whole bytes");
    CUDA_CHECK(cudaMemsetAsync(d_corrections, 0, (size_t)N * sizeof(float), stream));
    const int BLOCK = 256;
    const int gx = (int)std::min((long long)((N + BLOCK - 1) / BLOCK), 4096LL);
    if ((L & (L - 1)) || (Kr & (Kr - 1)))
        throw std::runtime_error("launch_encode_fused_cartesian: L and Kr must "
                                 "be powers of two for the cell search");
    if (y_transposed && M % 8 == 0 && (Br == 4 || Br == 8) &&
        (Ds == 4 || Ds == 8 || Ds == 16) && L <= 256 &&
        std::getenv("JHQ_ENCODE_GROUPED_OFF") == nullptr) {
        if (Ds == 4) {
            if (Br == 8) launch_grouped<4, 8>(d_y, d_levels, L, d_res_c1d, d_codes, d_res_codes, d_corrections, N, M, Kr, bpv, stream);
            else         launch_grouped<4, 4>(d_y, d_levels, L, d_res_c1d, d_codes, d_res_codes, d_corrections, N, M, Kr, bpv, stream);
        } else if (Ds == 8) {
            if (Br == 8) launch_grouped<8, 8>(d_y, d_levels, L, d_res_c1d, d_codes, d_res_codes, d_corrections, N, M, Kr, bpv, stream);
            else         launch_grouped<8, 4>(d_y, d_levels, L, d_res_c1d, d_codes, d_res_codes, d_corrections, N, M, Kr, bpv, stream);
        } else {
            if (Br == 8) launch_grouped<16, 8>(d_y, d_levels, L, d_res_c1d, d_codes, d_res_codes, d_corrections, N, M, Kr, bpv, stream);
            else         launch_grouped<16, 4>(d_y, d_levels, L, d_res_c1d, d_codes, d_res_codes, d_corrections, N, M, Kr, bpv, stream);
        }
        return;
    }
    encode_fused_cartesian_kernel<<<dim3(gx, M), BLOCK,
                                    (size_t)(2 * L + 2 * Kr) * sizeof(float), stream>>>(
        d_y, d_levels, d_res_c1d, d_codes, d_res_codes, d_corrections,
        L, N, d, M, Ds, Kr, Br, bpv, y_transposed);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace jhq_gpu
