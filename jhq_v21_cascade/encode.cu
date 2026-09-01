#include "jhq_v21_cascade/encode.cuh"
#include "common/cuda_utils.cuh"

namespace jhq_gpu {

__device__ __forceinline__
int nearest_sorted_dev(const float* arr, int n, float v) {
    if (n == 1) return 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (v < 0.5f * (arr[mid] + arr[mid + 1]))
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
__global__ void primary_encode_kernel(
    const float* __restrict__ d_y,
    uint8_t*                  d_codes,
    const float* __restrict__ d_cent,     // [M][K][Ds]
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
    uint8_t*                    d_res2_codes,
    const float*   __restrict__ d_res2_c1d, // [M][Kr2], null when Br2 == 0
    int N, int d, int M, int Ds, int K, int Kr, int Br, int bpv,
    int Kr2, int Br2, int bpv2)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= N) return;

    const float*   y     = d_y         + (long long)vid * d;
    const uint8_t* pcode = d_primary   + (long long)vid * M;
    uint8_t*       rcode = d_res_codes + (long long)vid * bpv;

    for (int i = 0; i < bpv; i++) rcode[i] = 0;

    uint8_t* rcode2 = (Br2 > 0) ? d_res2_codes + (long long)vid * bpv2 : nullptr;
    for (int i = 0; i < bpv2; i++) rcode2[i] = 0;

    // The reconstruction is yhat + rhat1 + rhat2, so the cross terms the
    // asymmetric distance cannot see at query time are
    //   2*(yhat.rhat1 + yhat.rhat2 + rhat1.rhat2),
    // all of which are known here. With Br2 = 0 the last two vanish and this
    // is the single-level correction unchanged.
    float dot_y_r1 = 0.f, dot_y_r2 = 0.f, dot_r1_r2 = 0.f;
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
        dot_y_r1 += yhat_j * rhat_j;

        if (Br2 > 0) {
            const float* rcb2 = d_res2_c1d + (long long)m * Kr2;
            float resid2  = resid - rhat_j;
            int   ri2     = nearest_sorted_dev(rcb2, Kr2, resid2);
            float rhat2_j = rcb2[ri2];
            if (Br2 == 2) {
                const int sh = 2 * (j & 3);
                rcode2[j >> 2] |= (uint8_t)((ri2 & 0x03) << sh);
            } else if (Br2 == 4) {
                if (j % 2 == 0) rcode2[j / 2]  = (uint8_t)(ri2 & 0x0F);
                else            rcode2[j / 2] |= (uint8_t)((ri2 & 0x0F) << 4);
            } else {
                rcode2[j] = (uint8_t)ri2;
            }
            dot_y_r2  += yhat_j * rhat2_j;
            dot_r1_r2 += rhat_j * rhat2_j;
        }
    }
    d_corrections[vid] = 2.f * (dot_y_r1 + dot_y_r2 + dot_r1_r2);
}

void launch_primary_encode(
    const float* d_y, uint8_t* d_codes, const float* d_cent,
    int N, int d, int M, int Ds, int K, cudaStream_t stream)
{
    const int BLOCK = 128;
    primary_encode_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_y, d_codes, d_cent, N, d, M, Ds, K);
    CUDA_CHECK(cudaGetLastError());
}

void launch_residual_encode(
    const float* d_y, const uint8_t* d_primary,
    uint8_t* d_res_codes, float* d_corrections,
    const float* d_cent, const float* d_res_c1d,
    int N, int d, int M, int Ds, int K, int Kr, int Br, int bpv,
    uint8_t* d_res2_codes, const float* d_res2_c1d,
    int Kr2, int Br2, int bpv2,
    cudaStream_t stream)
{
    const int BLOCK = 128;
    residual_encode_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_y, d_primary, d_res_codes, d_corrections,
        d_cent, d_res_c1d, d_res2_codes, d_res2_c1d,
        N, d, M, Ds, K, Kr, Br, bpv, Kr2, Br2, bpv2);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace jhq_gpu
