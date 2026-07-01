#include "jhq_v1_plain/encode.cuh"
#include "common/cuda_utils.cuh"

namespace jhq_gpu {

// Binary search in sorted arr[n], return index of nearest element to v.
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

// ── Kernel 1: primary encode ──────────────────────────────────────────────────
// Each thread encodes one vector: loop M subspaces × Ds dims → M bytes.
__global__ void primary_encode_kernel(
    const float* __restrict__ d_y,      // [N × d]
    uint8_t*                  d_codes,  // [N × M]
    const float* __restrict__ d_c1d,    // [K1D]
    int N, int d, int M, int Ds, int K1D, int bits_per_dim)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= N) return;

    const float* y    = d_y    + (long long)vid * d;
    uint8_t*     code = d_codes + (long long)vid * M;

    for (int m = 0; m < M; m++) {
        const float* ym = y + m * Ds;
        uint8_t cm = 0;
        for (int k = 0; k < Ds; k++) {
            int j = nearest_sorted_dev(d_c1d, K1D, ym[k]);
            cm |= (uint8_t)(j << (k * bits_per_dim));
        }
        code[m] = cm;
    }
}

// ── Kernel 2: residual encode + correction ────────────────────────────────────
// Each thread encodes one vector:
//   for each dim j:
//     yhat[j]   = c1d[ bits of primary code at dim j ]
//     resid[j]  = y[j] - yhat[j]
//     ri        = nearest residual codeword
//     pack ri into res_codes (nibble-packed for Br=4)
//     accumulate dot += yhat[j] * res_c1d[ri]
//   correction = 2 * dot
__global__ void residual_encode_kernel(
    const float*   __restrict__ d_y,           // [N × d]
    const uint8_t* __restrict__ d_primary,     // [N × M]
    uint8_t*                    d_res_codes,   // [N × bpv]
    float*                      d_corrections, // [N]
    const float*   __restrict__ d_c1d,         // [K1D]
    const float*   __restrict__ d_res_c1d,     // [Kr]
    int N, int d, int M, int Ds, int K1D, int Kr, int Br, int bpv, int bits_per_dim)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= N) return;

    const float*   y      = d_y       + (long long)vid * d;
    const uint8_t* pcode  = d_primary + (long long)vid * M;
    uint8_t*       rcode  = d_res_codes + (long long)vid * bpv;

    for (int i = 0; i < bpv; i++) rcode[i] = 0;

    const int kmask = K1D - 1;
    float dot = 0.f;

    for (int j = 0; j < d; j++) {
        int m = j / Ds;
        int k = j % Ds;
        int primary_idx = (pcode[m] >> (k * bits_per_dim)) & kmask;
        float yhat_j = d_c1d[primary_idx];

        float resid = y[j] - yhat_j;
        int   ri    = nearest_sorted_dev(d_res_c1d, Kr, resid);
        float rhat_j = d_res_c1d[ri];

        if (Br == 4) {
            if (j % 2 == 0)
                rcode[j / 2]  = (uint8_t)(ri & 0x0F);
            else
                rcode[j / 2] |= (uint8_t)((ri & 0x0F) << 4);
        } else {
            rcode[j] = (uint8_t)ri;
        }
        dot += yhat_j * rhat_j;
    }

    d_corrections[vid] = 2.f * dot;
}

// ── Launch wrappers ───────────────────────────────────────────────────────────

void launch_primary_encode(
    const float* d_y, uint8_t* d_codes, const float* d_c1d,
    int N, int d, int M, int Ds, int K1D, int bits_per_dim,
    cudaStream_t stream)
{
    const int BLOCK = 128;
    primary_encode_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_y, d_codes, d_c1d, N, d, M, Ds, K1D, bits_per_dim);
    CUDA_CHECK(cudaGetLastError());
}

void launch_residual_encode(
    const float* d_y, const uint8_t* d_primary,
    uint8_t* d_res_codes, float* d_corrections,
    const float* d_c1d, const float* d_res_c1d,
    int N, int d, int M, int Ds, int K1D, int Kr, int Br, int bpv, int bits_per_dim,
    cudaStream_t stream)
{
    const int BLOCK = 128;
    residual_encode_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        d_y, d_primary, d_res_codes, d_corrections,
        d_c1d, d_res_c1d,
        N, d, M, Ds, K1D, Kr, Br, bpv, bits_per_dim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace jhq_gpu
