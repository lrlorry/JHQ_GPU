#include "hblock_v10/encode.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hblock_v10 {

__global__ void assign_from_dots_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ cent_norms,
    int* assigns, int K, int nb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const float* row = dots + (long long)i * K;
    float best  = cent_norms[0] - 2.0f * row[0];
    int   bestk = 0;
    for (int c = 1; c < K; ++c) {
        float d = cent_norms[c] - 2.0f * row[c];
        if (d < best) { best = d; bestk = c; }
    }
    assigns[i] = bestk;
}

void launch_assign_from_dots(const float* d_dots, const float* d_cent_norms,
                              int* d_assigns, int K, int nb, cudaStream_t s)
{
    assign_from_dots_kernel<<<(nb + 255) / 256, 256, 0, s>>>(
        d_dots, d_cent_norms, d_assigns, K, nb);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void subtract_centroid_kernel(
    const float* __restrict__ d_in,
    const int*   __restrict__ d_codes,
    const float* __restrict__ d_cents,
    float* d_out, int n, int d)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n * d;
    for (; tid < total; tid += (long long)gridDim.x * blockDim.x) {
        int i = (int)(tid / d);
        int j = (int)(tid % d);
        int c = d_codes[i];
        d_out[tid] = d_in[tid] - d_cents[(long long)c * d + j];
    }
}

void launch_subtract_centroid(const float* d_in, const int* d_codes,
                               const float* d_cents, float* d_out,
                               int n, int d, cudaStream_t s)
{
    long long total = (long long)n * d;
    int grid = (int)std::min((total + 255) / 256, (long long)65535);
    subtract_centroid_kernel<<<grid, 256, 0, s>>>(d_in, d_codes, d_cents, d_out, n, d);
    CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__
int nearest_1d_dev(const float* arr, int n, float v)
{
    if (n == 1) return 0;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (v < 0.5f * (arr[mid] + arr[mid + 1])) hi = mid;
        else                                        lo = mid + 1;
    }
    return lo;
}

__global__ void fine_encode_kernel(
    const float* __restrict__ d_r2,
    const float* __restrict__ d_c1d,
    uint8_t*                  d_out,
    int n, int d, int Kr, int Br, int bpv)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n) return;

    const float* r   = d_r2 + (long long)vid * d;
    uint8_t*     out = d_out + (long long)vid * bpv;
    for (int b = 0; b < bpv; ++b) out[b] = 0;

    for (int j = 0; j < d; ++j) {
        int ri = nearest_1d_dev(d_c1d, Kr, r[j]);
        if (Br == 4) {
            if (j % 2 == 0)
                out[j / 2]  = (uint8_t)(ri & 0x0F);
            else
                out[j / 2] |= (uint8_t)((ri & 0x0F) << 4);
        } else {
            out[j] = (uint8_t)ri;
        }
    }
}

void launch_fine_encode(const float* d_r2, const float* d_c1d,
                        uint8_t* d_out, int n, int d, int Kr, int Br, int bpv,
                        cudaStream_t s)
{
    fine_encode_kernel<<<(n + 127) / 128, 128, 0, s>>>(
        d_r2, d_c1d, d_out, n, d, Kr, Br, bpv);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v10
