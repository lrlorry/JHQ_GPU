#include "v2_topk/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <math_constants.h>
#include <cmath>
#include <algorithm>

namespace jhq_gpu {

// ── Kernel 1: primary LUT ─────────────────────────────────────────────────────
__global__ void build_primary_lut_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_c1d,
    float*                    d_lut,
    int M, int Ds, int K1D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * Ds * K1D) return;
    int j = tid % K1D, mk = tid / K1D, k = mk % Ds, m = mk / Ds;
    float diff = d_q_rot[m * Ds + k] - d_c1d[j];
    d_lut[tid] = diff * diff;
}

// ── Kernel 2: ADC scan ────────────────────────────────────────────────────────
__global__ void adc_scan_kernel(
    const float*   __restrict__ d_lut,
    const uint8_t* __restrict__ d_codes,
    float*                      d_dists,
    int N, int M, int Ds, int K1D, int bits_per_dim)
{
    extern __shared__ float s_lut[];
    int lut_size = M * Ds * K1D;
    for (int i = threadIdx.x; i < lut_size; i += blockDim.x)
        s_lut[i] = d_lut[i];
    __syncthreads();

    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= N) return;

    const uint8_t* code  = d_codes + (long long)vid * M;
    const int      kmask = K1D - 1;
    float dist = 0.f;
    for (int m = 0; m < M; m++) {
        uint8_t cm      = code[m];
        const float* lm = s_lut + m * Ds * K1D;
        for (int k = 0; k < Ds; k++)
            dist += lm[k * K1D + ((cm >> (k * bits_per_dim)) & kmask)];
    }
    d_dists[vid] = dist;
}

// ── Kernel 3: pack (dist, idx) → uint64_t ────────────────────────────────────
// Non-negative float bits sort identically to uint32; high 32 bits = dist.
__global__ void pack_kernel(
    const float* __restrict__ d_dists,
    const int*   __restrict__ d_indices,
    uint64_t*                 d_packed,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    d_packed[i] = ((uint64_t)__float_as_uint(d_dists[i]) << 32)
                | (uint32_t)d_indices[i];
}

// ── Kernel 4: unpack top-ck packed entries back to (dist, idx) ───────────────
__global__ void unpack_kernel(
    const uint64_t* __restrict__ d_packed,
    float*                       d_dists,
    int*                         d_indices,
    int ck)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ck) return;
    d_dists[i]   = __uint_as_float((uint32_t)(d_packed[i] >> 32));
    d_indices[i] = (int)(d_packed[i] & 0xFFFFFFFFULL);
}

// ── Kernel 5: residual LUT ────────────────────────────────────────────────────
__global__ void build_residual_lut_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_res_c1d,
    float*                    d_lut_r,
    int d, int Kr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d * Kr) return;
    int j = tid % Kr, dim = tid / Kr;
    float diff = d_q_rot[dim] - d_res_c1d[j];
    d_lut_r[tid] = diff * diff;
}

// ── Kernel 6: residual refine ─────────────────────────────────────────────────
__global__ void residual_refine_kernel(
    const int*     __restrict__ d_cand_ids,
    const float*   __restrict__ d_cand_dists,
    const float*   __restrict__ d_lut_r,
    const uint8_t* __restrict__ d_res_codes,
    const float*   __restrict__ d_corrections,
    float*                      d_comp_dists,
    int ck, int d, int Kr, int Br, int bpv)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ck) return;

    int idx = d_cand_ids[tid];
    if (idx < 0) { d_comp_dists[tid] = CUDART_INF_F; return; }

    const uint8_t* rc = d_res_codes + (long long)idx * bpv;
    float d_res = 0.f;
    for (int j = 0; j < d; j++) {
        int ri = (Br == 4)
            ? ((j % 2 == 0) ? (rc[j/2] & 0x0F) : (rc[j/2] >> 4))
            : rc[j];
        d_res += d_lut_r[(long long)j * Kr + ri];
    }
    d_comp_dists[tid] = d_cand_dists[tid] + d_res + d_corrections[idx];
}

// ── search_gpu ────────────────────────────────────────────────────────────────
void search_gpu(
    cublasHandle_t        cublas,
    const float*          d_Pi,
    const float*          d_c1d,
    const float*          d_res_c1d,
    const uint8_t*        d_primary_codes,
    const uint8_t*        d_res_codes,
    const float*          d_corrections,
    const float*          d_queries,
    int nq, int N, int d, int M, int Ds, int K1D, int Kr,
    int Br, int bpv, int bits_per_dim,
    float alpha, int k,
    SearchWorkspace&      ws,
    float* h_out_dists,
    int*   h_out_ids)
{
    int ck = std::min((int)std::ceil(alpha * (float)k), N);

    float*    d_q_rot      = ws.d_q_rot;
    float*    d_lut        = ws.d_lut;
    float*    d_dists      = ws.d_dists;
    int*      d_indices    = ws.d_indices;
    float*    d_lut_r      = ws.d_lut_r;
    float*    d_comp_dists = ws.d_comp_dists;
    uint64_t* d_packed     = ws.d_packed;

    const float one = 1.f, zero = 0.f;

    thrust::device_ptr<float>    t_comp   (d_comp_dists);
    thrust::device_ptr<int>      t_indices(d_indices);
    thrust::device_ptr<uint64_t> t_packed (d_packed);

    for (int qi = 0; qi < nq; qi++) {
        const float* d_q = d_queries + (long long)qi * d;

        // Step 1: JL rotate query
        CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                                 d, d, &one, d_Pi, d,
                                 d_q, 1, &zero, d_q_rot, 1));

        // Step 2: primary LUT
        {
            int total = M * Ds * K1D;
            build_primary_lut_kernel<<<(total+255)/256, 256>>>(
                d_q_rot, d_c1d, d_lut, M, Ds, K1D);
            CUDA_CHECK(cudaGetLastError());
        }

        // Step 3: ADC scan
        {
            const int BLOCK = 256;
            int shm = M * Ds * K1D * (int)sizeof(float);
            adc_scan_kernel<<<(N+BLOCK-1)/BLOCK, BLOCK, shm>>>(
                d_lut, d_primary_codes, d_dists, N, M, Ds, K1D, bits_per_dim);
            CUDA_CHECK(cudaGetLastError());
        }

        // Step 4: top-ck via partial_sort on packed uint64_t — O(N log ck)
        thrust::sequence(t_indices, t_indices + N);
        pack_kernel<<<(N+255)/256, 256>>>(d_dists, d_indices, d_packed, N);
        CUDA_CHECK(cudaGetLastError());
        // nth_element O(N): puts ck smallest into [0,ck) in any order
        // sort O(ck log ck): order those ck entries
        thrust::nth_element(t_packed, t_packed + ck, t_packed + N);
        thrust::sort(t_packed, t_packed + ck);
        unpack_kernel<<<(ck+255)/256, 256>>>(d_packed, d_dists, d_indices, ck);
        CUDA_CHECK(cudaGetLastError());

        // Step 5: residual LUT
        {
            int total = d * Kr;
            build_residual_lut_kernel<<<(total+255)/256, 256>>>(
                d_q_rot, d_res_c1d, d_lut_r, d, Kr);
            CUDA_CHECK(cudaGetLastError());
        }

        // Step 6: residual refinement on top-ck candidates
        {
            residual_refine_kernel<<<(ck+255)/256, 256>>>(
                d_indices, d_dists,
                d_lut_r, d_res_codes, d_corrections,
                d_comp_dists, ck, d, Kr, Br, bpv);
            CUDA_CHECK(cudaGetLastError());
        }

        // Step 7: final top-k from ck composite distances
        thrust::sort_by_key(t_comp, t_comp + ck, t_indices);

        CUDA_CHECK(cudaMemcpy(h_out_dists + (long long)qi * k,
                              d_comp_dists, k * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out_ids + (long long)qi * k,
                              d_indices,   k * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }
}

} // namespace jhq_gpu
