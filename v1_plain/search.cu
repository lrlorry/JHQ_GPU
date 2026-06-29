#include "v1_plain/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <math_constants.h>
#include <cmath>
#include <algorithm>

namespace jhq_gpu {

// ── Kernel 1: build primary LUT for one query ─────────────────────────────────
// lut[ m * Ds * K1D + k * K1D + j ] = (q_rot[m*Ds+k] - c1d[j])^2
// One thread per (m, k, j) triple → total M*Ds*K1D threads.
__global__ void build_primary_lut_kernel(
    const float* __restrict__ d_q_rot,  // [d]
    const float* __restrict__ d_c1d,    // [K1D]
    float*                    d_lut,    // [M * Ds * K1D]
    int M, int Ds, int K1D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * Ds * K1D) return;

    int j  = tid % K1D;
    int mk = tid / K1D;
    int k  = mk  % Ds;
    int m  = mk  / Ds;

    float diff = d_q_rot[m * Ds + k] - d_c1d[j];
    d_lut[tid] = diff * diff;
}

// ── Kernel 2: ADC full scan (Phase 1) ────────────────────────────────────────
// Each block loads the full LUT into shared memory once, then each thread
// computes the primary distance for one database vector.
// Shared mem size = M * Ds * K1D * sizeof(float)
//   e.g. M=128, Ds=8, K1D=2 → 2048 * 4 = 8 KB  (well within 48 KB)
__global__ void adc_scan_kernel(
    const float*   __restrict__ d_lut,    // [M * Ds * K1D]  for this query
    const uint8_t* __restrict__ d_codes,  // [N * M]
    float*                      d_dists,  // [N]
    int N, int M, int Ds, int K1D, int bits_per_dim)
{
    extern __shared__ float s_lut[];
    int lut_size = M * Ds * K1D;

    // All threads in the block cooperate to load the LUT
    for (int i = threadIdx.x; i < lut_size; i += blockDim.x)
        s_lut[i] = d_lut[i];
    __syncthreads();

    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= N) return;

    const uint8_t* code  = d_codes + (long long)vid * M;
    const int      kmask = K1D - 1;
    float dist = 0.f;

    for (int m = 0; m < M; m++) {
        uint8_t cm        = code[m];
        const float* lm   = s_lut + m * Ds * K1D;
        for (int k = 0; k < Ds; k++) {
            int j  = (cm >> (k * bits_per_dim)) & kmask;
            dist  += lm[k * K1D + j];
        }
    }
    d_dists[vid] = dist;
}

// ── Kernel 3: build residual LUT for one query ────────────────────────────────
// lut_r[ dim * Kr + j ] = (q_rot[dim] - res_c1d[j])^2
__global__ void build_residual_lut_kernel(
    const float* __restrict__ d_q_rot,    // [d]
    const float* __restrict__ d_res_c1d,  // [Kr]
    float*                    d_lut_r,    // [d * Kr]
    int d, int Kr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d * Kr) return;

    int j   = tid % Kr;
    int dim = tid / Kr;
    float diff = d_q_rot[dim] - d_res_c1d[j];
    d_lut_r[tid] = diff * diff;
}

// ── Kernel 4: residual refine (Phase 2) ──────────────────────────────────────
// For each of the ck candidates, compute composite distance:
//   d_composite = d_primary + d_residual + correction
// One thread per candidate.
__global__ void residual_refine_kernel(
    const int*     __restrict__ d_cand_ids,    // [ck] vector ids
    const float*   __restrict__ d_cand_dists,  // [ck] primary distances
    const float*   __restrict__ d_lut_r,       // [d * Kr]
    const uint8_t* __restrict__ d_res_codes,   // [N * bpv]
    const float*   __restrict__ d_corrections,  // [N]
    float*                      d_comp_dists,   // [ck] output
    int ck, int d, int Kr, int Br, int bpv)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ck) return;

    int idx = d_cand_ids[tid];
    if (idx < 0) { d_comp_dists[tid] = CUDART_INF_F; return; }

    const uint8_t* rc = d_res_codes + (long long)idx * bpv;
    float d_res = 0.f;

    for (int j = 0; j < d; j++) {
        int ri;
        if (Br == 4) {
            ri = (j % 2 == 0) ? (rc[j / 2] & 0x0F) : (rc[j / 2] >> 4);
        } else {
            ri = rc[j];
        }
        d_res += d_lut_r[(long long)j * Kr + ri];
    }

    d_comp_dists[tid] = d_cand_dists[tid] + d_res + d_corrections[idx];
}

// ── search_gpu: main entry point ──────────────────────────────────────────────
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
    int ck = (int)std::ceil(alpha * (float)k);
    ck = std::min(ck, N);

    // Use pre-allocated workspace — no malloc in the hot path.
    float* d_q_rot      = ws.d_q_rot;
    float* d_lut        = ws.d_lut;
    float* d_dists      = ws.d_dists;
    int*   d_indices    = ws.d_indices;
    float* d_lut_r      = ws.d_lut_r;
    float* d_comp_dists = ws.d_comp_dists;

    const float one = 1.f, zero = 0.f;

    // Thrust wrappers
    thrust::device_ptr<float> t_dists  (d_dists);
    thrust::device_ptr<int>   t_indices(d_indices);
    thrust::device_ptr<float> t_comp   (d_comp_dists);

    for (int qi = 0; qi < nq; qi++) {
        const float* d_q = d_queries + (long long)qi * d;

        // ── Step 1: JL rotate query ───────────────────────────────────────────
        // d_Pi is Π stored col-major (d×d) from LAPACK.
        // We want q_rot = Π · q  (treating q as a column vector).
        // cublasSgemv(CUBLAS_OP_N): y = A * x  →  d_q_rot = d_Pi * d_q  ✓
        CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                                 d, d,
                                 &one, d_Pi, d,
                                 d_q, 1,
                                 &zero, d_q_rot, 1));

        // ── Step 2: build primary LUT ─────────────────────────────────────────
        {
            int total = M * Ds * K1D;
            build_primary_lut_kernel<<<(total + 255) / 256, 256>>>(
                d_q_rot, d_c1d, d_lut, M, Ds, K1D);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── Step 3: ADC scan over all N database vectors ──────────────────────
        {
            const int BLOCK = 256;
            int shm = M * Ds * K1D * (int)sizeof(float);
            adc_scan_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK, shm>>>(
                d_lut, d_primary_codes, d_dists, N, M, Ds, K1D, bits_per_dim);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── Step 4: top-ck selection (thrust sort) ────────────────────────────
        // After sort: d_dists[0..ck-1]   = top-ck primary distances (ascending)
        //             d_indices[0..ck-1] = corresponding vector ids
        thrust::sequence(t_indices, t_indices + N);
        thrust::sort_by_key(t_dists, t_dists + N, t_indices);

        // ── Step 5: build residual LUT ────────────────────────────────────────
        {
            int total = d * Kr;
            build_residual_lut_kernel<<<(total + 255) / 256, 256>>>(
                d_q_rot, d_res_c1d, d_lut_r, d, Kr);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── Step 6: residual refinement for top-ck candidates ─────────────────
        // d_indices[0..ck-1] are the candidate ids
        // d_dists  [0..ck-1] are their primary distances
        {
            residual_refine_kernel<<<(ck + 255) / 256, 256>>>(
                d_indices, d_dists,
                d_lut_r, d_res_codes, d_corrections,
                d_comp_dists,
                ck, d, Kr, Br, bpv);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── Step 7: final top-k from ck composite distances ───────────────────
        // Sort (d_comp_dists, d_indices)[0..ck-1] → top-k
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
