#include "hblock/search.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace jhq_gpu {

// ── select_probes_kernel ──────────────────────────────────────────────────────
__global__ void select_probes_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ cent_norms,
    const int*   __restrict__ list_offsets,
    int*                      probe_ids,
    int*                      probe_offsets,
    int*                      query_total,
    int nlist, int nprobe)
{
    extern __shared__ float s[];
    const int BLOCK = blockDim.x;
    float* red_val = s + nlist;
    int*   red_idx = (int*)(red_val + BLOCK);

    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    const float* row = dots + (long long)bqi * nlist;
    for (int c = tid; c < nlist; c += BLOCK)
        s[c] = cent_norms[c] - 2.0f * row[c];
    __syncthreads();

    int* my_ids  = probe_ids     + bqi * nprobe;
    int* my_offs = probe_offsets + bqi * (nprobe + 1);
    int  acc = 0;
    if (tid == 0) my_offs[0] = 0;

    for (int p = 0; p < nprobe; ++p) {
        float bv = __int_as_float(0x7F800000); int bi = -1;
        for (int c = tid; c < nlist; c += BLOCK)
            if (s[c] < bv) { bv = s[c]; bi = c; }
        red_val[tid] = bv; red_idx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && red_val[tid+stride] < red_val[tid]) {
                red_val[tid] = red_val[tid+stride]; red_idx[tid] = red_idx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = red_idx[0];
            s[w] = __int_as_float(0x7F800000);
            my_ids[p] = w;
            acc += list_offsets[w+1] - list_offsets[w];
            my_offs[p+1] = acc;
        }
        __syncthreads();
    }
    if (tid == 0) query_total[bqi] = acc;
}

// ── build_byte_lut_kernel — shared by primary and coarse passes ───────────────
// For primary: called with d_c1d      → lut approximates ||q - primary_hat||²
// For coarse:  called with d_c1d_coarse → lut approximates ||q - coarse_hat||²
// Both use q_rot as the query vector.
__global__ void build_byte_lut_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_c1d,
    float*                    d_byte_lut,
    int B, int d, int M, int Ds, int K1D, int bits_per_dim)
{
    long long total = (long long)B * M * 256;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi      = (int)(i / ((long long)M * 256));
        int local    = (int)(i % ((long long)M * 256));
        int m        = local / 256;
        int byte_val = local % 256;
        const float* q_m  = d_q_rot + (long long)bqi * d + (long long)m * Ds;
        const int    kmask = K1D - 1;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= Ds) break;
            int   j    = (byte_val >> (k * bits_per_dim)) & kmask;
            float diff = q_m[k] - d_c1d[j];
            sum += diff * diff;
        }
        d_byte_lut[i] = sum;
    }
}

// ── scan_ivf_coalesced_kernel — Stage 1 ──────────────────────────────────────
// Identical to v12: list_primary_t is [M, N] → coalesced warp access.
// Outputs top-ck1 candidates per query into topck1_pos / topck1_primary.
__global__ void scan_ivf_coalesced_kernel(
    const float*   __restrict__ d_byte_lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary_t,  // [M, N] IVF order
    const int*     __restrict__ query_total,
    float*                      topck_primary,
    int*                        topck_pos,
    int nprobe, int M, long long N, int ck)
{
    constexpr int K_LOCAL = 4;
    const float   INF     = __int_as_float(0x7F800000);
    const int     BLOCK   = blockDim.x;
    int bqi = blockIdx.x, tid = threadIdx.x;

    extern __shared__ char shm[];
    float* s_cdist   = (float*)shm;
    int*   s_cpos    = (int*)(s_cdist + K_LOCAL * BLOCK);
    float* s_red_val = (float*)(s_cpos + K_LOCAL * BLOCK);
    int*   s_red_idx = (int*)(s_red_val + BLOCK);

    const float* my_lut  = d_byte_lut + (long long)bqi * M * 256;
    int total            = query_total[bqi];
    const int* my_ids    = probe_ids     + bqi * nprobe;
    const int* my_offs   = probe_offsets + bqi * (nprobe + 1);

    float ld[K_LOCAL]; int lp[K_LOCAL];
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) { ld[i] = INF; lp[i] = -1; }

    for (int local_t = tid; local_t < total; local_t += BLOCK) {
        int p = 0;
        while (p + 1 < nprobe && local_t >= my_offs[p + 1]) ++p;
        int abs_pos = list_offsets[my_ids[p]] + (local_t - my_offs[p]);

        float dist = 0.0f;
        #pragma unroll 4
        for (int m = 0; m < M; ++m) {
            uint8_t cm = __ldg(&list_primary_t[(long long)m * N + abs_pos]);
            dist += my_lut[m * 256 + cm];
        }

        if (dist < ld[K_LOCAL - 1]) {
            ld[K_LOCAL - 1] = dist; lp[K_LOCAL - 1] = abs_pos;
            #pragma unroll
            for (int i = K_LOCAL - 1; i > 0 && ld[i] < ld[i-1]; --i) {
                float td = ld[i-1]; ld[i-1] = ld[i]; ld[i] = td;
                int   tp = lp[i-1]; lp[i-1] = lp[i]; lp[i] = tp;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) {
        s_cdist[tid * K_LOCAL + i] = ld[i];
        s_cpos [tid * K_LOCAL + i] = lp[i];
    }
    __syncthreads();

    const int  n_cands  = K_LOCAL * BLOCK;
    float* out_primary  = topck_primary + (long long)bqi * ck;
    int*   out_pos      = topck_pos     + (long long)bqi * ck;

    for (int c = 0; c < ck; ++c) {
        float bv = INF; int bi = -1;
        for (int ci = tid; ci < n_cands; ci += BLOCK)
            if (s_cdist[ci] < bv) { bv = s_cdist[ci]; bi = ci; }
        s_red_val[tid] = bv; s_red_idx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_red_val[tid+stride] < s_red_val[tid]) {
                s_red_val[tid] = s_red_val[tid+stride];
                s_red_idx[tid] = s_red_idx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = s_red_idx[0];
            out_primary[c] = (w >= 0) ? s_cdist[w] : INF;
            out_pos    [c] = (w >= 0) ? s_cpos [w] : -1;
            if (w >= 0) s_cdist[w] = INF;
        }
        __syncthreads();
    }
}

// ── coarse_cascade_kernel — Stage 2 ──────────────────────────────────────────
//
// Takes top-ck1 candidates from primary scan and selects top-ck2 by:
//   combined = primary_dist + coarse_dist
// where coarse_dist is computed from d_coarse_t [M,N] (IVF-order, transposed).
//
// The coarse byte_lut (built with c1d_coarse_ using the same q_rot) acts as a
// secondary discriminator: candidates whose coarse reconstruction is close to q_rot
// get a lower combined score → ranked higher.
//
// Coalescing: threads with consecutive tid access consecutive abs_pos values
// from topck1_pos → not guaranteed to be consecutive (topck1 is sorted by
// primary_dist, not abs_pos). But ck1 is small (≤256 for typical k≤64) so
// total memory reads are tiny regardless of access pattern.
//
__global__ void coarse_cascade_kernel(
    const float*   __restrict__ d_byte_lut,      // [B, M, 256] coarse LUT
    const int*     __restrict__ topck1_pos,       // [B, ck1]
    const float*   __restrict__ topck1_primary,   // [B, ck1]
    const uint8_t* __restrict__ d_coarse_t,       // [M, N] coarse codes (IVF order)
    int*                        topck2_pos,        // [B, ck2] output
    float*                      topck2_primary,    // [B, ck2] primary_dist of survivors
    int ck1, int ck2, int M, long long N)
{
    const float INF   = __int_as_float(0x7F800000);
    const int   BLOCK = blockDim.x;
    int bqi = blockIdx.x, tid = threadIdx.x;

    // Shared layout: [ck1 combined] [ck1 abs_pos] [ck1 primary] [BLOCK reduction]
    extern __shared__ char shm[];
    float* s_comb  = (float*)shm;
    int*   s_pos   = (int*)(s_comb + ck1);
    float* s_prim  = (float*)(s_pos + ck1);
    float* rv      = s_prim + ck1;
    int*   ri      = (int*)(rv + BLOCK);

    const float* my_lut   = d_byte_lut     + (long long)bqi * M * 256;
    const int*   in_pos   = topck1_pos     + (long long)bqi * ck1;
    const float* in_prim  = topck1_primary + (long long)bqi * ck1;

    for (int ci = tid; ci < ck1; ci += BLOCK) {
        int abs_pos = in_pos[ci];
        if (abs_pos < 0) { s_comb[ci] = INF; s_pos[ci] = -1; s_prim[ci] = INF; continue; }

        float cd = 0.0f;
        #pragma unroll 4
        for (int m = 0; m < M; ++m)
            cd += my_lut[(long long)m * 256 + __ldg(&d_coarse_t[(long long)m * N + abs_pos])];

        s_comb[ci] = in_prim[ci] + cd;
        s_pos[ci]  = abs_pos;
        s_prim[ci] = in_prim[ci];
    }
    // Slots [ck1..BLOCK) don't exist; only reduce over ck1 elements.
    __syncthreads();

    int*   out_pos  = topck2_pos     + (long long)bqi * ck2;
    float* out_prim = topck2_primary + (long long)bqi * ck2;

    for (int r = 0; r < ck2; ++r) {
        float bv = INF; int bi = -1;
        for (int ci = tid; ci < ck1; ci += BLOCK)
            if (s_comb[ci] < bv) { bv = s_comb[ci]; bi = ci; }
        rv[tid] = bv; ri[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && rv[tid+stride] < rv[tid]) {
                rv[tid] = rv[tid+stride]; ri[tid] = ri[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = ri[0];
            if (w >= 0) { out_pos[r] = s_pos[w]; out_prim[r] = s_prim[w]; s_comb[w] = INF; }
            else        { out_pos[r] = -1;        out_prim[r] = INF; }
        }
        __syncthreads();
    }
}

// ── build_residual_lut_batched_kernel — Stage 3a ──────────────────────────────
__global__ void build_residual_lut_batched_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_res_c1d,
    float*                    d_lut_r,
    int B, int d, int Kr)
{
    long long total = (long long)B * d * Kr;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        int j     = local % Kr;
        int dim   = local / Kr;
        float diff = d_q_rot[(long long)bqi * d + dim] - d_res_c1d[j];
        d_lut_r[i] = diff * diff;
    }
}

// ── residual_refine_batched_kernel — Stage 3b ─────────────────────────────────
// Operates on ck2 survivors from topck2.
__global__ void residual_refine_batched_kernel(
    const int*     __restrict__ topck_pos,
    const float*   __restrict__ topck_primary,
    const float*   __restrict__ lut_r,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int ck, int d, int Kr, int Br, int bpv, int B)
{
    long long total = (long long)B * ck;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int pos = topck_pos[i];
        if (pos < 0) { comp_dists[i] = __int_as_float(0x7F800000); continue; }
        int bqi = (int)(i / ck);
        const float*   my_lut_r = lut_r + (long long)bqi * d * Kr;
        const uint8_t* rc       = list_res + (long long)pos * bpv;
        float d_res = 0.0f;
        for (int j = 0; j < d; ++j) {
            int ri = (Br == 4)
                ? ((j % 2 == 0) ? (rc[j/2] & 0x0F) : (rc[j/2] >> 4))
                : rc[j];
            d_res += my_lut_r[(long long)j * Kr + ri];
        }
        comp_dists[i] = topck_primary[i] + d_res + list_corr[pos];
    }
}

// ── batched_topk_final_kernel ─────────────────────────────────────────────────
__global__ void batched_topk_final_kernel(
    const float* __restrict__ comp_dists,
    const int*   __restrict__ topck_pos,
    const int*   __restrict__ list_ids,
    float*                    final_dists,
    int*                      final_ids,
    int ck, int k, int B)
{
    const int   BLOCK = blockDim.x;
    const float INF   = __int_as_float(0x7F800000);
    int bqi = blockIdx.x, tid = threadIdx.x;
    if (bqi >= B) return;

    extern __shared__ char shm[];
    float* s_dists = (float*)shm;
    int*   s_pos   = (int*)(s_dists + ck);
    float* red_val = (float*)(s_pos + ck);
    int*   red_idx = (int*)(red_val + BLOCK);

    const float* my_dists = comp_dists + (long long)bqi * ck;
    const int*   my_pos   = topck_pos  + (long long)bqi * ck;
    for (int i = tid; i < ck; i += BLOCK) { s_dists[i] = my_dists[i]; s_pos[i] = my_pos[i]; }
    __syncthreads();

    float* out_dists = final_dists + (long long)bqi * k;
    int*   out_ids   = final_ids   + (long long)bqi * k;

    for (int r = 0; r < k; ++r) {
        float bv = INF; int bi = -1;
        for (int i = tid; i < ck; i += BLOCK)
            if (s_dists[i] < bv) { bv = s_dists[i]; bi = i; }
        red_val[tid] = bv; red_idx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && red_val[tid+stride] < red_val[tid]) {
                red_val[tid] = red_val[tid+stride]; red_idx[tid] = red_idx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = red_idx[0]; int pos = (w >= 0) ? s_pos[w] : -1;
            out_ids[r]   = (pos >= 0) ? list_ids[pos] : -1;
            out_dists[r] = (w  >= 0)  ? s_dists[w]   : INF;
            if (w >= 0) s_dists[w] = INF;
        }
        __syncthreads();
    }
}

// ── Buffer management ─────────────────────────────────────────────────────────
static void realloc_ck_buffers(SearchWorkspace& ws, int batch_cap, int ck1, int ck2, int k) {
    if (ck1 > ws.ck1_cap) {
        cudaFree(ws.d_topck1_pos);     ws.d_topck1_pos     = nullptr;
        cudaFree(ws.d_topck1_primary); ws.d_topck1_primary = nullptr;
        long long n = (long long)batch_cap * ck1;
        CUDA_CHECK(cudaMalloc(&ws.d_topck1_pos,     n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_topck1_primary, n * sizeof(float)));
        ws.ck1_cap = ck1;
    }
    if (ck2 > ws.ck2_cap) {
        cudaFree(ws.d_topck2_pos);     ws.d_topck2_pos     = nullptr;
        cudaFree(ws.d_topck2_primary); ws.d_topck2_primary = nullptr;
        cudaFree(ws.d_comp_dists);     ws.d_comp_dists     = nullptr;
        long long n = (long long)batch_cap * ck2;
        CUDA_CHECK(cudaMalloc(&ws.d_topck2_pos,     n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_topck2_primary, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_comp_dists,     n * sizeof(float)));
        ws.ck2_cap = ck2;
    }
    if (k > ws.k_cap) {
        cudaFree(ws.d_final_ids);   ws.d_final_ids   = nullptr;
        cudaFree(ws.d_final_dists); ws.d_final_dists = nullptr;
        long long n = (long long)batch_cap * k;
        CUDA_CHECK(cudaMalloc(&ws.d_final_ids,   n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_final_dists, n * sizeof(float)));
        ws.k_cap = k;
    }
}

// ── CUDA Graph capture ────────────────────────────────────────────────────────
static void capture_graph(
    SearchWorkspace& ws,
    cublasHandle_t cublas,
    const float*   d_Pi, const float* d_c1d, const float* d_c1d_coarse,
    const float*   d_res_c1d,
    const float*   d_centroids, const float* d_cent_norms,
    const int*     d_list_offsets, const int* d_list_ids,
    const uint8_t* d_primary_t, const uint8_t* d_coarse_t,
    const uint8_t* d_list_res, const float* d_list_corr,
    int B, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe, int Br, int bpv, int bits_per_dim,
    int ck1, int ck2, int k, long long ntotal)
{
    if (ws.graph_exec) { cudaGraphExecDestroy(ws.graph_exec); ws.graph_exec = nullptr; }
    if (ws.graph)      { cudaGraphDestroy(ws.graph);          ws.graph      = nullptr; }

    const float one = 1.0f, zero = 0.0f;
    const int   BLOCK = 256;

    // scan shared mem: K_LOCAL*2*BLOCK floats + 2*BLOCK ints (all 4 bytes)
    const int scan_smem    = (2 * 4 * BLOCK + 2 * BLOCK) * (int)sizeof(float);
    // cascade shared mem: (3*ck1 + 2*BLOCK) * 4 bytes
    const int cascade_smem = (3 * ck1 + 2 * BLOCK) * (int)sizeof(float);
    // topk shared mem: uses ck2 (fine refine outputs ck2 dists)
    const int topk_smem    = (2 * ck2 + 2 * BLOCK) * (int)sizeof(float);

    CUDA_CHECK(cudaStreamBeginCapture(ws.stream, cudaStreamCaptureModeGlobal));

    // 1. Rotate queries
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             d, B, d, &one, d_Pi, d, ws.d_q_batch, d, &zero, ws.d_q_rot, d));
    // 2. Centroid dots
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             nlist, B, d, &one, d_centroids, d,
                             ws.d_q_rot, d, &zero, ws.d_dots, nlist));
    // 3. Select probes
    select_probes_kernel<<<B, BLOCK, (nlist + 2*BLOCK) * (int)sizeof(float), ws.stream>>>(
        ws.d_dots, d_cent_norms, d_list_offsets,
        ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total, nlist, nprobe);

    // 4. Build primary byte LUT
    {
        long long tot  = (long long)B * M * 256;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_byte_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_c1d, ws.d_byte_lut, B, d, M, Ds, K1D, bits_per_dim);
    }

    // 5. Primary IVF scan → top-ck1
    scan_ivf_coalesced_kernel<<<B, BLOCK, scan_smem, ws.stream>>>(
        ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
        d_primary_t, ws.d_query_total,
        ws.d_topck1_primary, ws.d_topck1_pos,
        nprobe, M, ntotal, ck1);

    // 6. Overwrite byte LUT with coarse codebook (same buffer, step 5 is done)
    {
        long long tot  = (long long)B * M * 256;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_byte_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_c1d_coarse, ws.d_byte_lut, B, d, M, Ds, K1D, bits_per_dim);
    }

    // 7. Coarse cascade → top-ck2
    coarse_cascade_kernel<<<B, BLOCK, cascade_smem, ws.stream>>>(
        ws.d_byte_lut, ws.d_topck1_pos, ws.d_topck1_primary,
        d_coarse_t,
        ws.d_topck2_pos, ws.d_topck2_primary,
        ck1, ck2, M, ntotal);

    // 8. Fine residual LUT
    {
        long long tot  = (long long)B * d * Kr;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_residual_lut_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_res_c1d, ws.d_lut_r, B, d, Kr);
    }

    // 9. Fine residual refine on ck2 survivors
    {
        long long tot  = (long long)B * ck2;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        residual_refine_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_topck2_pos, ws.d_topck2_primary,
            ws.d_lut_r, d_list_res, d_list_corr,
            ws.d_comp_dists, ck2, d, Kr, Br, bpv, B);
    }

    // 10. Final top-k from ck2 refined candidates
    batched_topk_final_kernel<<<B, BLOCK, topk_smem, ws.stream>>>(
        ws.d_comp_dists, ws.d_topck2_pos, d_list_ids,
        ws.d_final_dists, ws.d_final_ids, ck2, k, B);

    CUDA_CHECK(cudaStreamEndCapture(ws.stream, &ws.graph));
    CUDA_CHECK(cudaGraphInstantiate(&ws.graph_exec, ws.graph, nullptr, nullptr, 0));

    ws.graph_ck1    = ck1;
    ws.graph_ck2    = ck2;
    ws.graph_nprobe = nprobe;
}

// ── search_hblock ─────────────────────────────────────────────────────────────
void search_hblock(
    cublasHandle_t cublas,
    const float* d_Pi, const float* d_c1d, const float* d_c1d_coarse,
    const float* d_res_c1d,
    const float* d_centroids, const float* d_cent_norms,
    const int* d_list_offsets, const int* d_list_ids,
    const uint8_t* d_primary_t, const uint8_t* d_coarse_t,
    const uint8_t* d_list_res, const float* d_list_corr,
    const float* h_queries,
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe,
    int Br, int bpv, int bits_per_dim,
    float alpha1, float alpha2, int k,
    int batch_size, int ntotal,
    SearchWorkspace& ws,
    float* h_out_dists, int* h_out_ids)
{
    if (ws.batch_cap <= 0)
        throw std::runtime_error("hblock: workspace not initialised — call add() first");
    if (!ws.stream || !ws.h_q_pinned)
        throw std::runtime_error("hblock: stream / pinned buffer missing");

    int ck1 = std::max(k, (int)std::ceil(alpha1 * (float)k));
    int ck2 = std::max(k, (int)std::ceil(alpha2 * (float)k));
    if (ck2 > ck1) ck2 = ck1;

    realloc_ck_buffers(ws, ws.batch_cap, ck1, ck2, k);

    const int B_full = ws.batch_cap;
    if (!ws.graph_exec || ws.graph_ck1 != ck1 || ws.graph_ck2 != ck2 ||
        ws.graph_nprobe != nprobe) {
        capture_graph(ws, cublas,
                      d_Pi, d_c1d, d_c1d_coarse, d_res_c1d,
                      d_centroids, d_cent_norms,
                      d_list_offsets, d_list_ids,
                      d_primary_t, d_coarse_t,
                      d_list_res, d_list_corr,
                      B_full, d, M, Ds, K1D, Kr, nlist, nprobe,
                      Br, bpv, bits_per_dim, ck1, ck2, k, ntotal);
    }

    for (int qoff = 0; qoff < nq; qoff += batch_size) {
        int B = std::min(batch_size, nq - qoff);
        std::memcpy(ws.h_q_pinned,
                    h_queries + (long long)qoff * d, (long long)B * d * sizeof(float));
        if (B < B_full)
            std::memset(ws.h_q_pinned + (long long)B * d, 0,
                        (long long)(B_full - B) * d * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                                   (long long)B_full * d * sizeof(float),
                                   cudaMemcpyHostToDevice, ws.stream));
        CUDA_CHECK(cudaGraphLaunch(ws.graph_exec, ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_out_ids   + (long long)qoff * k,
                                   ws.d_final_ids,
                                   (long long)B * k * sizeof(int),
                                   cudaMemcpyDeviceToHost, ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_out_dists + (long long)qoff * k,
                                   ws.d_final_dists,
                                   (long long)B * k * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream));
    }

    cudaError_t err;
    do { err = cudaStreamQuery(ws.stream); } while (err == cudaErrorNotReady);
    CUDA_CHECK(err);
}

} // namespace jhq_gpu
