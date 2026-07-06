#include "hblock_v11/search.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v11 {

// ── Routing kernels (identical to v10) ────────────────────────────────────────

__global__ void select_route_topk_kernel(
    const float* __restrict__ dots, const float* __restrict__ cent_norms,
    int* topk_ids, int K, int ck)
{
    const int   BLOCK = blockDim.x;
    const float INF   = __int_as_float(0x7F800000);
    int bqi = blockIdx.x, tid = threadIdx.x;
    extern __shared__ char shm[];
    float* s_dist = (float*)shm;
    float* s_rval = s_dist + K;
    int*   s_ridx = (int*)(s_rval + BLOCK);
    const float* row = dots + (long long)bqi * K;
    for (int c = tid; c < K; c += BLOCK)
        s_dist[c] = cent_norms[c] - 2.0f * row[c];
    __syncthreads();
    int* my_ids = topk_ids + bqi * ck;
    for (int r = 0; r < ck; ++r) {
        float bv = INF; int bi = -1;
        for (int c = tid; c < K; c += BLOCK)
            if (s_dist[c] < bv) { bv = s_dist[c]; bi = c; }
        s_rval[tid] = bv; s_ridx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_rval[tid+stride] < s_rval[tid]) {
                s_rval[tid] = s_rval[tid+stride]; s_ridx[tid] = s_ridx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = s_ridx[0];
            my_ids[r] = (w >= 0) ? w : -1;
            if (w >= 0) s_dist[w] = INF;
        }
        __syncthreads();
    }
}

__global__ void subtract_best_cent_kernel(
    const float* __restrict__ d_in, const int* __restrict__ top_ids,
    const float* __restrict__ d_cents, float* d_out, int d, int B, int stride)
{
    long long total = (long long)B * d;
    for (long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         tid < total; tid += (long long)gridDim.x * blockDim.x) {
        int b = (int)(tid / d), j = (int)(tid % d);
        d_out[tid] = d_in[tid] - d_cents[(long long)top_ids[b * stride] * d + j];
    }
}

__global__ void gather_leaf_blocks_kernel(
    const int* __restrict__ top1_ids, const int* __restrict__ top2_ids,
    const int* __restrict__ pair_start, const int* __restrict__ pair_cnt,
    int* leaf_sel, int* leaf_cnt,
    int B, int ck1, int ck2, int ck3, int K2, int leaf_cap_per_pair)
{
    int bqi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bqi >= B) return;
    const int* my1 = top1_ids + bqi * ck1;
    const int* my2 = top2_ids + bqi * ck2;
    int* my_sel = leaf_sel + bqi * ck3;
    int cnt = 0;
    for (int i1 = 0; i1 < ck1 && cnt < ck3; ++i1) {
        int c1 = my1[i1]; if (c1 < 0) continue;
        for (int i2 = 0; i2 < ck2 && cnt < ck3; ++i2) {
            int c2 = my2[i2]; if (c2 < 0) continue;
            int pidx = c1 * K2 + c2;
            int take = min(min(pair_cnt[pidx], leaf_cap_per_pair), ck3 - cnt);
            for (int b = 0; b < take; ++b) my_sel[cnt++] = pair_start[pidx] + b;
        }
    }
    leaf_cnt[bqi] = cnt;
}

__global__ void build_fine_lut_kernel(
    const float* __restrict__ d_q_r2, const float* __restrict__ d_c1d,
    float* d_lut_fine, int B, int d, int Kr)
{
    long long total = (long long)B * d * Kr;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        float diff = d_q_r2[(long long)bqi * d + local / Kr] - d_c1d[local % Kr];
        d_lut_fine[i] = diff * diff;
    }
}

void route_queries(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_route1_cents, const float* d_route1_norms,
    const float*   d_route2_cents, const float* d_route2_norms,
    const float*   d_fine_c1d,
    const int*     d_pair_blk_start, const int* d_pair_blk_count,
    const float*   h_queries,
    int nq, int d, int K1, int K2, int Kr, int Br,
    int leaf_size, int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws)
{
    if (nq > ws.batch_cap)
        throw std::runtime_error("hblock_v11: nq > batch_cap");

    const int B     = ws.batch_cap;
    const int BLOCK = 256;
    const float one = 1.f, zero = 0.f;
    const int leaf_cap_per_pair = std::max(1, ck3 / (ck1 * ck2));

    cudaStream_t s = ws.stream;

    std::memcpy(ws.h_q_pinned, h_queries, (long long)nq * d * sizeof(float));
    if (nq < B)
        std::memset(ws.h_q_pinned + (long long)nq * d, 0,
                    (long long)(B - nq) * d * sizeof(float));

    CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                               (long long)B * d * sizeof(float), cudaMemcpyHostToDevice, s));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             d, B, d, &one, d_Pi, d, ws.d_q_batch, d, &zero, ws.d_q_rot, d));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K1, B, d, &one, d_route1_cents, d, ws.d_q_rot, d, &zero, ws.d_dots1, K1));
    select_route_topk_kernel<<<B, BLOCK, (K1 + 2*BLOCK)*sizeof(float), s>>>(
        ws.d_dots1, d_route1_norms, ws.d_top1_ids, K1, ck1);
    CUDA_CHECK(cudaGetLastError());

    {
        long long tot = (long long)B * d;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        subtract_best_cent_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_q_rot, ws.d_top1_ids, d_route1_cents, ws.d_q_r1, d, B, ck1);
        CUDA_CHECK(cudaGetLastError());
    }

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K2, B, d, &one, d_route2_cents, d, ws.d_q_r1, d, &zero, ws.d_dots2, K2));
    select_route_topk_kernel<<<B, BLOCK, (K2 + 2*BLOCK)*sizeof(float), s>>>(
        ws.d_dots2, d_route2_norms, ws.d_top2_ids, K2, ck2);
    CUDA_CHECK(cudaGetLastError());

    {
        long long tot = (long long)B * d;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        subtract_best_cent_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_q_r1, ws.d_top2_ids, d_route2_cents, ws.d_q_r2, d, B, ck2);
        CUDA_CHECK(cudaGetLastError());
    }

    gather_leaf_blocks_kernel<<<(B + 63) / 64, 64, 0, s>>>(
        ws.d_top1_ids, ws.d_top2_ids, d_pair_blk_start, d_pair_blk_count,
        ws.d_leaf_sel, ws.d_leaf_cnt,
        B, ck1, ck2, ck3, K2, leaf_cap_per_pair);
    CUDA_CHECK(cudaGetLastError());

    {
        long long tot = (long long)B * d * Kr;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_fine_lut_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_q_r2, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpyAsync(ws.h_leaf_sel, ws.d_leaf_sel,
                               (long long)nq * ck3 * sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ws.h_leaf_cnt, ws.d_leaf_cnt,
                               (long long)nq * sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ── leaf_flat_kernel ──────────────────────────────────────────────────────────
// Grid  = n_pairs (one block per (query, leaf) pair — like v5)
// Block = leaf_size = 128 threads
// Smem  = ~1 KB (reduction scratch only — no leaf codes)
//
// No smem pressure → 16 blocks/SM on SM120 → 64 warps/SM → 100% occupancy.
// High occupancy hides the L2/HBM latency for both code and LUT reads.
// Pairs sorted by leaf_id → consecutive blocks share cached codes in L2.
__global__ void leaf_flat_kernel(
    const int*     __restrict__ d_pair_leaf_ids,
    const int*     __restrict__ d_pair_qids,
    const int*     __restrict__ leaf_sizes,
    const uint8_t* __restrict__ leaf_codes,
    const int*     __restrict__ leaf_ids_data,
    const float*   __restrict__ lut_fine,
    float*         d_out_dists,
    int*           d_out_ids,
    int d, int Kr, int Br, int bpv, int leaf_size, int top_p)
{
    const float INF = __int_as_float(0x7F800000);
    int pi  = blockIdx.x;
    int tid = threadIdx.x;

    int leaf_blk = d_pair_leaf_ids[pi];
    int qid      = d_pair_qids[pi];
    int n_vecs   = leaf_sizes[leaf_blk];

    // Smem: only reduction scratch (~1 KB, no codes)
    extern __shared__ char shm[];
    float* s_dist  = (float*)shm;
    int*   s_pos   = (int*)(s_dist + leaf_size);
    float* s_wdist = (float*)(s_pos  + leaf_size);
    int*   s_wpos  = (int*)(s_wdist + 4);

    const float* my_lut = lut_fine + (long long)qid * d * Kr;

    float fd = INF;
    if (tid < n_vecs) {
        fd = 0.f;
        const uint8_t* rc = leaf_codes + ((long long)leaf_blk * leaf_size + tid) * bpv;
        for (int j = 0; j < d; ++j) {
            int ri = (Br == 4)
                ? ((j & 1) ? (rc[j >> 1] >> 4) : (rc[j >> 1] & 0x0F))
                : rc[j];
            fd += __ldg(&my_lut[j * Kr + ri]);
        }
    }
    s_dist[tid] = fd;
    s_pos [tid] = tid;
    __syncthreads();

    int n_warps = (leaf_size + 31) / 32;
    long long out_base = (long long)pi * top_p;

    for (int r = 0; r < top_p; ++r) {
        float mv = s_dist[tid];
        int   mp = s_pos [tid];
        for (int mask = 16; mask > 0; mask >>= 1) {
            float v2 = __shfl_xor_sync(0xffffffff, mv, mask);
            int   p2 = __shfl_xor_sync(0xffffffff, mp, mask);
            if (v2 < mv) { mv = v2; mp = p2; }
        }
        if ((tid & 31) == 0) { s_wdist[tid >> 5] = mv; s_wpos[tid >> 5] = mp; }
        __syncthreads();
        if (tid == 0) {
            float bv = s_wdist[0]; int bp = s_wpos[0];
            for (int w = 1; w < n_warps; ++w)
                if (s_wdist[w] < bv) { bv = s_wdist[w]; bp = s_wpos[w]; }
            d_out_dists[out_base + r] = bv;
            d_out_ids  [out_base + r] = (bv < INF)
                ? leaf_ids_data[(long long)leaf_blk * leaf_size + bp] : -1;
            if (bv < INF) s_dist[bp] = INF;
        }
        __syncthreads();
    }
}

void launch_leaf_flat(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size,
    cudaStream_t   stream)
{
    int smem = leaf_size * (int)sizeof(float)
             + leaf_size * (int)sizeof(int)
             + 4 * (int)sizeof(float)
             + 4 * (int)sizeof(int);
    leaf_flat_kernel<<<n_pairs, leaf_size, smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_leaf_sizes, d_leaf_codes, d_leaf_ids_data,
        d_lut_fine, d_out_dists, d_out_ids,
        d, Kr, Br, bpv, leaf_size, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v11
