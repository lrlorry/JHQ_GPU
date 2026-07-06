#include "hblock_v15/search.cuh"
#include "common/cuda_utils.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v15 {

// ── Routing kernels (identical to v14) ───────────────────────────────────────

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
    for (int c = tid; c < K; c += BLOCK) s_dist[c] = cent_norms[c] - 2.0f * row[c];
    __syncthreads();
    int* my_ids = topk_ids + bqi * ck;
    for (int r = 0; r < ck; ++r) {
        float bv = INF; int bi = -1;
        for (int c = tid; c < K; c += BLOCK)
            if (s_dist[c] < bv) { bv = s_dist[c]; bi = c; }
        s_rval[tid] = bv; s_ridx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_rval[tid+stride] < s_rval[tid])
                { s_rval[tid] = s_rval[tid+stride]; s_ridx[tid] = s_ridx[tid+stride]; }
            __syncthreads();
        }
        if (tid == 0) {
            int w = s_ridx[0]; my_ids[r] = (w >= 0) ? w : -1;
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
    if (nq > ws.batch_cap) throw std::runtime_error("hblock_v15: nq > batch_cap");
    const int B = ws.batch_cap, BLOCK = 256;
    const float one = 1.f, zero = 0.f;
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
    {
        long long tot = (long long)B * d * Kr;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_fine_lut_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_q_r2, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ── v15 new kernels ───────────────────────────────────────────────────────────

__global__ void init_heap_kernel(float* vals, int* ids, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { vals[i] = __int_as_float(0x7F800000); ids[i] = -1; }
}

// For each query, record its top-ck1 c1 assignments into per-ci lists.
__global__ void precompute_ci_groups_kernel(
    const int* __restrict__ d_topk1, int nq, int ck1, int K1,
    int* d_ci_lists, int* d_ci_counts, int list_stride)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    for (int r = 0; r < ck1; r++) {
        int ci = d_topk1[q * ck1 + r];
        if (ci >= 0 && ci < K1) {
            int slot = atomicAdd(&d_ci_counts[ci], 1);
            if (slot < list_stride)
                d_ci_lists[(long long)ci * list_stride + slot] = q;
        }
    }
}

// Copy LUT rows for ci-group queries into compact buffer (fits in L2 cache).
__global__ void gather_lut_kernel(
    const float* __restrict__ d_lut_full,
    const int*   __restrict__ d_q_ci,
    float*       d_lut_ci,
    int n_ci, int dKr)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n_ci * dKr) return;
    int lq = (int)(i / dKr), off = (int)(i % dKr);
    d_lut_ci[i] = __ldg(&d_lut_full[(long long)d_q_ci[lq] * dKr + off]);
}

// Select leaf blocks belonging to c1=ci for each ci-group query.
// Fixed-stride layout; invalid slots padded with sentinel = n_leaf_blocks.
__global__ void gather_leaf_blocks_ci_kernel(
    const int* __restrict__ top2_ids,
    const int* __restrict__ pair_start,
    const int* __restrict__ pair_cnt,
    const int* __restrict__ d_q_ci,
    int*       leaf_sel_ci,
    int ci, int K2, int n_ci, int ck2, int ck3_ci, int leaf_cap, int n_leaf_blocks)
{
    int lqi = blockIdx.x * blockDim.x + threadIdx.x;
    if (lqi >= n_ci) return;
    int bqi = d_q_ci[lqi];
    int* my = leaf_sel_ci + (long long)lqi * ck3_ci;
    int cnt = 0;
    for (int i2 = 0; i2 < ck2 && cnt < ck3_ci; i2++) {
        int c2 = top2_ids[(long long)bqi * ck2 + i2];
        if (c2 < 0) continue;
        int pidx = ci * K2 + c2;
        int take = min(min(pair_cnt[pidx], leaf_cap), ck3_ci - cnt);
        for (int b = 0; b < take; b++) my[cnt++] = pair_start[pidx] + b;
    }
    for (int s = cnt; s < ck3_ci; s++) my[s] = n_leaf_blocks;  // sentinel
}

// Build pairs in fixed-stride layout; sentinel pairs get qid=-1 (skipped by kernel).
__global__ void build_pairs_ci_kernel(
    const int* __restrict__ d_leaf_sel_ci,
    int* d_pair_leaf, int* d_pair_qid,
    int n_ci, int ck3_ci)
{
    int lqi = blockIdx.x * blockDim.x + threadIdx.x;
    if (lqi >= n_ci) return;
    for (int s = 0; s < ck3_ci; s++) {
        int off = lqi * ck3_ci + s;
        d_pair_leaf[off] = d_leaf_sel_ci[(long long)lqi * ck3_ci + s];
        d_pair_qid[off]  = lqi;  // sentinel pairs detected by leaf_id >= n_leaf_blocks
    }
}

// Fill offsets[i] = i * stride (for fixed-stride segmented_topk).
__global__ void fill_stride_offsets_kernel(int* offsets, int n, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) offsets[i] = i * stride;
}

__global__ void fill_iota_kernel(int* arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

// Leaf PQ kernel — same algorithm as v14 but skips sentinel pairs.
// lut_fine here is the COMPACT ci-group LUT (n_ci rows, L2-resident).
__global__ void leaf_flat_ci_kernel(
    const int*     __restrict__ d_pair_leaf_ids,
    const int*     __restrict__ d_pair_qids,
    const int*     __restrict__ leaf_sizes,
    const uint8_t* __restrict__ leaf_codes,
    const int*     __restrict__ leaf_ids_data,
    const float*   __restrict__ lut_fine,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_leaf_blocks, int d, int Kr, int Br, int bpv, int leaf_size, int top_p)
{
    const float INF = __int_as_float(0x7F800000);
    int pi  = blockIdx.x, tid = threadIdx.x;
    int leaf_blk = d_pair_leaf_ids[pi];
    int qid      = d_pair_qids[pi];

    // Skip invalid (sentinel) pairs — write INF placeholders
    if (leaf_blk < 0 || leaf_blk >= n_leaf_blocks || qid < 0) {
        long long ob = (long long)pi * top_p;
        for (int r = tid; r < top_p; r += blockDim.x)
            { d_out_dists[ob+r] = INF; d_out_ids[ob+r] = -1; }
        return;
    }

    int n_vecs = leaf_sizes[leaf_blk];
    extern __shared__ char shm[];
    float* s_dist  = (float*)shm;
    int*   s_pos   = (int*)(s_dist + leaf_size);
    float* s_wdist = (float*)(s_pos  + leaf_size);
    int*   s_wpos  = (int*)(s_wdist + 4);

    const float*   my_lut    = lut_fine + (long long)qid * d * Kr;
    const uint8_t* leaf_base = leaf_codes + (long long)leaf_blk * bpv * leaf_size;

    float fd = INF;
    if (tid < n_vecs) {
        fd = 0.f;
        if (Br == 4) {
            for (int b = 0; b < bpv; ++b) {
                uint8_t c = __ldg(&leaf_base[b * leaf_size + tid]);
                int j0 = b * 2;
                fd += __ldg(&my_lut[ j0      * Kr + (c & 0x0F)]);
                fd += __ldg(&my_lut[(j0 + 1) * Kr + (c >> 4  )]);
            }
        } else {
            for (int b = 0; b < bpv; ++b) {
                uint8_t c = __ldg(&leaf_base[b * leaf_size + tid]);
                fd += __ldg(&my_lut[b * Kr + c]);
            }
        }
    }
    s_dist[tid] = fd; s_pos[tid] = tid; __syncthreads();

    int n_warps   = (leaf_size + 31) / 32;
    long long ob  = (long long)pi * top_p;
    for (int r = 0; r < top_p; ++r) {
        float mv = s_dist[tid]; int mp = s_pos[tid];
        for (int mask = 16; mask > 0; mask >>= 1) {
            float v2 = __shfl_xor_sync(0xffffffff, mv, mask);
            int   p2 = __shfl_xor_sync(0xffffffff, mp, mask);
            if (v2 < mv) { mv = v2; mp = p2; }
        }
        if ((tid & 31) == 0) { s_wdist[tid>>5] = mv; s_wpos[tid>>5] = mp; }
        __syncthreads();
        if (tid == 0) {
            float bv = s_wdist[0]; int bp = s_wpos[0];
            for (int w = 1; w < n_warps; ++w)
                if (s_wdist[w] < bv) { bv = s_wdist[w]; bp = s_wpos[w]; }
            d_out_dists[ob+r] = bv;
            d_out_ids[ob+r]   = (bv < INF)
                ? leaf_ids_data[(long long)leaf_blk * leaf_size + bp] : -1;
            if (bv < INF) s_dist[bp] = INF;
        }
        __syncthreads();
    }
}

// Segmented top-k for n_ci queries.
// Query qi's pairs are at fixed-stride positions [qi*ck3_ci, (qi+1)*ck3_ci).
__global__ void segmented_topk_ci_kernel(
    const float* __restrict__ d_out_dists,
    const int*   __restrict__ d_out_ids,
    float* d_topk_vals,
    int*   d_topk_ids,
    int n_ci, int k, int ck3_ci, int top_p)
{
    const float INF = __int_as_float(0x7F800000);
    int qi = blockIdx.x, tid = threadIdx.x, BLK = blockDim.x;
    if (qi >= n_ci) return;
    int seg_start = qi * ck3_ci;
    int seg_end   = seg_start + ck3_ci;
    extern __shared__ char shm[];
    float* s_dist = (float*)shm + tid * k;
    int*   s_id   = (int*)((float*)shm + BLK * k) + tid * k;
    for (int i = 0; i < k; i++) { s_dist[i] = INF; s_id[i] = -1; }
    for (int p = seg_start + tid; p < seg_end; p += BLK) {
        for (int r = 0; r < top_p; r++) {
            float dist = __ldg(&d_out_dists[(long long)p * top_p + r]);
            int   vid  = __ldg(&d_out_ids  [(long long)p * top_p + r]);
            if (vid < 0) continue;
            int worst = 0;
            for (int i = 1; i < k; i++) if (s_dist[i] > s_dist[worst]) worst = i;
            if (dist < s_dist[worst]) { s_dist[worst] = dist; s_id[worst] = vid; }
        }
    }
    __syncthreads();
    if (tid == 0) {
        float* od = d_topk_vals + (long long)qi * k;
        int*   oi = d_topk_ids  + (long long)qi * k;
        for (int i = 0; i < k; i++) { od[i] = INF; oi[i] = -1; }
        for (int t = 0; t < BLK; t++) {
            float* td = (float*)shm + t * k;
            int*   ti = (int*)((float*)shm + BLK * k) + t * k;
            for (int i = 0; i < k; i++) {
                if (ti[i] < 0) continue;
                int worst = 0;
                for (int j = 1; j < k; j++) if (od[j] > od[worst]) worst = j;
                if (td[i] < od[worst]) { od[worst] = td[i]; oi[worst] = ti[i]; }
            }
        }
    }
}

// Merge ci's per-query top-k into the global max-heap accumulator.
__global__ void accumulate_heap_kernel(
    const float* __restrict__ d_ci_vals,
    const int*   __restrict__ d_ci_ids,
    const int*   __restrict__ d_q_ci,
    int n_ci, int k,
    float* d_heap_vals,
    int*   d_heap_ids)
{
    int lqi = blockIdx.x * blockDim.x + threadIdx.x;
    if (lqi >= n_ci) return;
    int gqi = d_q_ci[lqi];
    float* hv = d_heap_vals + (long long)gqi * k;
    int*   hi = d_heap_ids  + (long long)gqi * k;
    for (int r = 0; r < k; r++) {
        float d = d_ci_vals[(long long)lqi * k + r];
        int   v = d_ci_ids [(long long)lqi * k + r];
        if (v < 0) continue;
        int worst = 0;
        for (int j = 1; j < k; j++) if (hv[j] > hv[worst]) worst = j;
        if (d < hv[worst]) { hv[worst] = d; hi[worst] = v; }
    }
}

// ── Public entry points ───────────────────────────────────────────────────────

void init_heap(float* vals, int* ids, int n, cudaStream_t s)
{
    if (n <= 0) return;
    init_heap_kernel<<<(n + 255) / 256, 256, 0, s>>>(vals, ids, n);
    CUDA_CHECK(cudaGetLastError());
}

void precompute_ci_groups(int nq, int ck1, int K1, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    CUDA_CHECK(cudaMemsetAsync(ws.d_ci_counts, 0, (long long)K1 * sizeof(int), s));
    precompute_ci_groups_kernel<<<(nq+255)/256, 256, 0, s>>>(
        ws.d_top1_ids, nq, ck1, K1, ws.d_ci_lists, ws.d_ci_counts, ws.batch_cap);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(ws.h_ci_counts, ws.d_ci_counts,
                               (long long)K1 * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));  // wait for h_ci_counts to be ready
}

void process_ci(
    int ci, int n_ci, int K2, int n_leaf_blocks,
    int ck2, int ck3_ci, int leaf_cap_ci, int k,
    const int*     d_pair_blk_start,
    const int*     d_pair_blk_count,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    int d, int Kr, int Br, int bpv, int leaf_size,
    SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    const int  BLOCK      = 256;
    const int  n_pairs_ci = n_ci * ck3_ci;
    const int* d_q_ci     = ws.d_ci_lists + (long long)ci * ws.batch_cap;

    // a. Gather compact LUT for ci's queries
    {
        long long tot = (long long)n_ci * d * Kr;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        gather_lut_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_lut_fine, d_q_ci, ws.d_lut_ci, n_ci, d * Kr);
        CUDA_CHECK(cudaGetLastError());
    }

    // b. Leaf block selection for ci
    gather_leaf_blocks_ci_kernel<<<(n_ci+63)/64, 64, 0, s>>>(
        ws.d_top2_ids, d_pair_blk_start, d_pair_blk_count,
        d_q_ci, ws.d_leaf_sel_ci,
        ci, K2, n_ci, ck2, ck3_ci, leaf_cap_ci, n_leaf_blocks);
    CUDA_CHECK(cudaGetLastError());

    // c. Build pairs (fixed stride, sentinels for short lists)
    build_pairs_ci_kernel<<<(n_ci+255)/256, 256, 0, s>>>(
        ws.d_leaf_sel_ci, ws.d_pair_leaf_ci_a, ws.d_pair_qid_ci_a,
        n_ci, ck3_ci);
    CUDA_CHECK(cudaGetLastError());

    // d. Leaf PQ kernel — fixed-stride layout, no sort needed
    //    pairs for query lqi are at [lqi*ck3_ci, (lqi+1)*ck3_ci); sentinels via leaf_id
    {
        int smem = leaf_size * (int)(sizeof(float) + sizeof(int))
                 + 4 * (int)(sizeof(float) + sizeof(int));
        leaf_flat_ci_kernel<<<n_pairs_ci, leaf_size, smem, s>>>(
            ws.d_pair_leaf_ci_a, ws.d_pair_qid_ci_a,
            d_leaf_sizes, d_leaf_codes, d_leaf_ids_data,
            ws.d_lut_ci,
            ws.d_out_dists_ci, ws.d_out_ids_ci,
            n_leaf_blocks, d, Kr, Br, bpv, leaf_size, TOP_P);
        CUDA_CHECK(cudaGetLastError());
    }

    // e. Segmented top-k: query qi's results at [qi*ck3_ci, (qi+1)*ck3_ci) — no qid sort needed
    {
        const int TOPK_BLOCK = 32;
        int smem = TOPK_BLOCK * k * (int)(sizeof(float) + sizeof(int));
        segmented_topk_ci_kernel<<<n_ci, TOPK_BLOCK, smem, s>>>(
            ws.d_out_dists_ci, ws.d_out_ids_ci,
            ws.d_ci_topk_vals, ws.d_ci_topk_ids,
            n_ci, k, ck3_ci, TOP_P);
        CUDA_CHECK(cudaGetLastError());
    }

    // g. Accumulate into global heap
    accumulate_heap_kernel<<<(n_ci+255)/256, 256, 0, s>>>(
        ws.d_ci_topk_vals, ws.d_ci_topk_ids,
        d_q_ci, n_ci, k,
        ws.d_heap_vals, ws.d_heap_ids);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v15
