#include "hblock_v8/search.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v8 {

// ── Routing kernels (same as v2/v5/v7) ───────────────────────────────────────

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
        int bqi = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        float diff = d_q_r2[(long long)bqi * d + local / Kr] - d_c1d[local % Kr];
        d_lut_fine[i] = diff * diff;
    }
}

// ── route_queries: GPU routing only, leaves d_leaf_sel + d_lut_fine on GPU ────
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
        throw std::runtime_error("hblock_v8: nq exceeds batch_cap; increase batch_size");

    const int B     = ws.batch_cap;
    const int BLOCK = 256;
    const float one = 1.f, zero = 0.f;
    const int route1_smem     = (K1 + 2 * BLOCK) * sizeof(float);
    const int route2_smem     = (K2 + 2 * BLOCK) * sizeof(float);
    const int leaf_cap_per_pair = std::max(1, ck3 / (ck1 * ck2));

    std::memcpy(ws.h_q_pinned, h_queries, (long long)nq * d * sizeof(float));
    if (nq < B)
        std::memset(ws.h_q_pinned + (long long)nq * d, 0,
                    (long long)(B - nq) * d * sizeof(float));

    cudaStream_t s = ws.stream;

    CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                               (long long)B * d * sizeof(float),
                               cudaMemcpyHostToDevice, s));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             d, B, d, &one, d_Pi, d, ws.d_q_batch, d, &zero, ws.d_q_rot, d));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K1, B, d, &one, d_route1_cents, d, ws.d_q_rot, d,
                             &zero, ws.d_dots1, K1));
    select_route_topk_kernel<<<B, BLOCK, route1_smem, s>>>(
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
                             K2, B, d, &one, d_route2_cents, d, ws.d_q_r1, d,
                             &zero, ws.d_dots2, K2));
    select_route_topk_kernel<<<B, BLOCK, route2_smem, s>>>(
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
        ws.d_top1_ids, ws.d_top2_ids,
        d_pair_blk_start, d_pair_blk_count,
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

    // D2H leaf_sel + leaf_cnt so CPU can build leaf queues for fan-out
    // d_lut_fine remains on GPU for the dispatch phase (not transferred)
    CUDA_CHECK(cudaMemcpyAsync(ws.h_leaf_sel,
                               ws.d_leaf_sel,
                               (long long)nq * ck3 * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ws.h_leaf_cnt,
                               ws.d_leaf_cnt,
                               (long long)nq * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    // d_leaf_sel and d_lut_fine remain on GPU for dispatch phase
}

// ── Leaf Flink kernel ─────────────────────────────────────────────────────────
// Grid: n_leaves_in_dispatch  (one block per leaf shard)
// Block: leaf_size (128 threads)
// smem: leaf codes (49 KB) + s_dist/s_pos scratch (1 KB) + warp scratch (32 B)
//
// For each (qid) in this leaf's queue:
//   1. All 128 threads compute one distance (thread v → vector v in leaf)
//   2. Parallel min reduction (warp shuffle + 4-warp smem) → TOP_P winners
//   3. Write TOP_P (dist, global_id) to d_out_dists/d_out_ids[pi × TOP_P]
//
// No fine_dists[B×ck3×leaf_size] buffer — distances never leave the block.
__global__ void leaf_flink_kernel(
    const int*     __restrict__ d_leaf_ids,    // [n_leaves]
    const int*     __restrict__ d_offsets,     // [n_leaves+1] CSR
    const int*     __restrict__ d_qids,        // [total_pairs]
    const int*     __restrict__ leaf_sizes,
    const uint8_t* __restrict__ leaf_codes,    // AoS [n_leaf_blocks, leaf_size, bpv]
    const int*     __restrict__ leaf_ids_data, // global vector ids
    const float*   __restrict__ lut_fine,      // [batch_cap, d, Kr] on GPU
    float*         d_out_dists,                // [total_pairs × TOP_P]
    int*           d_out_ids,                  // [total_pairs × TOP_P]
    int d, int Kr, int Br, int bpv, int leaf_size, int top_p)
{
    const float INF = __int_as_float(0x7F800000);
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int leaf_blk = d_leaf_ids[bid];
    int p_start  = d_offsets[bid];
    int p_end    = d_offsets[bid + 1];
    int n_vecs   = leaf_sizes[leaf_blk];

    // smem layout:
    //   s_codes    [leaf_size × bpv]  — leaf codes, loaded once
    //   s_dist     [leaf_size]        — per-vector distances for current pair
    //   s_pos      [leaf_size]        — original positions (for id lookup)
    //   s_wdist    [4]                — warp-level min distances
    //   s_wpos     [4]                — warp-level min positions
    extern __shared__ char shm[];
    uint8_t* s_codes = (uint8_t*)shm;
    float*   s_dist  = (float*)(s_codes + leaf_size * bpv);
    int*     s_pos   = (int*)  (s_dist  + leaf_size);
    float*   s_wdist = (float*)(s_pos   + leaf_size);
    int*     s_wpos  = (int*)  (s_wdist + 4);

    // Load leaf codes once, reuse for all queries in this leaf's queue
    int n_bytes = n_vecs * bpv;
    for (int b = tid; b < n_bytes; b += blockDim.x)
        s_codes[b] = leaf_codes[(long long)leaf_blk * leaf_size * bpv + b];
    __syncthreads();

    int n_warps = (leaf_size + 31) / 32;

    for (int pi = p_start; pi < p_end; ++pi) {
        int qid = d_qids[pi];
        const float* my_lut = lut_fine + (long long)qid * d * Kr;

        // Compute distance for vector tid
        float fd = INF;
        if (tid < n_vecs) {
            fd = 0.f;
            const uint8_t* rc = s_codes + (long long)tid * bpv;
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

        long long out_base = (long long)pi * top_p;

        // TOP_P parallel min selections
        for (int r = 0; r < top_p; ++r) {
            // Warp-level reduction
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
                if (bv < INF) s_dist[bp] = INF;  // mark used
            }
            __syncthreads();
        }
    }
}

// ── dispatch_leaves: pack + launch leaf_flink_kernel ─────────────────────────
void dispatch_leaves(
    const int*     d_dispatch_leaf_ids,
    const int*     d_dispatch_offsets,
    const int*     d_dispatch_qids,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,
    float*         d_out_dists,
    int*           d_out_ids,
    int n_leaves, int /*total_pairs*/,
    int d, int Kr, int Br, int bpv, int leaf_size,
    cudaStream_t stream)
{
    if (n_leaves == 0) return;
    int smem = leaf_size * bpv              // leaf codes
             + leaf_size * sizeof(float)    // s_dist
             + leaf_size * sizeof(int)      // s_pos
             + 4         * sizeof(float)    // s_wdist
             + 4         * sizeof(int);     // s_wpos

    // A100 default smem/block = 48 KB; 50 KB leaf codes need explicit raise
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)leaf_flink_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    leaf_flink_kernel<<<n_leaves, leaf_size, smem, stream>>>(
        d_dispatch_leaf_ids, d_dispatch_offsets, d_dispatch_qids,
        d_leaf_sizes, d_leaf_codes, d_leaf_ids_data,
        d_lut_fine,
        d_out_dists, d_out_ids,
        d, Kr, Br, bpv, leaf_size, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v8
