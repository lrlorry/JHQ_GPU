#include "hblock_v1/search.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock {

// ── Route top-k selection ─────────────────────────────────────────────────────
// One thread block per query. Selects top-ck from [B, K] centroid distances.
// Distance metric: cent_norms[c] - 2 * dots[b, c]  (omitting ||q||² which is constant).
//
// Shared memory layout: s_dist[K], then scratch for K-element sequential select.
__global__ void select_route_topk_kernel(
    const float* __restrict__ dots,       // [B, K]
    const float* __restrict__ cent_norms, // [K]
    int*                      topk_ids,   // [B, ck]
    int K, int ck)
{
    const int   BLOCK = blockDim.x;
    const float INF   = __int_as_float(0x7F800000);
    int bqi = blockIdx.x;
    int tid = threadIdx.x;

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
            if (tid < stride && s_rval[tid + stride] < s_rval[tid]) {
                s_rval[tid] = s_rval[tid + stride];
                s_ridx[tid] = s_ridx[tid + stride];
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

// ── Subtract best routing centroid ───────────────────────────────────────────
// d_out[b, j] = d_in[b, j] - d_cents[top_ids[b * stride], j]
__global__ void subtract_best_cent_kernel(
    const float* __restrict__ d_in,
    const int*   __restrict__ top_ids,   // [B, ck], stride=ck between queries
    const float* __restrict__ d_cents,   // [K, d]
    float* d_out, int d, int B, int stride)
{
    long long total = (long long)B * d;
    for (long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         tid < total; tid += (long long)gridDim.x * blockDim.x) {
        int b = (int)(tid / d);
        int j = (int)(tid % d);
        int c = top_ids[(long long)b * stride];
        d_out[tid] = d_in[tid] - d_cents[(long long)c * d + j];
    }
}

// ── Gather leaf block indices ─────────────────────────────────────────────────
// For each query: iterate top-ck1 × top-ck2 pairs, append leaf blocks into d_leaf_sel.
// Uses single thread per query (pairs are small: ck1 × ck2 ≤ 256).
__global__ void gather_leaf_blocks_kernel(
    const int* __restrict__ top1_ids,    // [B, ck1]
    const int* __restrict__ top2_ids,    // [B, ck2]
    const int* __restrict__ pair_start,  // [K1 * K2]
    const int* __restrict__ pair_cnt,    // [K1 * K2]
    int* leaf_sel,   // [B, ck3]
    int* leaf_cnt,   // [B]
    int B, int ck1, int ck2, int ck3, int K2, int leaf_cap_per_pair)
{
    int bqi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bqi >= B) return;

    const int* my1   = top1_ids + bqi * ck1;
    const int* my2   = top2_ids + bqi * ck2;
    int*       my_sel = leaf_sel + bqi * ck3;
    int cnt = 0;

    for (int i1 = 0; i1 < ck1 && cnt < ck3; ++i1) {
        int c1 = my1[i1];
        if (c1 < 0) continue;
        for (int i2 = 0; i2 < ck2 && cnt < ck3; ++i2) {
            int c2 = my2[i2];
            if (c2 < 0) continue;
            int pidx = c1 * K2 + c2;
            int ps   = pair_start[pidx];
            int pc   = pair_cnt[pidx];
            int take = min(pc, leaf_cap_per_pair);
            take     = min(take, ck3 - cnt);
            for (int b = 0; b < take; ++b)
                my_sel[cnt++] = ps + b;
        }
    }
    leaf_cnt[bqi] = cnt;
}

// ── Fine residual LUT ─────────────────────────────────────────────────────────
// d_lut_fine[b, j, ri] = (d_q_r2[b, j] - d_c1d[ri])²
__global__ void build_fine_lut_kernel(
    const float* __restrict__ d_q_r2,
    const float* __restrict__ d_c1d,
    float* d_lut_fine,
    int B, int d, int Kr)
{
    long long total = (long long)B * d * Kr;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        int j     = local / Kr;
        int ri    = local % Kr;
        float diff = d_q_r2[(long long)bqi * d + j] - d_c1d[ri];
        d_lut_fine[i] = diff * diff;
    }
}

// ── Leaf block fine distance computation ──────────────────────────────────────
// Grid: (B) — one block per query.
// Shared memory: LUT for this query (d × Kr floats), loaded once and reused
//   across all ck3 leaf blocks. Eliminates ck3× redundant global LUT reads vs.
//   the old 2D-grid design.
// Each block iterates leaf slots serially; 128 threads handle one leaf block in
//   parallel (one thread = one vector).
__global__ void leaf_fine_compute_kernel(
    const int*     __restrict__ leaf_sel,    // [B, ck3]
    const int*     __restrict__ leaf_cnt,    // [B]
    const int*     __restrict__ leaf_sizes,  // [n_blocks]
    const uint8_t* __restrict__ leaf_codes,  // [n_blocks, leaf_size, bpv]
    const int*     __restrict__ leaf_ids,    // [n_blocks, leaf_size]
    const float*   __restrict__ lut_fine,    // [B, d, Kr]
    float* fine_dists,   // [B, ck3 * leaf_size]
    int*   fine_ids,     // [B, ck3 * leaf_size]
    int ck3, int leaf_size, int bpv, int d, int Kr, int Br)
{
    const float INF = __int_as_float(0x7F800000);
    int bqi = blockIdx.x;
    int v   = threadIdx.x;

    // Load this query's LUT into shared memory once (d*Kr floats, e.g. 48KB for d=768,Kr=16).
    extern __shared__ float s_lut[];
    const float* gbl_lut = lut_fine + (long long)bqi * d * Kr;
    for (int i = v; i < d * Kr; i += blockDim.x)
        s_lut[i] = gbl_lut[i];
    __syncthreads();

    int n_sel = leaf_cnt[bqi];

    for (int slot = 0; slot < ck3; ++slot) {
        float* my_dists = fine_dists + ((long long)bqi * ck3 + slot) * leaf_size;
        int*   my_ids   = fine_ids   + ((long long)bqi * ck3 + slot) * leaf_size;

        if (slot >= n_sel) {
            my_dists[v] = INF; my_ids[v] = -1;
            continue;
        }

        int leaf_idx = leaf_sel[(long long)bqi * ck3 + slot];
        int n_vecs   = leaf_sizes[leaf_idx];

        if (v >= n_vecs) {
            my_dists[v] = INF; my_ids[v] = -1;
        } else {
            const uint8_t* rc = leaf_codes + ((long long)leaf_idx * leaf_size + v) * bpv;
            float fd = 0.f;
            for (int j = 0; j < d; ++j) {
                int ri = (Br == 4)
                    ? ((j & 1) ? (rc[j >> 1] >> 4) : (rc[j >> 1] & 0x0F))
                    : rc[j];
                fd += s_lut[j * Kr + ri];
            }
            my_dists[v] = fd;
            my_ids[v]   = leaf_ids[(long long)leaf_idx * leaf_size + v];
        }
    }
}

// ── Final top-k across all leaf candidates ────────────────────────────────────
// One thread block per query. Reads fine_dists/ids from global memory.
// Shared memory: only 2*BLOCK for reduction scratch (2KB).
// Tracks eliminated candidates via a global-memory "muted" sentinel (set to INF).
__global__ void final_topk_kernel(
    float*       fine_dists,  // [B, ck3 * leaf_size]  — mutated in-place to elim winners
    const int*   __restrict__ fine_ids,
    float* out_dists,
    int*   out_ids,
    int n_cands, int k, int B)
{
    const int   BLOCK = blockDim.x;
    const float INF   = __int_as_float(0x7F800000);
    int bqi = blockIdx.x;
    int tid = threadIdx.x;
    if (bqi >= B) return;

    extern __shared__ char shm[];
    float* s_rval = (float*)shm;
    int*   s_ridx = (int*)(s_rval + BLOCK);

    float* my_d = fine_dists + (long long)bqi * n_cands;
    float* od   = out_dists  + (long long)bqi * k;
    int*   oi   = out_ids    + (long long)bqi * k;

    for (int r = 0; r < k; ++r) {
        float bv = INF; int bi = -1;
        for (int c = tid; c < n_cands; c += BLOCK) {
            float v = my_d[c];
            if (v < bv) { bv = v; bi = c; }
        }
        s_rval[tid] = bv; s_ridx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_rval[tid + stride] < s_rval[tid]) {
                s_rval[tid] = s_rval[tid + stride];
                s_ridx[tid] = s_ridx[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = s_ridx[0];
            od[r] = (w >= 0) ? my_d[w]                                       : INF;
            oi[r] = (w >= 0) ? fine_ids[(long long)bqi * n_cands + w] : -1;
            if (w >= 0) my_d[w] = INF;
        }
        __syncthreads();
    }
}

// ── search_hblock ─────────────────────────────────────────────────────────────
void search_hblock(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_route1_cents, const float* d_route1_norms,
    const float*   d_route2_cents, const float* d_route2_norms,
    const float*   d_fine_c1d,
    const int*     d_pair_blk_start, const int* d_pair_blk_count,
    const uint8_t* d_leaf_codes, const int* d_leaf_ids, const int* d_leaf_sizes,
    const float*   h_queries,
    int nq, int d, int K1, int K2, int Kr, int Br, int bpv,
    int leaf_size, int ck1, int ck2, int ck3, int k,
    int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists, int* h_out_ids)
{
    if (ws.batch_cap <= 0)
        throw std::runtime_error("hblock: workspace not initialised");

    const int B      = ws.batch_cap;
    const int BLOCK  = 256;
    const int n_cands = ck3 * leaf_size;

    const float one = 1.f, zero = 0.f;

    // Shared memory for select_route_topk_kernel: K floats + 2*BLOCK
    const int route1_smem = (K1 + 2 * BLOCK) * (int)sizeof(float);
    const int route2_smem = (K2 + 2 * BLOCK) * (int)sizeof(float);
    // Shared memory for final_topk_kernel: only 2*BLOCK for reduction scratch
    const int topk_smem = 2 * BLOCK * (int)sizeof(float);

    const int leaf_cap_per_pair = std::max(1, ck3 / (ck1 * ck2));

    // Set shared memory limit for leaf kernel once (d*Kr floats, e.g. 48KB for d=768,Kr=16)
    const int leaf_smem = d * Kr * (int)sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(
        leaf_fine_compute_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, leaf_smem));

    for (int qoff = 0; qoff < nq; qoff += batch_size) {
        int Bq = std::min(batch_size, nq - qoff);

        // Copy query batch (pad to B with zeros)
        std::memcpy(ws.h_q_pinned,
                    h_queries + (long long)qoff * d,
                    (long long)Bq * d * sizeof(float));
        if (Bq < B)
            std::memset(ws.h_q_pinned + (long long)Bq * d, 0,
                        (long long)(B - Bq) * d * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                                   (long long)B * d * sizeof(float),
                                   cudaMemcpyHostToDevice, ws.stream));

        // 1. Rotate
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 d, B, d, &one,
                                 d_Pi, d, ws.d_q_batch, d,
                                 &zero, ws.d_q_rot, d));

        // 2. L1 centroid dots: [B, K1] = q_rot @ C1^T
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K1, B, d, &one,
                                 d_route1_cents, d, ws.d_q_rot, d,
                                 &zero, ws.d_dots1, K1));

        // 3. Select top-ck1 L1 centroids
        select_route_topk_kernel<<<B, BLOCK, route1_smem, ws.stream>>>(
            ws.d_dots1, d_route1_norms, ws.d_top1_ids, K1, ck1);
        CUDA_CHECK(cudaGetLastError());

        // 4. Compute L1 residual: q_r1 = q_rot - C1[top1_ids[b, 0]]
        {
            long long tot = (long long)B * d;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            subtract_best_cent_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_rot, ws.d_top1_ids, d_route1_cents, ws.d_q_r1, d, B, ck1);
            CUDA_CHECK(cudaGetLastError());
        }

        // 5. L2 centroid dots: [B, K2] = q_r1 @ C2^T
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K2, B, d, &one,
                                 d_route2_cents, d, ws.d_q_r1, d,
                                 &zero, ws.d_dots2, K2));

        // 6. Select top-ck2 L2 centroids
        select_route_topk_kernel<<<B, BLOCK, route2_smem, ws.stream>>>(
            ws.d_dots2, d_route2_norms, ws.d_top2_ids, K2, ck2);
        CUDA_CHECK(cudaGetLastError());

        // 7. Compute L2 residual: q_r2 = q_r1 - C2[top2_ids[b, 0]]
        {
            long long tot = (long long)B * d;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            subtract_best_cent_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_r1, ws.d_top2_ids, d_route2_cents, ws.d_q_r2, d, B, ck2);
            CUDA_CHECK(cudaGetLastError());
        }

        // 8. Gather leaf blocks for each query
        gather_leaf_blocks_kernel<<<(B + 63) / 64, 64, 0, ws.stream>>>(
            ws.d_top1_ids, ws.d_top2_ids,
            d_pair_blk_start, d_pair_blk_count,
            ws.d_leaf_sel, ws.d_leaf_cnt,
            B, ck1, ck2, ck3, K2, leaf_cap_per_pair);
        CUDA_CHECK(cudaGetLastError());

        // 9. Build fine residual LUT
        {
            long long tot = (long long)B * d * Kr;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            build_fine_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_r2, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
            CUDA_CHECK(cudaGetLastError());
        }

        // 10. Fine distance computation — LUT loaded once per query into shared memory.
        {
            leaf_fine_compute_kernel<<<B, leaf_size, leaf_smem, ws.stream>>>(
                ws.d_leaf_sel, ws.d_leaf_cnt,
                d_leaf_sizes, d_leaf_codes, d_leaf_ids,
                ws.d_lut_fine,
                ws.d_fine_dists, ws.d_fine_ids,
                ck3, leaf_size, bpv, d, Kr, Br);
            CUDA_CHECK(cudaGetLastError());
        }

        // 11. Final top-k
        final_topk_kernel<<<B, BLOCK, topk_smem, ws.stream>>>(
            ws.d_fine_dists, ws.d_fine_ids,
            ws.d_final_dists, ws.d_final_ids,
            n_cands, k, B);
        CUDA_CHECK(cudaGetLastError());

        // D2H
        CUDA_CHECK(cudaMemcpyAsync(h_out_ids    + (long long)qoff * k,
                                   ws.d_final_ids,
                                   (long long)Bq * k * sizeof(int),
                                   cudaMemcpyDeviceToHost, ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_out_dists  + (long long)qoff * k,
                                   ws.d_final_dists,
                                   (long long)Bq * k * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream));
    }

    // Spin-wait
    cudaError_t err;
    do { err = cudaStreamQuery(ws.stream); } while (err == cudaErrorNotReady);
    CUDA_CHECK(err);
}

} // namespace hblock
