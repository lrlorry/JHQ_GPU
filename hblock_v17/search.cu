#include "hblock_v17/search.cuh"
#include "common/cuda_utils.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v17 {

// ── Routing kernels ───────────────────────────────────────────────────────────

// Select top-ck from K centroids using precomputed dot products.
// s_dist[c] = cent_norms[c] - 2*dots[c]  → smallest = nearest
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

// Subtract centroid for best (first) assignment: d_out[b] = d_in[b] - cents[top_ids[b*stride]]
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

// Gather leaf blocks for each query: scan (c1, c2) pairs → leaf block IDs
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

// Build fine LUT: LUT[qi][dim][k] = (q_r2[qi][dim] - fine_c1d[k])^2
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

void route_queries_v17(
    cublasHandle_t cublas,
    const float*   d_Pi1,
    const float*   d_Pi2,
    const float*   d_route1_cents_proj,
    const float*   d_route1_cents_full,
    const float*   d_route1_norms,
    const float*   d_route2_cents_proj,
    const float*   d_route2_cents_full,
    const float*   d_route2_norms,
    const float*   d_fine_c1d,
    const int*     d_pair_blk_start,
    const int*     d_pair_blk_count,
    const float*   h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int Kr, int Br,
    int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws)
{
    if (nq > ws.batch_cap)
        throw std::runtime_error("hblock_v17: nq > batch_cap");

    const int B     = ws.batch_cap;
    const int BLOCK = 256;
    const float one = 1.f, zero = 0.f;
    const int leaf_cap_per_pair = std::max(1, ck3 / (ck1 * ck2));

    cudaStream_t s = ws.stream;

    // H2D
    std::memcpy(ws.h_q_pinned, h_queries, (long long)nq * d * sizeof(float));
    if (nq < B)
        std::memset(ws.h_q_pinned + (long long)nq * d, 0,
                    (long long)(B - nq) * d * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                               (long long)B * d * sizeof(float), cudaMemcpyHostToDevice, s));

    // L1 projection: d_q_proj1 = d_q_batch × Pi1^T  [B, d_proj]
    // d_Pi1: [d_proj, d] (row-major) → in CUBLAS column-major: [d, d_proj]
    // cublasSgemm: C[d_proj, B] = Pi1[d_proj, d] × q^T[d, B]
    //   => CUBLAS_OP_N × CUBLAS_OP_N: M=d_proj, N=B, K=d
    //   A = Pi1 (d_proj × d, stored col-major as d × d_proj? No.)
    //
    // Row-major [d_proj, d] stored in row-major = FORTRAN [d, d_proj]
    // q [B, d] row-major = FORTRAN [d, B]
    // We want result [B, d_proj] row-major = FORTRAN [d_proj, B]
    //
    // In cuBLAS (col-major): C = A × B
    // C[d_proj, B] = Pi1[d_proj, d] × q_batch[d, B]
    // => A = Pi1 as FORTRAN col-major? Pi1 is stored row-major [d_proj, d]
    //    treating it as col-major [d, d_proj] = Pi1^T.
    // So: C = A × B = Pi1^T × q_batch... we want Pi1 × q_batch = Pi1 (OP_T) × q_batch (OP_N)
    // CUBLAS: C[M×N] = A[M×K] × B[K×N] in col-major
    // We want: d_q_proj1[d_proj, B] = Pi1[d_proj, d] × d_q_batch[d, B]
    //          M=d_proj, N=B, K=d
    //          A = Pi1 as Fortran [d, d_proj] but stored row-major → treat with OP_T
    //          B = d_q_batch as Fortran [d, B] (col-major: d×B)
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d_proj, B, d,
    //             &one, d_Pi1, d, d_q_batch, d, &zero, d_q_proj1, d_proj)
    //
    // This gives d_q_proj1 as [d_proj, B] col-major = [B, d_proj] row-major: CORRECT
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_proj, B, d, &one,
                             d_Pi1, d,           // Pi1 stored row-major [d_proj, d]
                             ws.d_q_batch, d, &zero,
                             ws.d_q_proj1, d_proj));

    // L1 GEMM: dots1[B, K1] = d_q_proj1[B, d_proj] × route1_cents_proj^T[d_proj, K1]
    // Fortran: dots1[K1, B] = cents_proj[K1, d_proj] × q_proj[d_proj, B]
    //   OP_T on cents (row-major [K1, d_proj] → treat as [d_proj, K1] + OP_T)
    //   OP_N on q_proj ([d_proj, B])
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K1, B, d_proj, &one,
                             d_route1_cents_proj, d_proj,
                             ws.d_q_proj1, d_proj, &zero,
                             ws.d_dots1, K1));
    select_route_topk_kernel<<<B, BLOCK, (K1 + 2*BLOCK)*sizeof(float), s>>>(
        ws.d_dots1, d_route1_norms, ws.d_top1_ids, K1, ck1);
    CUDA_CHECK(cudaGetLastError());

    // L1 residual in full d space: r1 = q - C1_full[best_c1]
    {
        long long tot = (long long)B * d;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        subtract_best_cent_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_q_batch, ws.d_top1_ids, d_route1_cents_full, ws.d_q_r1, d, B, ck1);
        CUDA_CHECK(cudaGetLastError());
    }

    // L2 projection: d_q_proj2 = d_q_r1 × Pi2^T  [B, d_proj]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_proj, B, d, &one,
                             d_Pi2, d,
                             ws.d_q_r1, d, &zero,
                             ws.d_q_proj2, d_proj));

    // L2 GEMM: dots2[B, K2] = d_q_proj2[B, d_proj] × route2_cents_proj^T
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K2, B, d_proj, &one,
                             d_route2_cents_proj, d_proj,
                             ws.d_q_proj2, d_proj, &zero,
                             ws.d_dots2, K2));
    select_route_topk_kernel<<<B, BLOCK, (K2 + 2*BLOCK)*sizeof(float), s>>>(
        ws.d_dots2, d_route2_norms, ws.d_top2_ids, K2, ck2);
    CUDA_CHECK(cudaGetLastError());

    // L2 residual: r2 = r1 - C2_full[best_c2]  (used for LUT)
    {
        long long tot = (long long)B * d;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        subtract_best_cent_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_q_r1, ws.d_top2_ids, d_route2_cents_full, ws.d_q_r2, d, B, ck2);
        CUDA_CHECK(cudaGetLastError());
    }

    // Gather leaf block IDs
    gather_leaf_blocks_kernel<<<(B + 63) / 64, 64, 0, s>>>(
        ws.d_top1_ids, ws.d_top2_ids,
        d_pair_blk_start, d_pair_blk_count,
        ws.d_leaf_sel, ws.d_leaf_cnt,
        B, ck1, ck2, ck3, K2, leaf_cap_per_pair);
    CUDA_CHECK(cudaGetLastError());

    // Build fine LUT: [B, d, Kr]
    {
        long long tot = (long long)B * d * Kr;
        int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_fine_lut_kernel<<<grid, BLOCK, 0, s>>>(
            ws.d_q_r2, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
        CUDA_CHECK(cudaGetLastError());
    }

    // D2H leaf counts (needed by CPU to sum n_pairs)
    CUDA_CHECK(cudaMemcpyAsync(ws.h_leaf_cnt, ws.d_leaf_cnt,
                               (long long)nq * sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ── build_pairs_kernel ────────────────────────────────────────────────────────

__global__ void build_pairs_kernel(
    const int* __restrict__ d_leaf_sel,
    const int* __restrict__ d_leaf_cnt,
    const int* __restrict__ d_query_offsets,
    int* d_pair_leaf_ids,
    int* d_pair_qids,
    int n_leaf_blocks, int ck3, int nq)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq) return;
    int cnt = d_leaf_cnt[qi];
    int off = d_query_offsets[qi];
    for (int s = 0; s < cnt; s++) {
        int lb = d_leaf_sel[qi * ck3 + s];
        if (lb >= 0 && lb < n_leaf_blocks) {
            d_pair_leaf_ids[off + s] = lb;
            d_pair_qids    [off + s] = qi;
        }
    }
}

void gpu_build_and_sort_pairs_v17(
    int nq, int n_pairs, int n_leaf_blocks,
    int ck3, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_leaf_cnt, ws.d_query_offsets, nq, s));

    build_pairs_kernel<<<(nq + 255) / 256, 256, 0, s>>>(
        ws.d_leaf_sel, ws.d_leaf_cnt, ws.d_query_offsets,
        ws.d_pair_leaf_a, ws.d_pair_qid_a,
        n_leaf_blocks, ck3, nq);
    CUDA_CHECK(cudaGetLastError());

    int end_bit = 1;
    while ((1 << end_bit) < n_leaf_blocks) end_bit++;
    end_bit = std::min(end_bit + 1, 32);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_pair_leaf_a, ws.d_pair_leaf_b,
        ws.d_pair_qid_a,  ws.d_pair_qid_b,
        n_pairs, 0, end_bit, s));
}

// ── leaf_flat_kernel (same as v14, one block per (query,leaf) pair) ───────────

__global__ void leaf_flat_kernel_v17(
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

    extern __shared__ char shm[];
    float* s_dist  = (float*)shm;
    int*   s_pos   = (int*)(s_dist + leaf_size);
    float* s_wdist = (float*)(s_pos  + leaf_size);
    int*   s_wpos  = (int*)(s_wdist + 4);

    const float* my_lut  = lut_fine + (long long)qid * d * Kr;
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

void launch_leaf_flat_v17(
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
    leaf_flat_kernel_v17<<<n_pairs, leaf_size, smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_leaf_sizes, d_leaf_codes, d_leaf_ids_data,
        d_lut_fine, d_out_dists, d_out_ids,
        d, Kr, Br, bpv, leaf_size, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

// ── iota + qid sort + segmented top-k (for PQ merge and rerank) ──────────────

__global__ void fill_iota_kernel(int* arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

__global__ void segmented_topk_kernel(
    const int*   __restrict__ d_pair_perm,
    const int*   __restrict__ d_query_offsets,
    const float* __restrict__ d_out_dists,
    const int*   __restrict__ d_out_ids,
    float* d_final_dists,
    int*   d_final_ids,
    int nq, int k, int n_pairs, int top_p)
{
    const float INF = __int_as_float(0x7F800000);
    int qi  = blockIdx.x;
    int tid = threadIdx.x;
    int BLK = blockDim.x;
    if (qi >= nq) return;

    int seg_start = d_query_offsets[qi];
    int seg_end   = (qi + 1 < nq) ? d_query_offsets[qi + 1] : n_pairs;

    extern __shared__ char shm[];
    float* s_dist = (float*)shm          + tid * k;
    int*   s_id   = (int*)((float*)shm + BLK * k) + tid * k;

    for (int i = 0; i < k; i++) { s_dist[i] = INF; s_id[i] = -1; }

    for (int p = seg_start + tid; p < seg_end; p += BLK) {
        int pi = d_pair_perm[p];
        for (int r = 0; r < top_p; r++) {
            float dist = __ldg(&d_out_dists[(long long)pi * top_p + r]);
            int   vid  = __ldg(&d_out_ids  [(long long)pi * top_p + r]);
            if (vid < 0) continue;
            int worst = 0;
            for (int i = 1; i < k; i++)
                if (s_dist[i] > s_dist[worst]) worst = i;
            if (dist < s_dist[worst]) { s_dist[worst] = dist; s_id[worst] = vid; }
        }
    }
    __syncthreads();

    if (tid == 0) {
        float* out_d = d_final_dists + (long long)qi * k;
        int*   out_i = d_final_ids   + (long long)qi * k;
        for (int i = 0; i < k; i++) { out_d[i] = INF; out_i[i] = -1; }
        for (int t = 0; t < BLK; t++) {
            float* td = (float*)shm + t * k;
            int*   ti = (int*)((float*)shm + BLK * k) + t * k;
            for (int i = 0; i < k; i++) {
                float d = td[i]; int v = ti[i];
                if (v < 0) continue;
                int worst = 0;
                for (int j = 1; j < k; j++)
                    if (out_d[j] > out_d[worst]) worst = j;
                if (d < out_d[worst]) { out_d[worst] = d; out_i[worst] = v; }
            }
        }
    }
}

void gpu_merge_pq_topk_v17(int nq, int n_pairs, int rerank_r, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    int k = rerank_r;

    fill_iota_kernel<<<(n_pairs + 255) / 256, 256, 0, s>>>(ws.d_pair_leaf_a, n_pairs);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_pair_qid_b,  ws.d_pair_leaf_b,
        ws.d_pair_leaf_a, ws.d_pair_qid_a,
        n_pairs, 0, 10, s));

    const int BLOCK = 32;
    int smem = BLOCK * k * (int)(sizeof(float) + sizeof(int));
    segmented_topk_kernel<<<nq, BLOCK, smem, s>>>(
        ws.d_pair_qid_a, ws.d_query_offsets,
        ws.d_out_dists, ws.d_out_ids,
        ws.d_pq_dists, ws.d_pq_ids,
        nq, k, n_pairs, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_merge_topk_v17(int nq, int n_pairs, int k, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;

    fill_iota_kernel<<<(n_pairs + 255) / 256, 256, 0, s>>>(ws.d_pair_leaf_a, n_pairs);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_pair_qid_b,  ws.d_pair_leaf_b,
        ws.d_pair_leaf_a, ws.d_pair_qid_a,
        n_pairs, 0, 10, s));

    const int BLOCK = 32;
    int smem = BLOCK * k * (int)(sizeof(float) + sizeof(int));
    segmented_topk_kernel<<<nq, BLOCK, smem, s>>>(
        ws.d_pair_qid_a, ws.d_query_offsets,
        ws.d_out_dists, ws.d_out_ids,
        ws.d_final_dists, ws.d_final_ids,
        nq, k, n_pairs, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

// ── Gather + exact rerank ─────────────────────────────────────────────────────

__global__ void gather_vecs_kernel(
    const float* __restrict__ d_base_vecs,
    const int*   __restrict__ d_cand_ids,
    float*                    d_cand_vecs,
    int R, int d)
{
    int qi = blockIdx.y;
    int ri = blockIdx.x;
    int id = d_cand_ids[(long long)qi * R + ri];
    if (id < 0) return;
    float* out = d_cand_vecs + ((long long)qi * R + ri) * d;
    const float* src = d_base_vecs + (long long)id * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        out[j] = src[j];
}

void launch_gather_vecs(
    const float*   d_base_vecs,
    const int*     d_cand_ids,
    float*         d_cand_vecs,
    int B, int R, int d,
    cudaStream_t   s)
{
    // grid: [R, B], block: 128
    dim3 grid(R, B);
    gather_vecs_kernel<<<grid, 128, 0, s>>>(d_base_vecs, d_cand_ids, d_cand_vecs, R, d);
    CUDA_CHECK(cudaGetLastError());
}

// Exact inner product rerank + top-k selection.
// One block per query. 128 threads (4 warps), each warp handles R/4 candidates.
// R must be <= 128 (fits in block) and <= K_MAX.
__global__ void exact_rerank_kernel(
    const float* __restrict__ d_q_batch,    // [B, d]
    const float* __restrict__ d_cand_vecs,  // [B, R, d]
    const int*   __restrict__ d_cand_ids,   // [B, R]
    float*                    d_exact_dists, // [B, R] (written for diagnostics)
    float*                    d_final_dists, // [B, k]
    int*                      d_final_ids,   // [B, k]
    int R, int d, int k)
{
    const float INF = __int_as_float(0x7F800000);
    int qi    = blockIdx.x;
    int tid   = threadIdx.x;
    int lane  = tid & 31;
    int wid   = tid >> 5;          // warp index in [0, n_warps)
    int n_warps = blockDim.x >> 5; // = blockDim.x / 32

    const float* q   = d_q_batch  + (long long)qi * d;
    const int*   ids = d_cand_ids + (long long)qi * R;

    extern __shared__ float shm[];  // [R] dot products
    float* s_dots = shm;

    // Round-robin: warp wid handles candidates wid, wid+n_warps, wid+2*n_warps, ...
    for (int ri = wid; ri < R; ri += n_warps) {
        float dot = 0.f;
        if (ids[ri] >= 0) {
            const float* c = d_cand_vecs + ((long long)qi * R + ri) * d;
            for (int j = lane; j < d; j += 32)
                dot += q[j] * c[j];
        }
        // Warp reduction
        for (int mask = 16; mask > 0; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);
        if (lane == 0) s_dots[ri] = dot;
    }
    __syncthreads();

    // Thread 0: write exact dists + top-k selection
    if (tid == 0) {
        float* ed  = d_exact_dists + (long long)qi * R;
        float* out_d = d_final_dists + (long long)qi * k;
        int*   out_i = d_final_ids   + (long long)qi * k;
        for (int i = 0; i < k; i++) { out_d[i] = INF; out_i[i] = -1; }

        for (int r = 0; r < R; ++r) {
            if (ids[r] < 0) { ed[r] = INF; continue; }
            float dist = -s_dots[r];  // negative inner product → smaller = better
            ed[r] = dist;
            int worst = 0;
            for (int i = 1; i < k; i++)
                if (out_d[i] > out_d[worst]) worst = i;
            if (dist < out_d[worst]) { out_d[worst] = dist; out_i[worst] = ids[r]; }
        }
    }
}

void launch_exact_rerank(
    const float*   d_q_batch,
    const float*   d_cand_vecs,
    const int*     d_cand_ids,
    float*         d_exact_dists,
    float*         d_final_dists,
    int*           d_final_ids,
    int B, int R, int d, int k,
    cudaStream_t   s)
{
    // 4 warps (128 threads) per query block; each warp handles R/4 candidates.
    // For R=64: 4 warps × 16 candidates/warp = 64 ✓
    // For R=128: 4 warps × 32 candidates/warp = 128 ✓
    const int BLOCK = 128;
    int smem = R * (int)sizeof(float);
    exact_rerank_kernel<<<B, BLOCK, smem, s>>>(
        d_q_batch, d_cand_vecs, d_cand_ids,
        d_exact_dists, d_final_dists, d_final_ids,
        R, d, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v17
