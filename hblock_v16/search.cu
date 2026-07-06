#include "hblock_v16/search.cuh"
#include "common/cuda_utils.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v16 {

// ── Routing kernels (unchanged from v13) ──────────────────────────────────────

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
        throw std::runtime_error("hblock_v16: nq > batch_cap");

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

void gpu_build_and_sort_pairs(
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

// ── v16: multi-query leaf kernel ──────────────────────────────────────────────
// Grid  = n_leaf_blocks (one block per leaf block, empty ones early-exit)
// Block = leaf_size threads (one thread per vector slot)
// Shared mem: leaf codes loaded ONCE, then looped over all queries in the segment.
// This amortises HBM leaf-code reads: ~15 queries/leaf → ~15× bandwidth reduction.

__global__ void count_leaves_kernel(const int* d_sorted_leaves, int n_pairs,
                                    int* d_leaf_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_pairs) atomicAdd(&d_leaf_counts[d_sorted_leaves[i]], 1);
}

__global__ void multi_query_leaf_kernel(
    const int*     __restrict__ d_seg_offsets,   // [n_leaf_blocks + 1] CSR
    const int*     __restrict__ d_pair_qid_b,    // qids in leaf-sorted order
    const uint8_t* __restrict__ d_leaf_codes,
    const int*     __restrict__ d_leaf_ids_data,
    const int*     __restrict__ d_leaf_sizes,
    const float*   __restrict__ d_lut_fine,      // [B × d × Kr]
    float*         d_out_dists,                  // [n_pairs × TOP_P]
    int*           d_out_ids,
    int d, int Kr, int Br, int bpv, int leaf_size, int top_p)
{
    const float INF = __int_as_float(0x7F800000);
    int leaf_blk  = blockIdx.x;
    int seg_start = d_seg_offsets[leaf_blk];
    int seg_end   = d_seg_offsets[leaf_blk + 1];
    if (seg_start == seg_end) return;   // no query visits this leaf block

    int tid     = threadIdx.x;
    int n_vecs  = d_leaf_sizes[leaf_blk];
    int n_warps = (leaf_size + 31) / 32;

    // Shared memory layout:
    //   s_codes [bpv × leaf_size bytes]   — loaded once from HBM
    //   s_dist  [leaf_size floats]         — reset per query
    //   s_pos   [leaf_size ints]           — set once (tid), never changed
    //   s_wdist [4 floats]                 — warp-min scratch
    //   s_wpos  [4 ints]
    extern __shared__ char shm[];
    uint8_t* s_codes = (uint8_t*)shm;
    float*   s_dist  = (float*)(s_codes + bpv * leaf_size);
    int*     s_pos   = (int*)  (s_dist  + leaf_size);
    float*   s_wdist = (float*)(s_pos   + leaf_size);
    int*     s_wpos  = (int*)  (s_wdist + 4);

    // Load leaf codes into shared memory (amortised across all queries)
    const uint8_t* leaf_base = d_leaf_codes + (long long)leaf_blk * bpv * leaf_size;
    for (int i = tid; i < bpv * leaf_size; i += blockDim.x)
        s_codes[i] = leaf_base[i];
    s_pos[tid] = tid;   // position array: set once, never modified
    __syncthreads();

    // Loop over all queries in this leaf block's segment
    for (int qi_local = 0; qi_local < (seg_end - seg_start); qi_local++) {
        int pair_idx      = seg_start + qi_local;
        int qid           = d_pair_qid_b[pair_idx];
        const float* my_lut = d_lut_fine + (long long)qid * d * Kr;

        // Compute PQ distance for this thread's vector slot
        float fd = INF;
        if (tid < n_vecs) {
            fd = 0.f;
            if (Br == 4) {
                for (int b = 0; b < bpv; ++b) {
                    uint8_t c = s_codes[b * leaf_size + tid];
                    int j0 = b * 2;
                    fd += __ldg(&my_lut[ j0      * Kr + (c & 0x0F)]);
                    fd += __ldg(&my_lut[(j0 + 1) * Kr + (c >> 4  )]);
                }
            } else {
                for (int b = 0; b < bpv; ++b) {
                    uint8_t c = s_codes[b * leaf_size + tid];
                    fd += __ldg(&my_lut[b * Kr + c]);
                }
            }
        }
        s_dist[tid] = fd;
        __syncthreads();

        long long out_base = (long long)pair_idx * top_p;
        for (int r = 0; r < top_p; ++r) {
            float mv = s_dist[tid];
            int   mp = s_pos [tid];
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
                d_out_dists[out_base + r] = bv;
                d_out_ids  [out_base + r] = (bv < INF)
                    ? d_leaf_ids_data[(long long)leaf_blk * leaf_size + bp] : -1;
                if (bv < INF) s_dist[bp] = INF;
            }
            __syncthreads();
        }
    }
}

void gpu_find_segments_and_launch_leaf(
    int n_pairs, int n_leaf_blocks,
    const int*     d_pair_qid_b,
    const uint8_t* d_leaf_codes,
    const int*     d_leaf_ids_data,
    const int*     d_leaf_sizes,
    int d, int Kr, int Br, int bpv, int leaf_size,
    SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;

    // Build per-leaf-block pair counts, then prefix sum → CSR offsets
    CUDA_CHECK(cudaMemsetAsync(ws.d_leaf_counts, 0,
                               (long long)(n_leaf_blocks + 1) * sizeof(int), s));
    count_leaves_kernel<<<(n_pairs + 255) / 256, 256, 0, s>>>(
        ws.d_pair_leaf_b, n_pairs, ws.d_leaf_counts);
    CUDA_CHECK(cudaGetLastError());

    // ExclusiveSum over n_leaf_blocks+1 elements:
    //   d_seg_offsets[0..n_leaf_blocks-1] = exclusive prefix sums
    //   d_seg_offsets[n_leaf_blocks]      = n_pairs  (sentinel, because d_leaf_counts[n_leaf_blocks]=0)
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_leaf_counts, ws.d_seg_offsets, n_leaf_blocks + 1, s));

    // Launch: one block per leaf block, leaf_size threads each
    // s_codes alone = bpv*leaf_size = 384*128 = 49KB > 48KB default limit.
    // Must opt-in to the larger per-block shared memory budget.
    int smem = bpv  * leaf_size  * (int)sizeof(uint8_t)  // s_codes
             + leaf_size         * (int)sizeof(float)     // s_dist
             + leaf_size         * (int)sizeof(int)       // s_pos
             + 4                 * (int)sizeof(float)     // s_wdist
             + 4                 * (int)sizeof(int);      // s_wpos
    CUDA_CHECK(cudaFuncSetAttribute(multi_query_leaf_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    multi_query_leaf_kernel<<<n_leaf_blocks, leaf_size, smem, s>>>(
        ws.d_seg_offsets, d_pair_qid_b,
        d_leaf_codes, d_leaf_ids_data, d_leaf_sizes,
        ws.d_lut_fine,
        ws.d_out_dists, ws.d_out_ids,
        d, Kr, Br, bpv, leaf_size, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

// ── gpu_merge_topk: iota → qid sort → segmented top-k ────────────────────────

__global__ void fill_iota_kernel(int* arr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

// One block per query. blockDim threads split the segment, each maintaining
// a private top-k heap in shared memory. Thread 0 merges all heaps.
// smem layout: [BLOCK × k floats] then [BLOCK × k ints]
// Output stride is k (not K_MAX) so D2H is exactly nq × k elements.
__global__ void segmented_topk_kernel(
    const int*   __restrict__ d_pair_perm,      // [n_pairs] qid-sorted → leaf-sorted idx
    const int*   __restrict__ d_query_offsets,  // [nq]
    const float* __restrict__ d_out_dists,      // [n_pairs × TOP_P]
    const int*   __restrict__ d_out_ids,        // [n_pairs × TOP_P]
    float* d_final_dists,                       // [nq × k]  (stride = k)
    int*   d_final_ids,
    int nq, int k, int n_pairs, int top_p)
{
    const float INF = __int_as_float(0x7F800000);
    int qi  = blockIdx.x;
    int tid = threadIdx.x;
    int BLK = blockDim.x;

    if (qi >= nq) return;

    int seg_start = d_query_offsets[qi];
    // Last query uses n_pairs as seg_end (avoids writing d_query_offsets[nq])
    int seg_end   = (qi + 1 < nq) ? d_query_offsets[qi + 1] : n_pairs;

    extern __shared__ char shm[];
    float* s_dist = (float*)shm          + tid * k;
    int*   s_id   = (int*)((float*)shm + BLK * k) + tid * k;

    for (int i = 0; i < k; i++) { s_dist[i] = INF; s_id[i] = -1; }

    // Each thread processes its strided subset of pairs
    for (int p = seg_start + tid; p < seg_end; p += BLK) {
        int pi = d_pair_perm[p];
        for (int r = 0; r < top_p; r++) {
            float dist = __ldg(&d_out_dists[(long long)pi * top_p + r]);
            int   vid  = __ldg(&d_out_ids  [(long long)pi * top_p + r]);
            if (vid < 0) continue;
            // Max-heap: find worst slot, replace if new dist is smaller
            int worst = 0;
            for (int i = 1; i < k; i++)
                if (s_dist[i] > s_dist[worst]) worst = i;
            if (dist < s_dist[worst]) { s_dist[worst] = dist; s_id[worst] = vid; }
        }
    }
    __syncthreads();

    // Thread 0 merges all per-thread heaps into the global output (stride = k)
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

void gpu_merge_topk(int nq, int n_pairs, int k, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;

    // Fill d_pair_leaf_a with iota: it becomes the sort value (pair indices)
    fill_iota_kernel<<<(n_pairs + 255) / 256, 256, 0, s>>>(ws.d_pair_leaf_a, n_pairs);
    CUDA_CHECK(cudaGetLastError());

    // Radix sort by qid (10 bits: batch_size ≤ 1024 = 2^10).
    // key_in  = d_pair_qid_b  (qids in leaf-sorted order)
    // val_in  = d_pair_leaf_a (iota = original leaf-sorted indices)
    // key_out = d_pair_leaf_b (sorted qids, discarded)
    // val_out = d_pair_qid_a  (permutation: qid-sorted → leaf-sorted pair index)
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_pair_qid_b,  ws.d_pair_leaf_b,
        ws.d_pair_leaf_a, ws.d_pair_qid_a,
        n_pairs, 0, 10, s));

    // Segmented top-k: one block per query, 32 threads split the segment.
    // n_pairs is passed to handle the last query's seg_end (avoids d_query_offsets[nq]).
    // Output stride = k (not K_MAX): D2H is exactly nq × k elements.
    const int BLOCK = 32;
    int smem = BLOCK * k * (int)(sizeof(float) + sizeof(int));
    segmented_topk_kernel<<<nq, BLOCK, smem, s>>>(
        ws.d_pair_qid_a,      // permutation
        ws.d_query_offsets,   // [nq] (nq-th entry accessed via n_pairs param)
        ws.d_out_dists,
        ws.d_out_ids,
        ws.d_final_dists,
        ws.d_final_ids,
        nq, k, n_pairs, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v16
