#include "hblock_v21/search.cuh"
#include "common/cuda_utils.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v21 {

// ── Routing kernels ───────────────────────────────────────────────────────────

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

// L2 beam with bitmask: only non-empty c2 children are considered.
__global__ void route_l2_beam_kernel_v21(
    const float*    __restrict__ q,
    const float*    __restrict__ C1_full,
    const float*    __restrict__ Pi2,
    const float*    __restrict__ C2_proj,
    const float*    __restrict__ C2_norms,
    const int*      __restrict__ top1,
    const uint16_t* __restrict__ d_valid_c2,  // [K1] bitmask per c1
    float*       r1_beam,
    int*         top2_beam,
    int B, int d, int d_proj, int K2, int ck1, int ck2)
{
    int beam = blockIdx.x;
    int qi   = beam / ck1;
    int c1_i = beam % ck1;
    if (qi >= B) return;

    int c1 = top1[qi * ck1 + c1_i];
    if (c1 < 0) {
        int* my_top2 = top2_beam + beam * ck2;
        for (int r = threadIdx.x; r < ck2; r += blockDim.x) my_top2[r] = -1;
        return;
    }

    extern __shared__ float shm_l2[];
    float* s_r1   = shm_l2;
    float* s_proj = s_r1 + d;
    float* s_dist = s_proj + d_proj;

    const float* q_ptr  = q       + (long long)qi * d;
    const float* c1_ptr = C1_full + (long long)c1 * d;
    float*       r1_ptr = r1_beam + (long long)beam * d;

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = q_ptr[j] - c1_ptr[j];
        s_r1[j] = v;
        r1_ptr[j] = v;
    }
    __syncthreads();

    for (int jp = threadIdx.x; jp < d_proj; jp += blockDim.x) {
        float v = 0.f;
        const float* pi_row = Pi2 + (long long)jp * d;
        for (int j = 0; j < d; j++) v += pi_row[j] * s_r1[j];
        s_proj[jp] = v;
    }
    __syncthreads();

    const float INF = __int_as_float(0x7F800000);
    uint16_t valid_mask = d_valid_c2[c1];
    const float* c2_proj_base  = C2_proj  + (long long)c1 * K2 * d_proj;
    const float* c2_norms_base = C2_norms + (long long)c1 * K2;
    for (int c2 = threadIdx.x; c2 < K2; c2 += blockDim.x) {
        if ((valid_mask >> c2) & 1) {
            float dot = 0.f;
            const float* c2_row = c2_proj_base + (long long)c2 * d_proj;
            for (int jp = 0; jp < d_proj; jp++) dot += s_proj[jp] * c2_row[jp];
            s_dist[c2] = c2_norms_base[c2] - 2.f * dot;
        } else {
            s_dist[c2] = INF;  // empty child: never selected
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int* my_top2 = top2_beam + beam * ck2;
        for (int r = 0; r < ck2; r++) {
            float bv = INF; int bi = -1;
            for (int c2 = 0; c2 < K2; c2++)
                if (s_dist[c2] < bv) { bv = s_dist[c2]; bi = c2; }
            my_top2[r] = bi;
            if (bi >= 0) s_dist[bi] = INF;
        }
    }
}

// L3 beam with bitmask: only non-empty c3 children are considered.
__global__ void route_l3_beam_kernel_v21(
    const float*    __restrict__ r1_beam,
    const float*    __restrict__ C2_full,
    const float*    __restrict__ Pi3,
    const float*    __restrict__ C3_proj,
    const float*    __restrict__ C3_norms,
    const int*      __restrict__ top1,
    const int*      __restrict__ top2_beam,
    const uint16_t* __restrict__ d_valid_c3,  // [K1*K2] bitmask per (c1,c2)
    int*         top3_beam,
    int B, int d, int d_proj, int K2, int K3, int ck1, int ck2, int ck3)
{
    int blk  = blockIdx.x;
    int qi   = blk / (ck1 * ck2);
    int rem  = blk % (ck1 * ck2);
    int c1_i = rem / ck2;
    int c2_j = rem % ck2;
    if (qi >= B) return;

    int c1  = top1      [qi * ck1 + c1_i];
    int b1  = qi * ck1 + c1_i;
    int c2  = top2_beam [b1 * ck2 + c2_j];
    if (c1 < 0 || c2 < 0) {
        int* my_top3 = top3_beam + blk * ck3;
        for (int r = threadIdx.x; r < ck3; r += blockDim.x) my_top3[r] = -1;
        return;
    }
    int c12 = c1 * K2 + c2;

    extern __shared__ float shm_l3[];
    float* s_r2   = shm_l3;
    float* s_proj = s_r2 + d;
    float* s_dist = s_proj + d_proj;

    const float* r1_ptr = r1_beam + (long long)b1  * d;
    const float* c2_ptr = C2_full + (long long)c12 * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        s_r2[j] = r1_ptr[j] - c2_ptr[j];
    __syncthreads();

    for (int jp = threadIdx.x; jp < d_proj; jp += blockDim.x) {
        float v = 0.f;
        const float* pi_row = Pi3 + (long long)jp * d;
        for (int j = 0; j < d; j++) v += pi_row[j] * s_r2[j];
        s_proj[jp] = v;
    }
    __syncthreads();

    const float INF = __int_as_float(0x7F800000);
    uint16_t valid_mask = d_valid_c3[c12];
    const float* c3_proj_base  = C3_proj  + (long long)c12 * K3 * d_proj;
    const float* c3_norms_base = C3_norms + (long long)c12 * K3;
    for (int c3 = threadIdx.x; c3 < K3; c3 += blockDim.x) {
        if ((valid_mask >> c3) & 1) {
            float dot = 0.f;
            const float* c3_row = c3_proj_base + (long long)c3 * d_proj;
            for (int jp = 0; jp < d_proj; jp++) dot += s_proj[jp] * c3_row[jp];
            s_dist[c3] = c3_norms_base[c3] - 2.f * dot;
        } else {
            s_dist[c3] = INF;  // empty child: never selected
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int* my_top3 = top3_beam + blk * ck3;
        for (int r = 0; r < ck3; r++) {
            float bv = INF; int bi = -1;
            for (int c3 = 0; c3 < K3; c3++)
                if (s_dist[c3] < bv) { bv = s_dist[c3]; bi = c3; }
            my_top3[r] = bi;
            if (bi >= 0) s_dist[bi] = INF;
        }
    }
}

__global__ void extract_best_r3_kernel(
    const float* __restrict__ q,
    const float* __restrict__ C1_full,
    const float* __restrict__ C2_full,
    const float* __restrict__ C3_full,
    const int*   __restrict__ top1,
    const int*   __restrict__ top2_beam,
    const int*   __restrict__ top3_beam,
    float*       d_q_r3,
    int B, int d, int K2, int K3, int ck1, int ck2, int ck3)
{
    int qi = blockIdx.x;
    if (qi >= B) return;

    int c1   = top1      [qi * ck1 + 0];
    int c2   = top2_beam [(qi * ck1 + 0) * ck2 + 0];
    int c3   = top3_beam [(qi * ck1 * ck2 + 0) * ck3 + 0];
    int c12  = c1 * K2 + c2;
    int c123 = c12 * K3 + c3;

    const float* q_ptr  = q       + (long long)qi   * d;
    const float* c1_ptr = C1_full + (long long)c1   * d;
    const float* c2_ptr = C2_full + (long long)c12  * d;
    const float* c3_ptr = C3_full + (long long)c123 * d;
    float*       r3_ptr = d_q_r3  + (long long)qi   * d;

    for (int j = threadIdx.x; j < d; j += blockDim.x)
        r3_ptr[j] = q_ptr[j] - c1_ptr[j] - c2_ptr[j] - c3_ptr[j];
}

__global__ void gather_leaf_blocks_hierarchical_kernel(
    const int* __restrict__ top1_ids,
    const int* __restrict__ top2_beam,
    const int* __restrict__ top3_beam,
    const int* __restrict__ pair_start,
    const int* __restrict__ pair_cnt,
    int* leaf_sel, int* leaf_cnt,
    int B, int ck1, int ck2, int ck3, int K2, int K3,
    int max_leaf_sel)
{
    int bqi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bqi >= B) return;
    int* my_sel = leaf_sel + bqi * max_leaf_sel;
    int cnt = 0;
    for (int i1 = 0; i1 < ck1; ++i1) {
        int c1 = top1_ids[bqi * ck1 + i1];
        if (c1 < 0) continue;
        int b1 = bqi * ck1 + i1;
        for (int i2 = 0; i2 < ck2; ++i2) {
            int c2 = top2_beam[b1 * ck2 + i2];
            if (c2 < 0) continue;
            int b2 = b1 * ck2 + i2;
            for (int i3 = 0; i3 < ck3; ++i3) {
                int c3 = top3_beam[b2 * ck3 + i3];
                if (c3 < 0) continue;
                int pidx = c1 * K2 * K3 + c2 * K3 + c3;
                int nblk = pair_cnt[pidx];
                int base = pair_start[pidx];
                for (int b = 0; b < nblk && cnt < max_leaf_sel; ++b)
                    my_sel[cnt++] = base + b;
            }
        }
    }
    leaf_cnt[bqi] = cnt;
}

__global__ void build_fine_lut_kernel(
    const float* __restrict__ d_q_r3, const float* __restrict__ d_c1d,
    float* d_lut_fine, int B, int d, int Kr)
{
    long long total = (long long)B * d * Kr;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        float diff = d_q_r3[(long long)bqi * d + local / Kr] - d_c1d[local % Kr];
        d_lut_fine[i] = diff * diff;
    }
}

void route_queries_v21(
    cublasHandle_t cublas,
    const float*    d_Pi1,
    const float*    d_Pi2,
    const float*    d_Pi3,
    const float*    d_route1_cents_proj, const float* d_route1_cents_full, const float* d_route1_norms,
    const float*    d_route2_cents_proj, const float* d_route2_cents_full, const float* d_route2_norms,
    const float*    d_route3_cents_proj, const float* d_route3_cents_full, const float* d_route3_norms,
    const float*    d_fine_c1d,
    const int*      d_pair_blk_start,
    const int*      d_pair_blk_count,
    const uint16_t* d_valid_c2,
    const uint16_t* d_valid_c3,
    const float*    h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int K3, int Kr, int Br,
    int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws)
{
    if (nq > ws.batch_cap)
        throw std::runtime_error("hblock_v21: nq > batch_cap");

    const int B      = ws.batch_cap;
    const int BLOCK  = 128;
    const float one  = 1.f, zero = 0.f;
    cudaStream_t s   = ws.stream;

    std::memcpy(ws.h_q_pinned, h_queries, (long long)nq * d * sizeof(float));
    if (nq < B)
        std::memset(ws.h_q_pinned + (long long)nq * d, 0,
                    (long long)(B - nq) * d * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                               (long long)B * d * sizeof(float), cudaMemcpyHostToDevice, s));

    // L1: global routing (no bitmask at L1 — all K1 centroids always valid)
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_proj, B, d, &one,
                             d_Pi1, d, ws.d_q_batch, d, &zero,
                             ws.d_q_proj1, d_proj));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K1, B, d_proj, &one,
                             d_route1_cents_proj, d_proj,
                             ws.d_q_proj1, d_proj, &zero,
                             ws.d_dots1, K1));
    {
        int shm1 = (K1 + 2 * BLOCK) * (int)sizeof(float);
        select_route_topk_kernel<<<B, BLOCK, shm1, s>>>(
            ws.d_dots1, d_route1_norms, ws.d_top1_ids, K1, ck1);
        CUDA_CHECK(cudaGetLastError());
    }

    // L2 beam with bitmask
    {
        int shm_l2 = (d + d_proj + K2) * (int)sizeof(float);
        route_l2_beam_kernel_v21<<<B * ck1, BLOCK, shm_l2, s>>>(
            ws.d_q_batch, d_route1_cents_full, d_Pi2,
            d_route2_cents_proj, d_route2_norms,
            ws.d_top1_ids, d_valid_c2,
            ws.d_r1_beam, ws.d_top2_beam,
            B, d, d_proj, K2, ck1, ck2);
        CUDA_CHECK(cudaGetLastError());
    }

    // L3 beam with bitmask
    {
        int shm_l3 = (d + d_proj + K3) * (int)sizeof(float);
        route_l3_beam_kernel_v21<<<B * ck1 * ck2, BLOCK, shm_l3, s>>>(
            ws.d_r1_beam, d_route2_cents_full, d_Pi3,
            d_route3_cents_proj, d_route3_norms,
            ws.d_top1_ids, ws.d_top2_beam, d_valid_c3,
            ws.d_top3_beam,
            B, d, d_proj, K2, K3, ck1, ck2, ck3);
        CUDA_CHECK(cudaGetLastError());
    }

    extract_best_r3_kernel<<<B, BLOCK, 0, s>>>(
        ws.d_q_batch, d_route1_cents_full, d_route2_cents_full, d_route3_cents_full,
        ws.d_top1_ids, ws.d_top2_beam, ws.d_top3_beam,
        ws.d_q_r3, B, d, K2, K3, ck1, ck2, ck3);
    CUDA_CHECK(cudaGetLastError());

    gather_leaf_blocks_hierarchical_kernel<<<(B + 63) / 64, 64, 0, s>>>(
        ws.d_top1_ids, ws.d_top2_beam, ws.d_top3_beam,
        d_pair_blk_start, d_pair_blk_count,
        ws.d_leaf_sel, ws.d_leaf_cnt,
        B, ck1, ck2, ck3, K2, K3, ws.max_leaf_sel);
    CUDA_CHECK(cudaGetLastError());

    {
        const int BLUT = 256;
        long long tot  = (long long)B * d * Kr;
        int grid = (int)std::min((tot + BLUT - 1) / BLUT, (long long)65535);
        build_fine_lut_kernel<<<grid, BLUT, 0, s>>>(
            ws.d_q_r3, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpyAsync(ws.h_leaf_cnt, ws.d_leaf_cnt,
                               (long long)nq * sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ── Remaining functions identical to v17 ─────────────────────────────────────

__global__ void build_pairs_kernel(
    const int* __restrict__ d_leaf_sel,
    const int* __restrict__ d_leaf_cnt,
    const int* __restrict__ d_query_offsets,
    int* d_pair_leaf_ids,
    int* d_pair_qids,
    int n_leaf_blocks, int max_leaf_sel, int nq)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq) return;
    int cnt = d_leaf_cnt[qi];
    int off = d_query_offsets[qi];
    for (int s = 0; s < cnt; s++) {
        int lb = d_leaf_sel[qi * max_leaf_sel + s];
        if (lb >= 0 && lb < n_leaf_blocks) {
            d_pair_leaf_ids[off + s] = lb;
            d_pair_qids    [off + s] = qi;
        }
    }
}

void gpu_build_and_sort_pairs_v21(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_leaf_cnt, ws.d_query_offsets, nq, s));

    build_pairs_kernel<<<(nq + 255) / 256, 256, 0, s>>>(
        ws.d_leaf_sel, ws.d_leaf_cnt, ws.d_query_offsets,
        ws.d_pair_leaf_a, ws.d_pair_qid_a,
        n_leaf_blocks, max_leaf_sel, nq);
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

__global__ void leaf_flat_kernel_v21(
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

void launch_leaf_flat_v21(
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
    leaf_flat_kernel_v21<<<n_pairs, leaf_size, smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_leaf_sizes, d_leaf_codes, d_leaf_ids_data,
        d_lut_fine, d_out_dists, d_out_ids,
        d, Kr, Br, bpv, leaf_size, TOP_P);
    CUDA_CHECK(cudaGetLastError());
}

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

void gpu_merge_pq_topk_v21(int nq, int n_pairs, int rerank_r, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    int k = std::min(rerank_r, (int)K_MAX);

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

void launch_gather_vecs_v21(
    const float*   d_base_vecs,
    const int*     d_cand_ids,
    float*         d_cand_vecs,
    int B, int R, int d,
    cudaStream_t   s)
{
    dim3 grid(R, B);
    gather_vecs_kernel<<<grid, 128, 0, s>>>(d_base_vecs, d_cand_ids, d_cand_vecs, R, d);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void exact_rerank_kernel(
    const float* __restrict__ d_q_batch,
    const float* __restrict__ d_cand_vecs,
    const int*   __restrict__ d_cand_ids,
    float*                    d_exact_dists,
    float*                    d_final_dists,
    int*                      d_final_ids,
    int R, int d, int k)
{
    const float INF = __int_as_float(0x7F800000);
    int qi    = blockIdx.x;
    int tid   = threadIdx.x;
    int lane  = tid & 31;
    int wid   = tid >> 5;
    int n_warps = blockDim.x >> 5;

    const float* q   = d_q_batch  + (long long)qi * d;
    const int*   ids = d_cand_ids + (long long)qi * R;

    extern __shared__ char shm_rerank[];
    float* s_dots = (float*)shm_rerank;

    for (int ri = wid; ri < R; ri += n_warps) {
        float l2sq = 0.f;
        if (ids[ri] >= 0) {
            const float* c = d_cand_vecs + ((long long)qi * R + ri) * d;
            for (int j = lane; j < d; j += 32) {
                float diff = q[j] - c[j];
                l2sq += diff * diff;
            }
        }
        for (int mask = 16; mask > 0; mask >>= 1)
            l2sq += __shfl_xor_sync(0xffffffff, l2sq, mask);
        if (lane == 0) s_dots[ri] = l2sq;
    }
    __syncthreads();

    if (tid == 0) {
        float* ed  = d_exact_dists + (long long)qi * R;
        float* out_d = d_final_dists + (long long)qi * k;
        int*   out_i = d_final_ids   + (long long)qi * k;
        for (int i = 0; i < k; i++) { out_d[i] = INF; out_i[i] = -1; }

        for (int r = 0; r < R; ++r) {
            if (ids[r] < 0) { ed[r] = INF; continue; }
            float dist = s_dots[r];
            ed[r] = dist;
            int worst = 0;
            for (int i = 1; i < k; i++)
                if (out_d[i] > out_d[worst]) worst = i;
            if (dist < out_d[worst]) { out_d[worst] = dist; out_i[worst] = ids[r]; }
        }
    }
}

void launch_exact_rerank_v21(
    const float*   d_q_batch,
    const float*   d_cand_vecs,
    const int*     d_cand_ids,
    float*         d_exact_dists,
    float*         d_final_dists,
    int*           d_final_ids,
    int B, int R, int d, int k,
    cudaStream_t   s)
{
    const int BLOCK = 128;
    int smem = R * (int)sizeof(float);
    exact_rerank_kernel<<<B, BLOCK, smem, s>>>(
        d_q_batch, d_cand_vecs, d_cand_ids,
        d_exact_dists, d_final_dists, d_final_ids,
        R, d, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v21
