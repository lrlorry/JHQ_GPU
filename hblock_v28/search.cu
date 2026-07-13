// hblock_v28/search.cu
// Per-block exact rerank: PQ filter → exact L2 within block (using d_base_vecs) → merge.
// Eliminates cross-block PQ ranking errors; oracle recall ≈ actual recall.
#include "hblock_v28/search.cuh"
#include "hblock_v27/search.cuh"   // for gpu_build_block_adj_v27
#include "common/cuda_utils.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v28 {

// ── Routing kernels (identical to v27, different namespace) ───────────────────

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

__global__ void route_l2_beam_kernel(
    const float* __restrict__ q, const float* __restrict__ C1_full,
    const float* __restrict__ Pi2, const float* __restrict__ C2_proj,
    const float* __restrict__ C2_norms, const int* __restrict__ top1,
    float* r1_beam, int* top2_beam,
    int B, int d, int d_proj, int K2, int ck1, int ck2)
{
    int beam = blockIdx.x, qi = beam / ck1, c1_i = beam % ck1;
    if (qi >= B) return;
    int c1 = top1[qi * ck1 + c1_i];
    if (c1 < 0) {
        int* my_top2 = top2_beam + beam * ck2;
        for (int r = threadIdx.x; r < ck2; r += blockDim.x) my_top2[r] = -1;
        return;
    }
    extern __shared__ float shm_l2[];
    float* s_r1 = shm_l2, *s_proj = s_r1 + d, *s_dist = s_proj + d_proj;
    const float* q_ptr  = q       + (long long)qi * d;
    const float* c1_ptr = C1_full + (long long)c1 * d;
    float*       r1_ptr = r1_beam + (long long)beam * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = q_ptr[j] - c1_ptr[j]; s_r1[j] = v; r1_ptr[j] = v;
    }
    __syncthreads();
    for (int jp = threadIdx.x; jp < d_proj; jp += blockDim.x) {
        float v = 0.f;
        const float* pi_row = Pi2 + (long long)jp * d;
        for (int j = 0; j < d; j++) v += pi_row[j] * s_r1[j];
        s_proj[jp] = v;
    }
    __syncthreads();
    const float* c2_proj_base  = C2_proj  + (long long)c1 * K2 * d_proj;
    const float* c2_norms_base = C2_norms + (long long)c1 * K2;
    for (int c2 = threadIdx.x; c2 < K2; c2 += blockDim.x) {
        float dot = 0.f;
        const float* c2_row = c2_proj_base + (long long)c2 * d_proj;
        for (int jp = 0; jp < d_proj; jp++) dot += s_proj[jp] * c2_row[jp];
        s_dist[c2] = c2_norms_base[c2] - 2.f * dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float INF = __int_as_float(0x7F800000);
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

__global__ void route_l3_beam_kernel(
    const float* __restrict__ r1_beam, const float* __restrict__ C2_full,
    const float* __restrict__ Pi3, const float* __restrict__ C3_proj,
    const float* __restrict__ C3_norms, const int* __restrict__ top1,
    const int* __restrict__ top2_beam, int* top3_beam,
    int B, int d, int d_proj, int K2, int K3, int ck1, int ck2, int ck3)
{
    int blk = blockIdx.x, qi = blk / (ck1*ck2), rem = blk % (ck1*ck2);
    int c1_i = rem / ck2, c2_j = rem % ck2;
    if (qi >= B) return;
    int c1 = top1[qi*ck1+c1_i], b1 = qi*ck1+c1_i;
    int c2 = top2_beam[b1*ck2+c2_j];
    if (c1 < 0 || c2 < 0) {
        int* my_top3 = top3_beam + blk * ck3;
        for (int r = threadIdx.x; r < ck3; r += blockDim.x) my_top3[r] = -1;
        return;
    }
    int c12 = c1*K2+c2;
    extern __shared__ float shm_l3[];
    float* s_r2 = shm_l3, *s_proj = s_r2+d, *s_dist = s_proj+d_proj;
    const float* r1_ptr = r1_beam + (long long)b1 * d;
    const float* c2_ptr = C2_full + (long long)c12 * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) s_r2[j] = r1_ptr[j] - c2_ptr[j];
    __syncthreads();
    for (int jp = threadIdx.x; jp < d_proj; jp += blockDim.x) {
        float v = 0.f;
        const float* pi_row = Pi3 + (long long)jp * d;
        for (int j = 0; j < d; j++) v += pi_row[j] * s_r2[j];
        s_proj[jp] = v;
    }
    __syncthreads();
    const float* c3_proj_base  = C3_proj  + (long long)c12 * K3 * d_proj;
    const float* c3_norms_base = C3_norms + (long long)c12 * K3;
    for (int c3 = threadIdx.x; c3 < K3; c3 += blockDim.x) {
        float dot = 0.f;
        const float* c3_row = c3_proj_base + (long long)c3 * d_proj;
        for (int jp = 0; jp < d_proj; jp++) dot += s_proj[jp] * c3_row[jp];
        s_dist[c3] = c3_norms_base[c3] - 2.f * dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float INF = __int_as_float(0x7F800000);
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
    const float* __restrict__ C1_full, const float* __restrict__ C2_full,
    const float* __restrict__ C3_full,
    const int* __restrict__ top1, const int* __restrict__ top2_beam,
    const int* __restrict__ top3_beam,
    float* d_q_r3,
    int B, int d, int K2, int K3, int ck1, int ck2, int ck3)
{
    int qi = blockIdx.x; if (qi >= B) return;
    int c1  = top1      [qi * ck1 + 0];
    int c2  = top2_beam [(qi * ck1 + 0) * ck2 + 0];
    int c3  = top3_beam [(qi * ck1 * ck2 + 0) * ck3 + 0];
    int c12 = c1*K2+c2, c123 = c12*K3+c3;
    const float* q_ptr  = q       + (long long)qi   * d;
    const float* c1_ptr = C1_full + (long long)c1   * d;
    const float* c2_ptr = C2_full + (long long)c12  * d;
    const float* c3_ptr = C3_full + (long long)c123 * d;
    float* r3_ptr = d_q_r3 + (long long)qi * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        r3_ptr[j] = q_ptr[j] - c1_ptr[j] - c2_ptr[j] - c3_ptr[j];
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

void route_gpu_v28(
    cublasHandle_t cublas,
    const float* d_Pi1, const float* d_Pi2, const float* d_Pi3,
    const float* d_route1_cents_proj, const float* d_route1_cents_full, const float* d_route1_norms,
    const float* d_route2_cents_proj, const float* d_route2_cents_full, const float* d_route2_norms,
    const float* d_route3_cents_proj, const float* d_route3_cents_full, const float* d_route3_norms,
    const float* d_fine_c1d,
    const float* h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int K3, int Kr,
    int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws)
{
    if (nq > ws.batch_cap) throw std::runtime_error("hblock_v28: nq > batch_cap");
    const int B = ws.batch_cap, BLOCK = 128;
    const float one = 1.f, zero = 0.f;
    cudaStream_t s = ws.stream;

    std::memcpy(ws.h_q_pinned, h_queries, (long long)nq * d * sizeof(float));
    if (nq < B) std::memset(ws.h_q_pinned + (long long)nq*d, 0, (long long)(B-nq)*d*sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                               (long long)B*d*sizeof(float), cudaMemcpyHostToDevice, s));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_proj, B, d, &one, d_Pi1, d, ws.d_q_batch, d, &zero, ws.d_q_proj1, d_proj));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K1, B, d_proj, &one, d_route1_cents_proj, d_proj, ws.d_q_proj1, d_proj, &zero, ws.d_dots1, K1));
    {
        int shm1 = (K1 + 2*BLOCK) * (int)sizeof(float);
        select_route_topk_kernel<<<B, BLOCK, shm1, s>>>(ws.d_dots1, d_route1_norms, ws.d_top1_ids, K1, ck1);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int shm_l2 = (d + d_proj + K2) * (int)sizeof(float);
        route_l2_beam_kernel<<<B*ck1, BLOCK, shm_l2, s>>>(
            ws.d_q_batch, d_route1_cents_full, d_Pi2,
            d_route2_cents_proj, d_route2_norms,
            ws.d_top1_ids, ws.d_r1_beam, ws.d_top2_beam,
            B, d, d_proj, K2, ck1, ck2);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int shm_l3 = (d + d_proj + K3) * (int)sizeof(float);
        route_l3_beam_kernel<<<B*ck1*ck2, BLOCK, shm_l3, s>>>(
            ws.d_r1_beam, d_route2_cents_full, d_Pi3,
            d_route3_cents_proj, d_route3_norms,
            ws.d_top1_ids, ws.d_top2_beam, ws.d_top3_beam,
            B, d, d_proj, K2, K3, ck1, ck2, ck3);
        CUDA_CHECK(cudaGetLastError());
    }
    extract_best_r3_kernel<<<B, BLOCK, 0, s>>>(
        ws.d_q_batch, d_route1_cents_full, d_route2_cents_full, d_route3_cents_full,
        ws.d_top1_ids, ws.d_top2_beam, ws.d_top3_beam,
        ws.d_q_r3, B, d, K2, K3, ck1, ck2, ck3);
    CUDA_CHECK(cudaGetLastError());
    {
        const int BLUT = 256;
        long long tot = (long long)B * d * Kr;
        int grid = (int)std::min((tot + BLUT - 1) / BLUT, (long long)65535);
        build_fine_lut_kernel<<<grid, BLUT, 0, s>>>(ws.d_q_r3, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ── Pair build (identical to v27) ─────────────────────────────────────────────

__global__ void build_pairs_kernel_v28(
    const int* __restrict__ d_leaf_sel,
    const int* __restrict__ d_leaf_cnt,
    const int* __restrict__ d_query_offsets,
    int* d_pair_leaf_ids, int* d_pair_qids,
    int n_leaf_blocks, int max_leaf_sel, int nq)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq) return;
    int cnt = d_leaf_cnt[qi], off = d_query_offsets[qi];
    for (int s = 0; s < cnt; s++) {
        int lb = d_leaf_sel[qi * max_leaf_sel + s];
        if (lb >= 0 && lb < n_leaf_blocks) {
            d_pair_leaf_ids[off + s] = lb;
            d_pair_qids    [off + s] = qi;
        }
    }
}

void gpu_build_and_sort_pairs_v28(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_cub_tmp, ws.cub_bytes, ws.d_leaf_cnt, ws.d_query_offsets, nq, s));
    build_pairs_kernel_v28<<<(nq+255)/256, 256, 0, s>>>(
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

// ── Beam search (identical to v27, different namespace) ───────────────────────

__device__ __forceinline__ float
blk_qdist_v28(const float* __restrict__ s_q, float q_norm,
              const float* __restrict__ d_blk_proj,
              const float* __restrict__ d_blk_norm,
              int b, int d_proj)
{
    const float* pb = d_blk_proj + (long long)b * d_proj;
    float dot = 0.f;
    for (int jp = 0; jp < d_proj; jp++) dot += s_q[jp] * pb[jp];
    return q_norm + d_blk_norm[b] - 2.f * dot;
}

__device__ __forceinline__ bool
try_visit_v28(int* __restrict__ vis, int b, int n_blks)
{
    if (b < 0 || b >= n_blks) return false;
    int w = b >> 5, bit = b & 31;
    return !((atomicOr(&vis[w], 1 << bit) >> bit) & 1);
}

template <int SPT>
__global__ void block_search_fused_v28(
    const int*   __restrict__ d_top1,  const int*   __restrict__ d_top2,
    const int*   __restrict__ d_top3,
    const int*   __restrict__ d_blk_start, const int* __restrict__ d_blk_count,
    const int*   __restrict__ d_block_adj,
    const float* __restrict__ d_blk_proj, const float* __restrict__ d_blk_norm,
    const float* __restrict__ d_q_proj1,
    int* d_leaf_sel, int* d_leaf_cnt, int* d_visited,
    int n_blks, int d_proj, int K2, int K3,
    int ck1, int ck2, int ck3,
    int degree, int budget, int max_ls,
    int entry_per_cell, int bitmap_words)
{
    const int qi = blockIdx.x, tid = threadIdx.x;
    constexpr unsigned FULL = 0xffffffff;
    extern __shared__ float s_q[];
    for (int jp = tid; jp < d_proj; jp += 32)
        s_q[jp] = d_q_proj1[(long long)qi * d_proj + jp];
    __syncwarp();
    float q_norm = 0.f;
    for (int jp = 0; jp < d_proj; jp++) q_norm += s_q[jp] * s_q[jp];

    int* vis = d_visited + (long long)qi * bitmap_words;
    for (int w = tid; w < bitmap_words; w += 32) vis[w] = 0;
    __syncwarp();

    float my_dist[SPT]; int my_id[SPT]; bool my_exp[SPT];
    for (int s = 0; s < SPT; s++) { my_dist[s] = 1e38f; my_id[s] = -1; my_exp[s] = true; }

    auto local_worst = [&](float& lw, int& ls) {
        lw = my_dist[0]; ls = 0;
        for (int s = 1; s < SPT; s++) if (my_dist[s] > lw) { lw = my_dist[s]; ls = s; }
    };

    auto try_insert = [&](float nd, int ni) {
        float lw; int ls; local_worst(lw, ls);
        float bmax = lw;
        for (int o = 16; o >= 1; o >>= 1) bmax = fmaxf(bmax, __shfl_xor_sync(FULL, bmax, o));
        if (nd >= bmax) return;
        unsigned em = __ballot_sync(FULL, lw == bmax);
        int el = __ffs(em) - 1;
        if (tid == el) { my_dist[ls] = nd; my_id[ls] = ni; my_exp[ls] = false; }
    };

    for (int i1 = 0; i1 < ck1; i1++) {
        int c1 = d_top1[qi*ck1+i1]; if (c1 < 0) continue;
        for (int i2 = 0; i2 < ck2; i2++) {
            int c2 = d_top2[qi*ck1*ck2+i1*ck2+i2]; if (c2 < 0) continue;
            for (int i3 = 0; i3 < ck3; i3++) {
                int c3 = d_top3[qi*(ck1*ck2*ck3)+(i1*ck2+i2)*ck3+i3]; if (c3 < 0) continue;
                int c123 = c1*K2*K3+c2*K3+c3;
                int bs = d_blk_start[c123], bc = d_blk_count[c123];
                if (bc == 0) continue;
                for (int start = 0; start < bc; start += 32) {
                    int bi = start + tid;
                    float ck_d = 1e38f; int ck_id = -1;
                    if (bi < bc) {
                        ck_d  = blk_qdist_v28(s_q, q_norm, d_blk_proj, d_blk_norm, bs+bi, d_proj);
                        ck_id = bs + bi;
                    }
                    for (int ep = 0; ep < entry_per_cell; ep++) {
                        float mn = ck_d;
                        for (int o = 16; o >= 1; o >>= 1) mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
                        if (mn >= 1e37f) break;
                        unsigned wmask = __ballot_sync(FULL, ck_d == mn);
                        int wlane = __ffs(wmask) - 1;
                        int ins_id = __shfl_sync(FULL, ck_id, wlane);
                        if (tid == wlane) { ck_d = 1e38f; ck_id = -1; }
                        bool fv = false;
                        if (tid == 0 && ins_id >= 0) fv = try_visit_v28(vis, ins_id, n_blks);
                        if ((bool)__shfl_sync(FULL, (int)fv, 0)) try_insert(mn, ins_id);
                    }
                }
            }
        }
    }

    int out_cnt = 0;
    for (int iter = 0; iter < budget; iter++) {
        float lb = 1e38f; int lbs = -1;
        for (int s = 0; s < SPT; s++)
            if (!my_exp[s] && my_id[s] >= 0 && my_dist[s] < lb) { lb = my_dist[s]; lbs = s; }
        float mn = lb;
        for (int o = 16; o >= 1; o >>= 1) mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
        if (mn >= 1e37f) break;
        unsigned wm = __ballot_sync(FULL, lb == mn);
        int win_lane = __ffs(wm) - 1;
        int expand_id = -1;
        if (tid == win_lane) { my_exp[lbs] = true; expand_id = my_id[lbs]; }
        expand_id = __shfl_sync(FULL, expand_id, win_lane);
        if (tid == 0 && out_cnt < max_ls) d_leaf_sel[(long long)qi*max_ls+out_cnt] = expand_id;
        out_cnt++;
        float new_d = 1e38f; int new_id = -1;
        if (tid < degree) {
            int nb = d_block_adj[(long long)expand_id*degree+tid];
            if (try_visit_v28(vis, nb, n_blks)) {
                new_d  = blk_qdist_v28(s_q, q_norm, d_blk_proj, d_blk_norm, nb, d_proj);
                new_id = nb;
            }
        }
        for (int src = 0; src < 32; src++) {
            float nd = __shfl_sync(FULL, new_d,  src);
            int   ni = __shfl_sync(FULL, new_id, src);
            if (ni >= 0) try_insert(nd, ni);
        }
    }
    if (tid == 0) d_leaf_cnt[qi] = out_cnt;
}

#define LAUNCH_BEAM_V28(SPT_VAL) \
    block_search_fused_v28<SPT_VAL><<<B, 32, smem, ws.stream>>>( \
        ws.d_top1_ids, ws.d_top2_beam, ws.d_top3_beam, \
        d_pair_blk_start, d_pair_blk_count, \
        d_block_adj, d_blk_proj, d_blk_norm, \
        ws.d_q_proj1, \
        ws.d_leaf_sel, ws.d_leaf_cnt, ws.d_visited, \
        n_blks, d_proj, K2, K3, \
        ck1, ck2, ck3, degree, budget, max_ls, entry_per_cell, ws.bitmap_words)

void gpu_block_search_v27(  // name kept for linkage compatibility
    int B, int n_blks, int d_proj,
    int K2, int K3, int ck1, int ck2, int ck3,
    int degree, int budget, int max_ls, int entry_per_cell,
    const int*   d_block_adj, const float* d_blk_proj, const float* d_blk_norm,
    const int*   d_pair_blk_start, const int*   d_pair_blk_count,
    SearchWorkspace& ws)
{
    if (degree > 32) throw std::runtime_error("gpu_block_search: graph_degree must be <= 32");
    int smem = d_proj * (int)sizeof(float);
    if      (ws.beam_size <= 32) { LAUNCH_BEAM_V28(1); }
    else if (ws.beam_size <= 64) { LAUNCH_BEAM_V28(2); }
    else                         { LAUNCH_BEAM_V28(4); }
    CUDA_CHECK(cudaGetLastError());
}
#undef LAUNCH_BEAM_V28

// ── v28 key kernel: per-block PQ filter → exact L2 → top klocal ─────────────
//
// Phase 1: PQ distances (128 threads, one per vector)
// Phase 2: Bitonic warp sort (4 warps, ascending)
// Phase 3: Load query into smem
// Phase 4: Thread-0 k-way merge → top per_block_r candidate IDs in s_sel
// Phase 5: For each candidate, all 128 threads compute exact L2;
//           thread-0 keeps top klocal by exact dist in s_exact
// Phase 6: Thread-0 writes top klocal to output
//
// smem layout (per_block_r ≤ 32):
//   float s_dist[leaf_size]        (PQ sort output)
//   int   s_id  [leaf_size]        (PQ sort output)
//   float s_query[d]               (query vector, loaded once)
//   float s_wsum[4]                (warp partial sums for exact L2 reduce)
//   int   s_sel [per_block_r]      (PQ-selected candidate IDs)
//   float s_exact[per_block_r]     (exact L2 for each candidate)

__global__ void leaf_flat_kernel_v28(
    const int*     __restrict__ d_pair_leaf_ids,
    const int*     __restrict__ d_pair_qids,
    const int*     __restrict__ leaf_sizes,
    const uint8_t* __restrict__ leaf_codes,
    const int*     __restrict__ leaf_ids_data,
    const float*   __restrict__ lut_fine,
    const float*   __restrict__ d_base_vecs,
    const float*   __restrict__ d_q_batch,
    float* d_out_dists, int* d_out_ids,
    int d, int Kr, int Br, int bpv, int leaf_size,
    int per_block_r, int klocal)
{
    const float INF = __int_as_float(0x7F800000);
    constexpr unsigned FULL = 0xffffffff;
    const int pi   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int n_warps = leaf_size >> 5;  // = 4 for leaf_size=128

    const int leaf_blk = d_pair_leaf_ids[pi];
    const int qid      = d_pair_qids[pi];
    const int n_vecs   = leaf_sizes[leaf_blk];

    const float*   my_lut    = lut_fine   + (long long)qid      * d * Kr;
    const uint8_t* leaf_base = leaf_codes + (long long)leaf_blk * bpv * leaf_size;

    // ── Phase 1: PQ distances ─────────────────────────────────────────────────
    float my_dist = INF; int my_id = -1;
    if (tid < n_vecs) {
        my_dist = 0.f;
        if (Br == 4) {
            for (int b = 0; b < bpv; ++b) {
                uint8_t c = __ldg(&leaf_base[b * leaf_size + tid]);
                int j0 = b * 2;
                my_dist += __ldg(&my_lut[ j0      * Kr + (c & 0x0F)]);
                my_dist += __ldg(&my_lut[(j0 + 1) * Kr + (c >> 4  )]);
            }
        } else {
            for (int b = 0; b < bpv; ++b) {
                uint8_t c = __ldg(&leaf_base[b * leaf_size + tid]);
                my_dist += __ldg(&my_lut[b * Kr + c]);
            }
        }
        my_id = leaf_ids_data[(long long)leaf_blk * leaf_size + tid];
    }

    // ── Phase 2: Bitonic warp sort (ascending, lane 0 = min dist) ────────────
    for (int k = 2; k <= 32; k <<= 1) {
        bool asc = ((lane & k) == 0);
        for (int j = k >> 1; j > 0; j >>= 1) {
            float od = __shfl_xor_sync(FULL, my_dist, j);
            int   oi = __shfl_xor_sync(FULL, my_id,   j);
            bool lower    = ((lane & j) == 0);
            bool want_min = (asc == lower);
            bool swap     = want_min ? (my_dist > od) : (my_dist < od);
            if (swap) { my_dist = od; my_id = oi; }
        }
    }

    // Write sorted warps to smem
    extern __shared__ char shm[];
    float* s_dist  = (float*)shm;
    int*   s_id    = (int*)(s_dist + leaf_size);
    float* s_query = (float*)(s_id + leaf_size);
    float* s_wsum  = s_query + d;                      // [4] warp partial sums
    int*   s_sel   = (int*)(s_wsum + 4);               // [per_block_r] selected IDs
    float* s_exact = (float*)(s_sel + per_block_r);    // [per_block_r] exact dists

    s_dist[tid] = my_dist;
    s_id  [tid] = my_id;
    __syncthreads();

    // ── Phase 3: Load query vector into smem ──────────────────────────────────
    const float* q_ptr = d_q_batch + (long long)qid * d;
    for (int j = tid; j < d; j += blockDim.x)
        s_query[j] = q_ptr[j];
    __syncthreads();

    // ── Phase 4: Thread-0 k-way merge → top per_block_r → s_sel ─────────────
    if (tid == 0) {
        int ptrs[4] = {0, 32, 64, 96};
        int actual_r = (per_block_r < n_vecs) ? per_block_r : n_vecs;
        for (int r = 0; r < actual_r; r++) {
            float best_d = INF; int best_w = -1;
            for (int w = 0; w < n_warps; w++) {
                int p = ptrs[w];
                if (p < (w+1)*32 && s_dist[p] < best_d) { best_d = s_dist[p]; best_w = w; }
            }
            s_sel[r] = (best_w >= 0 && best_d < INF) ? s_id[ptrs[best_w]] : -1;
            if (best_w >= 0) ptrs[best_w]++;
        }
        for (int r = actual_r; r < per_block_r; r++) s_sel[r] = -1;
    }
    __syncthreads();  // ensure s_sel ready for all threads

    // ── Phase 5: Exact L2 for each of per_block_r candidates ─────────────────
    // All 128 threads cooperate for each candidate (d/128 loads per thread).
    // 2 syncs per iteration: (1) protect s_wsum from prev iter; (2) ensure written.
    for (int r = 0; r < per_block_r; r++) {
        int cand_id = s_sel[r];
        float partial = 0.f;
        if (cand_id >= 0) {
            const float* vec = d_base_vecs + (long long)cand_id * d;
            for (int j = tid; j < d; j += blockDim.x) {
                float diff = s_query[j] - vec[j];
                partial += diff * diff;
            }
        }
        // Warp reduce
        for (int o = 16; o >= 1; o >>= 1)
            partial += __shfl_xor_sync(FULL, partial, o);

        // Sync 1: ensures thread-0 has finished reading s_wsum from prev iteration
        __syncthreads();
        if (lane == 0) s_wsum[wid] = partial;
        // Sync 2: ensures all 4 warp sums are written before thread-0 reads
        __syncthreads();

        if (tid == 0) {
            float total = s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3];
            s_exact[r] = (cand_id >= 0) ? total : INF;
        }
    }
    __syncthreads();  // ensure s_exact fully written

    // ── Phase 6: Thread-0 selects top klocal from per_block_r exact results ──
    if (tid == 0) {
        long long out_base = (long long)pi * klocal;
        // Initialize output
        for (int r = 0; r < klocal; r++) { d_out_dists[out_base+r] = INF; d_out_ids[out_base+r] = -1; }

        bool used[32] = {};
        for (int slot = 0; slot < klocal; slot++) {
            float best = INF; int bi = -1;
            for (int r = 0; r < per_block_r; r++) {
                if (!used[r] && s_sel[r] >= 0 && s_exact[r] < best) {
                    best = s_exact[r]; bi = r;
                }
            }
            if (bi >= 0) {
                d_out_dists[out_base + slot] = best;
                d_out_ids  [out_base + slot] = s_sel[bi];
                used[bi] = true;
            }
        }
    }
}

void launch_leaf_flat_v28(
    const int* d_pair_leaf_ids, const int* d_pair_qids,
    const uint8_t* d_leaf_codes, const int* d_leaf_ids_data, const int* d_leaf_sizes,
    const float* d_lut_fine,
    const float* d_base_vecs, const float* d_q_batch,
    float* d_out_dists, int* d_out_ids,
    int n_pairs, int d, int Kr, int Br, int bpv, int leaf_size,
    int per_block_r, int klocal,
    cudaStream_t stream)
{
    // smem = PQ sort (float+int per vector) + query (float) + wsum(4f) + sel+exact (int+float per candidate)
    int smem = leaf_size * (int)(sizeof(float) + sizeof(int))
             + d         * (int)sizeof(float)
             + 4         * (int)sizeof(float)
             + per_block_r * (int)(sizeof(int) + sizeof(float));
    leaf_flat_kernel_v28<<<n_pairs, leaf_size, smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_leaf_sizes, d_leaf_codes, d_leaf_ids_data,
        d_lut_fine, d_base_vecs, d_q_batch,
        d_out_dists, d_out_ids,
        d, Kr, Br, bpv, leaf_size, per_block_r, klocal);
    CUDA_CHECK(cudaGetLastError());
}

// ── Final merge: per-block exact results → global top-k per query ────────────
// Pairs are stored in leaf-block order (d_pair_leaf_b, d_pair_qid_b).
// We re-sort by qid to group per query, then do segmented topk.

__global__ void fill_iota_kernel_v28(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

// d_perm[j] = position pi in the leaf-sorted pair array for qid-sorted slot j.
// d_query_offsets[qi] = start of query qi's segment (ExclusiveSum of d_leaf_cnt).
// d_leaf_cnt[qi]      = number of pairs for query qi.
__global__ void segmented_exact_topk_kernel_v28(
    const int*   __restrict__ d_perm,          // [n_pairs] perm[j] = pi in leaf-sorted array
    const int*   __restrict__ d_query_offsets, // [nq]
    const int*   __restrict__ d_leaf_cnt,      // [nq]
    const float* __restrict__ d_out_dists,     // [n_pairs × klocal]
    const int*   __restrict__ d_out_ids,
    float* d_final_dists, int* d_final_ids,
    int nq, int k, int klocal)
{
    const float INF = __int_as_float(0x7F800000);
    int qi = blockIdx.x, tid = threadIdx.x, BLK = blockDim.x;
    if (qi >= nq) return;
    int seg_start = d_query_offsets[qi];
    int seg_end   = seg_start + d_leaf_cnt[qi];

    extern __shared__ char shm[];
    float* s_dist = (float*)shm          + tid * k;
    int*   s_id   = (int*)((float*)shm + BLK * k) + tid * k;
    for (int i = 0; i < k; i++) { s_dist[i] = INF; s_id[i] = -1; }

    for (int p = seg_start + tid; p < seg_end; p += BLK) {
        int pi = d_perm[p];  // position in leaf-sorted array → index into d_out_dists
        for (int r = 0; r < klocal; r++) {
            float dist = d_out_dists[(long long)pi * klocal + r];
            int   vid  = d_out_ids  [(long long)pi * klocal + r];
            if (vid < 0 || dist >= INF) continue;
            int worst = 0;
            for (int i = 1; i < k; i++) if (s_dist[i] > s_dist[worst]) worst = i;
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
            int*   ti = (int*)((float*)shm + BLK*k) + t * k;
            for (int i = 0; i < k; i++) {
                float dv = td[i]; int v = ti[i]; if (v < 0) continue;
                int worst = 0;
                for (int j = 1; j < k; j++) if (out_d[j] > out_d[worst]) worst = j;
                if (dv < out_d[worst]) { out_d[worst] = dv; out_i[worst] = v; }
            }
        }
    }
}

void launch_final_merge_v28(int nq, int n_pairs, int klocal, int k, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    // Fill d_pair_leaf_a with iota: position pi in the leaf-sorted pair array.
    fill_iota_kernel_v28<<<(n_pairs+255)/256, 256, 0, s>>>(ws.d_pair_leaf_a, n_pairs);
    // Sort by qid: keys_in=d_pair_qid_b (leaf-sorted qids), values_in=d_pair_leaf_a (iota).
    // After sort: d_pair_qid_a=sorted qids (monotone), d_pair_leaf_b=perm into leaf-sorted positions.
    // d_query_offsets[qi] (from ExclusiveSum of d_leaf_cnt) still gives the start of
    // query qi's segment in the qid-sorted result since queries appear in ascending order.
    int end_bit = 1;
    { int nq_ = nq; while ((1 << end_bit) < nq_) end_bit++; end_bit = std::min(end_bit+1, 32); }
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_pair_qid_b, ws.d_pair_leaf_a,   // keys_in: qids, values_in: iota
        ws.d_pair_qid_a, ws.d_pair_leaf_b,   // keys_out: sorted qids, values_out: perm
        n_pairs, 0, end_bit, s));

    const int BLOCK = 32;
    int smem = BLOCK * k * (int)(sizeof(float) + sizeof(int));
    segmented_exact_topk_kernel_v28<<<nq, BLOCK, smem, s>>>(
        ws.d_pair_leaf_b,       // perm: pi in leaf-sorted array
        ws.d_query_offsets,     // segment starts (ExclusiveSum of d_leaf_cnt)
        ws.d_leaf_cnt,          // segment lengths
        ws.d_out_dists, ws.d_out_ids,
        ws.d_final_dists, ws.d_final_ids,
        nq, k, klocal);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v28
