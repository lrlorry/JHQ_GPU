// hblock_v41/search.cu
// v41 keeps v40 routing and candidate semantics, but separates the
// region-indirected PQ and exact-L2 stages so each payload can be streamed in
// independent bounded waves. No PRR or certified bounds are used.
#include "hblock_v41/search.cuh"
#include "hblock_v27/search.cuh"   // for gpu_build_block_adj_v27
#include "common/cuda_utils.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v41 {

// ══════════════════════════════════════════════════════════════════════════════
// Block beam search kernel (identical to v36 — operates on projected route
// space, never touches raw base vectors).
// ══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float
blk_qdist_v34(const float* __restrict__ s_q, float q_norm,
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
try_visit_v34(int* __restrict__ vis, int b, int n_blks)
{
    if (b < 0 || b >= n_blks) return false;
    int w = b >> 5, bit = b & 31;
    return !((atomicOr(&vis[w], 1 << bit) >> bit) & 1);
}

// SPT = slots per thread; beam_size = 32 × SPT; max SPT = 8 (beam ≤ 256).
// W = 4: expand 4 blocks per round, check termination once per round.
// Output: d_leaf_sel records each block AT EXPANSION TIME (v30 semantics).
template <int SPT>
__global__ void block_search_v35_kernel(
    const int*   __restrict__ d_top1,
    const int*   __restrict__ d_top2,
    const int*   __restrict__ d_top3,
    const int*   __restrict__ d_blk_start,
    const int*   __restrict__ d_blk_count,
    const int*   __restrict__ d_block_adj,
    const float* __restrict__ d_blk_proj,
    const float* __restrict__ d_blk_norm,
    const float* __restrict__ d_q_proj1,
    int*   d_leaf_sel,
    int*   d_leaf_cnt,
    int*   d_visited,
    int n_blks, int d_proj, int K2, int K3,
    int ck1, int ck2, int ck3,
    int degree, int max_rounds, int max_ls,
    int entry_per_cell, int bitmap_words)
{
    constexpr int W = 4;
    const int qi  = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr unsigned FULL = 0xffffffff;
    const float INF = 1e38f;

    extern __shared__ float s_q[];
    for (int jp = tid; jp < d_proj; jp += 32)
        s_q[jp] = d_q_proj1[(long long)qi * d_proj + jp];
    __syncwarp();

    float q_norm = 0.f;
    for (int jp = 0; jp < d_proj; jp++) q_norm += s_q[jp] * s_q[jp];

    int* vis = d_visited + (long long)qi * bitmap_words;
    for (int w = tid; w < bitmap_words; w += 32) vis[w] = 0;
    __syncwarp();

    // Beam: SPT slots per thread (beam_size = 32 * SPT, capped at 256)
    float my_dist[SPT]; int my_id[SPT]; bool my_exp[SPT];
    for (int s = 0; s < SPT; s++) { my_dist[s] = INF; my_id[s] = -1; my_exp[s] = true; }

    auto local_worst = [&](float& lw, int& ls) {
        lw = my_dist[0]; ls = 0;
        for (int s = 1; s < SPT; s++)
            if (my_dist[s] > lw) { lw = my_dist[s]; ls = s; }
    };

    auto try_insert = [&](float nd, int ni) {
        float lw; int ls;
        local_worst(lw, ls);
        float bmax = lw;
        for (int o = 16; o >= 1; o >>= 1)
            bmax = fmaxf(bmax, __shfl_xor_sync(FULL, bmax, o));
        if (nd >= bmax) return;
        unsigned em = __ballot_sync(FULL, lw == bmax);
        int el = __ffs(em) - 1;
        if (tid == el) { my_dist[ls] = nd; my_id[ls] = ni; my_exp[ls] = false; }
    };

    // ── Phase 1: Entry selection ────────────────────────────────
    for (int i1 = 0; i1 < ck1; i1++) {
        int c1 = d_top1[qi * ck1 + i1]; if (c1 < 0) continue;
        for (int i2 = 0; i2 < ck2; i2++) {
            int c2 = d_top2[qi * ck1 * ck2 + i1 * ck2 + i2]; if (c2 < 0) continue;
            for (int i3 = 0; i3 < ck3; i3++) {
                int c3 = d_top3[qi * (ck1*ck2*ck3) + (i1*ck2+i2)*ck3 + i3];
                if (c3 < 0) continue;
                int c123 = c1 * K2 * K3 + c2 * K3 + c3;
                int bs = d_blk_start[c123];
                int bc = d_blk_count[c123];
                if (bc == 0) continue;

                for (int start = 0; start < bc; start += 32) {
                    int bi = start + tid;
                    float ck_d = INF; int ck_id = -1;
                    if (bi < bc) {
                        ck_d  = blk_qdist_v34(s_q, q_norm, d_blk_proj, d_blk_norm, bs + bi, d_proj);
                        ck_id = bs + bi;
                    }
                    for (int ep = 0; ep < entry_per_cell; ep++) {
                        float mn = ck_d;
                        for (int o = 16; o >= 1; o >>= 1)
                            mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
                        if (mn >= INF - 1e6f) break;
                        unsigned wmask = __ballot_sync(FULL, ck_d == mn);
                        int wlane = __ffs(wmask) - 1;
                        int ins_id = __shfl_sync(FULL, ck_id, wlane);
                        if (tid == wlane) { ck_d = INF; ck_id = -1; }
                        bool fv = false;
                        if (tid == 0 && ins_id >= 0)
                            fv = try_visit_v34(vis, ins_id, n_blks);
                        bool do_ins = (bool)(__shfl_sync(FULL, (int)fv, 0));
                        if (do_ins) try_insert(mn, ins_id);
                    }
                }
            }
        }
    }

    // ── Phase 2: Batched beam search, full budget, no early termination ─────────
    int out_cnt = 0;
    for (int round = 0; round < max_rounds; round++) {
        if (out_cnt >= max_ls) break;  // output buffer full

        float best_unexp = INF;
        for (int s = 0; s < SPT; s++)
            if (!my_exp[s] && my_id[s] >= 0)
                best_unexp = fminf(best_unexp, my_dist[s]);
        for (int o = 16; o >= 1; o >>= 1)
            best_unexp = fminf(best_unexp, __shfl_xor_sync(FULL, best_unexp, o));

        if (best_unexp >= INF - 1e6f) break;  // no unexpanded blocks left

        for (int w = 0; w < W; w++) {
            if (out_cnt >= max_ls) break;

            float lb = INF; int lbs = -1;
            for (int s = 0; s < SPT; s++)
                if (!my_exp[s] && my_id[s] >= 0 && my_dist[s] < lb)
                    { lb = my_dist[s]; lbs = s; }
            float mn = lb;
            for (int o = 16; o >= 1; o >>= 1)
                mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
            if (mn >= INF - 1e6f) break;

            unsigned wm = __ballot_sync(FULL, lb == mn);
            int win_lane = __ffs(wm) - 1;
            int eid = -1;
            if (tid == win_lane && lbs >= 0) { my_exp[lbs] = true; eid = my_id[lbs]; }
            eid = __shfl_sync(FULL, eid, win_lane);
            if (eid < 0) break;

            if (tid == 0) d_leaf_sel[(long long)qi * max_ls + out_cnt] = eid;
            out_cnt++;

            int nb = (tid < degree) ? d_block_adj[(long long)eid * degree + tid] : -1;
            float nd = INF; int ni = -1;
            if (nb >= 0 && try_visit_v34(vis, nb, n_blks)) {
                nd = blk_qdist_v34(s_q, q_norm, d_blk_proj, d_blk_norm, nb, d_proj);
                ni = nb;
            }
            for (int src = 0; src < 32; src++) {
                float sd = __shfl_sync(FULL, nd, src);
                int   si = __shfl_sync(FULL, ni, src);
                if (si >= 0) try_insert(sd, si);
            }
        }
    }

    if (tid == 0) d_leaf_cnt[qi] = out_cnt;
}

#define LAUNCH_V35(SPT_VAL) \
    block_search_v35_kernel<SPT_VAL><<<B, 32, smem, ws.stream>>>( \
        ws.d_top1_ids, ws.d_top2_beam, ws.d_top3_beam, \
        d_pair_blk_start, d_pair_blk_count, \
        d_block_adj, d_blk_proj, d_blk_norm, \
        ws.d_q_proj1, \
        ws.d_leaf_sel, ws.d_leaf_cnt, ws.d_visited, \
        n_blks, d_proj, K2, K3, \
        ck1, ck2, ck3, \
        degree, max_rounds, max_ls, entry_per_cell, \
        ws.bitmap_words)

void gpu_block_search_v35(
    int B, int n_blks, int d_proj,
    int K2, int K3, int ck1, int ck2, int ck3,
    int degree, int ef, int max_ls, int entry_per_cell,
    const int*   d_block_adj,
    const float* d_blk_proj,
    const float* d_blk_norm,
    const int*   d_pair_blk_start,
    const int*   d_pair_blk_count,
    SearchWorkspace& ws)
{
    if (degree > 32) throw std::runtime_error("gpu_block_search_v35: degree must be <= 32");
    const int smem = d_proj * (int)sizeof(float);
    const int beam = ef;              // no cap — beam grows with ef
    const int max_rounds = ef / 4 + 1;
    if      (beam <= 32)  { LAUNCH_V35(1); }
    else if (beam <= 64)  { LAUNCH_V35(2); }
    else if (beam <= 128) { LAUNCH_V35(4); }
    else                  { LAUNCH_V35(8); }  // ef=256 → beam=256, SPT=8
    CUDA_CHECK(cudaGetLastError());
}
#undef LAUNCH_V35

// ── Routing kernels (identical to v36/v27, different namespace) ───────────────

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

void route_gpu_v29(
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
    if (nq > ws.batch_cap) throw std::runtime_error("hblock_v41: nq > batch_cap");
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

// ── Pair build (identical to v36) ─────────────────────────────────────────────

__global__ void build_pairs_kernel_v29(
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

void gpu_build_and_sort_pairs_v29(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_cub_tmp, ws.cub_bytes, ws.d_leaf_cnt, ws.d_query_offsets, nq, s));
    build_pairs_kernel_v29<<<(nq+255)/256, 256, 0, s>>>(
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

// ── v41 split region kernels ─────────────────────────────────────────────────────────────

__global__ void leaf_pq_topk_kernel_v41(
    const int*     __restrict__ d_pair_leaf_ids,
    const int*     __restrict__ d_pair_qids,
    const uint8_t* __restrict__ d_code_region_pool,
    const int*     __restrict__ d_code_region_slot,
    const int*     __restrict__ d_block_code_region,
    const int*     __restrict__ d_block_code_offset,
    const int*     __restrict__ leaf_sizes,
    const float*   __restrict__ lut_fine,
    int* d_candidate_ids, int* d_candidate_pos,
    int region_bytes,
    int d, int Kr, int Br, int bpv, int leaf_size, int per_block_r)
{
    const float INF = __int_as_float(0x7F800000);
    constexpr unsigned FULL = 0xffffffff;
    const int pi   = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int n_warps = leaf_size >> 5;

    const int leaf_blk = d_pair_leaf_ids[pi];
    const int qid      = d_pair_qids[pi];
    const int n_vecs   = leaf_sizes[leaf_blk];

    const int code_region = d_block_code_region[leaf_blk];
    const int code_slot   = d_code_region_slot[code_region];
    if (code_slot < 0) {
        if (tid < per_block_r) {
            d_candidate_ids[(long long)pi * per_block_r + tid] = -1;
            d_candidate_pos[(long long)pi * per_block_r + tid] = -1;
        }
        return;
    }
    const uint8_t* leaf_base = d_code_region_pool
        + (long long)code_slot * region_bytes
        + d_block_code_offset[leaf_blk];
    const int* leaf_ids_data = (const int*)(leaf_base + align4((size_t)bpv * leaf_size));

    const float* my_lut = lut_fine + (long long)qid * d * Kr;

    float my_dist = INF;
    int my_id = -1, my_pos = -1;
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
        my_id = __ldg(&leaf_ids_data[tid]);
        my_pos = tid;
    }

    for (int k = 2; k <= 32; k <<= 1) {
        bool asc = ((lane & k) == 0);
        for (int j = k >> 1; j > 0; j >>= 1) {
            float od = __shfl_xor_sync(FULL, my_dist, j);
            int   oi = __shfl_xor_sync(FULL, my_id,   j);
            int   op = __shfl_xor_sync(FULL, my_pos,  j);
            bool lower    = ((lane & j) == 0);
            bool want_min = (asc == lower);
            bool swap     = want_min ? (my_dist > od) : (my_dist < od);
            if (swap) { my_dist = od; my_id = oi; my_pos = op; }
        }
    }

    extern __shared__ char shm[];
    float* s_dist = reinterpret_cast<float*>(shm);
    int* s_id = reinterpret_cast<int*>(s_dist + leaf_size);
    int* s_pos = s_id + leaf_size;

    s_dist[tid] = my_dist;
    s_id  [tid] = my_id;
    s_pos [tid] = my_pos;
    __syncthreads();

    if (tid == 0) {
        int ptrs[4] = {0, 32, 64, 96};
        int actual_r = (per_block_r < n_vecs) ? per_block_r : n_vecs;
        for (int r = 0; r < actual_r; r++) {
            float best_d = INF; int best_w = -1;
            for (int w = 0; w < n_warps; w++) {
                int p = ptrs[w];
                if (p < (w+1)*32 && s_dist[p] < best_d) { best_d = s_dist[p]; best_w = w; }
            }
            const int src = best_w >= 0 ? ptrs[best_w] : -1;
            d_candidate_ids[(long long)pi * per_block_r + r] =
                (src >= 0 && best_d < INF) ? s_id[src] : -1;
            d_candidate_pos[(long long)pi * per_block_r + r] =
                (src >= 0 && best_d < INF) ? s_pos[src] : -1;
            if (best_w >= 0) ptrs[best_w]++;
        }
        for (int r = actual_r; r < per_block_r; r++) {
            d_candidate_ids[(long long)pi * per_block_r + r] = -1;
            d_candidate_pos[(long long)pi * per_block_r + r] = -1;
        }
    }
}

void launch_leaf_pq_topk_v41(
    const int* d_pair_leaf_ids, const int* d_pair_qids,
    const uint8_t* d_code_region_pool, const int* d_code_region_slot,
    const int* d_block_code_region, const int* d_block_code_offset,
    const int* d_leaf_sizes, const float* d_lut_fine,
    int* d_candidate_ids, int* d_candidate_pos,
    int region_bytes, int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size, int per_block_r,
    cudaStream_t stream)
{
    const size_t smem = (size_t)leaf_size *
                        (sizeof(float) + 2 * sizeof(int));
    leaf_pq_topk_kernel_v41<<<n_pairs, leaf_size, (int)smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_code_region_pool, d_code_region_slot,
        d_block_code_region, d_block_code_offset,
        d_leaf_sizes, d_lut_fine,
        d_candidate_ids, d_candidate_pos,
        region_bytes, d, Kr, Br, bpv, leaf_size, per_block_r);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void leaf_exact_kernel_v41(
    const int*     __restrict__ d_pair_leaf_ids,
    const int*     __restrict__ d_pair_qids,
    const int*     __restrict__ d_candidate_ids,
    const int*     __restrict__ d_candidate_pos,
    const uint8_t* __restrict__ d_raw_region_pool,
    const int*     __restrict__ d_raw_region_slot,
    const int*     __restrict__ d_block_raw_region,
    const int*     __restrict__ d_block_raw_offset,
    const int8_t*  __restrict__ d_q_batch_i8,
    float* d_out_dists, int* d_out_ids,
    int region_bytes, int d, int per_block_r, int klocal)
{
    const float INF = __int_as_float(0x7F800000);
    constexpr unsigned FULL = 0xffffffff;
    const int pi = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int leaf_blk = d_pair_leaf_ids[pi];
    const int qid = d_pair_qids[pi];
    const int raw_region = d_block_raw_region[leaf_blk];
    const int raw_slot = d_raw_region_slot[raw_region];

    if (raw_slot < 0) {
        if (tid < klocal) {
            d_out_dists[(long long)pi * klocal + tid] = INF;
            d_out_ids[(long long)pi * klocal + tid] = -1;
        }
        return;
    }

    const int8_t* raw_blk_base = reinterpret_cast<const int8_t*>(
        d_raw_region_pool + (long long)raw_slot * region_bytes +
        d_block_raw_offset[leaf_blk]);

    extern __shared__ char shm[];
    size_t off = 0;
    int8_t* s_query = reinterpret_cast<int8_t*>(shm + off);
    off += (size_t)d * sizeof(int8_t);
    off = align4(off);
    int* s_wsum = reinterpret_cast<int*>(shm + off);
    off += 4 * sizeof(int);
    float* s_exact = reinterpret_cast<float*>(shm + off);

    const int8_t* q_ptr = d_q_batch_i8 + (long long)qid * d;
    for (int j = tid; j < d; j += blockDim.x) s_query[j] = q_ptr[j];
    __syncthreads();

    for (int r = 0; r < per_block_r; ++r) {
        const int cand_id = d_candidate_ids[(long long)pi * per_block_r + r];
        const int local_pos = d_candidate_pos[(long long)pi * per_block_r + r];
        int partial = 0;
        if (cand_id >= 0 && local_pos >= 0) {
            const int8_t* vec = raw_blk_base + (long long)local_pos * d;
            for (int j = tid; j < d; j += blockDim.x) {
                const int diff = (int)s_query[j] - (int)vec[j];
                partial += diff * diff;
            }
        }
        for (int o = 16; o >= 1; o >>= 1)
            partial += __shfl_xor_sync(FULL, partial, o);
        __syncthreads();
        if (lane == 0) s_wsum[wid] = partial;
        __syncthreads();
        if (tid == 0) {
            const int total = s_wsum[0] + s_wsum[1] +
                              s_wsum[2] + s_wsum[3];
            s_exact[r] = (cand_id >= 0 && local_pos >= 0) ?
                         (float)total : INF;
        }
    }
    __syncthreads();

    if (tid == 0) {
        long long out_base = (long long)pi * klocal;
        for (int r = 0; r < klocal; r++) { d_out_dists[out_base+r] = INF; d_out_ids[out_base+r] = -1; }
        bool used[32] = {};
        for (int slot = 0; slot < klocal; slot++) {
            float best = INF; int bi = -1;
            for (int r = 0; r < per_block_r; r++) {
                const int cand_id =
                    d_candidate_ids[(long long)pi * per_block_r + r];
                if (!used[r] && cand_id >= 0 && s_exact[r] < best) {
                    best = s_exact[r]; bi = r;
                }
            }
            if (bi >= 0) {
                d_out_dists[out_base + slot] = best;
                d_out_ids[out_base + slot] =
                    d_candidate_ids[(long long)pi * per_block_r + bi];
                used[bi] = true;
            }
        }
    }
}

void launch_leaf_exact_v41(
    const int* d_pair_leaf_ids, const int* d_pair_qids,
    const int* d_candidate_ids, const int* d_candidate_pos,
    const uint8_t* d_raw_region_pool, const int* d_raw_region_slot,
    const int* d_block_raw_region, const int* d_block_raw_offset,
    const int8_t* d_q_batch_i8,
    float* d_out_dists, int* d_out_ids,
    int region_bytes, int n_pairs,
    int d, int leaf_size, int per_block_r, int klocal,
    cudaStream_t stream)
{
    size_t smem = (size_t)d * sizeof(int8_t);
    smem = align4(smem);
    smem += 4 * sizeof(int) + (size_t)per_block_r * sizeof(float);
    leaf_exact_kernel_v41<<<n_pairs, leaf_size, (int)smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_candidate_ids, d_candidate_pos,
        d_raw_region_pool, d_raw_region_slot, d_block_raw_region, d_block_raw_offset,
        d_q_batch_i8,
        d_out_dists, d_out_ids,
        region_bytes, d, per_block_r, klocal);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fill_iota_kernel_v29(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

__global__ void segmented_exact_topk_kernel_v29(
    const int*   __restrict__ d_perm,
    const int*   __restrict__ d_query_offsets,
    const int*   __restrict__ d_leaf_cnt,
    const float* __restrict__ d_out_dists,
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
        int pi = d_perm[p];
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

void launch_final_merge_v29(int nq, int n_pairs, int klocal, int k, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    fill_iota_kernel_v29<<<(n_pairs+255)/256, 256, 0, s>>>(ws.d_pair_leaf_a, n_pairs);
    int end_bit = 1;
    { int nq_ = nq; while ((1 << end_bit) < nq_) end_bit++; end_bit = std::min(end_bit+1, 32); }
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_pair_qid_b, ws.d_pair_qid_a,
        ws.d_pair_leaf_a, ws.d_pair_leaf_b,
        n_pairs, 0, end_bit, s));

    const int BLOCK = 32;
    int smem = BLOCK * k * (int)(sizeof(float) + sizeof(int));
    segmented_exact_topk_kernel_v29<<<nq, BLOCK, smem, s>>>(
        ws.d_pair_leaf_b,
        ws.d_query_offsets,
        ws.d_leaf_cnt,
        ws.d_out_dists, ws.d_out_ids,
        ws.d_final_dists, ws.d_final_ids,
        nq, k, klocal);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v41
