// hblock_v32/search.cu
// Single beam parameter: global top-ef routing (not per-parent) + HNSW natural termination.
// Replaces ck1/ck2/ck3/depth/beam_size with a single `ef`.
#include "hblock_v32/search.cuh"
#include "hblock_v27/search.cuh"   // gpu_build_block_adj_v27
#include "common/cuda_utils.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v32 {

// ── L1 topk (same as v30) ─────────────────────────────────────────────────────

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

// ── L2 global kernel ──────────────────────────────────────────────────────────
// B*ef blocks. Each block handles one (query, L1_slot) pair.
// Computes K2 distances for the L1 parent, stores r1 and distances.

__global__ void route_l2_global_kernel(
    const float* __restrict__ q,
    const float* __restrict__ C1_full,
    const float* __restrict__ Pi2,
    const float* __restrict__ C2_proj,
    const float* __restrict__ C2_norms,
    const int*   __restrict__ top1,
    float* r1_beam,   // [B*ef*d]
    float* dist2_all, // [B*ef*K2]
    int B, int d, int d_proj, int K2, int ef)
{
    const int beam_id = blockIdx.x;
    const int qi      = beam_id / ef;
    const int l1_slot = beam_id % ef;
    if (qi >= B) return;

    const float INF   = __int_as_float(0x7F800000);
    float* my_dists   = dist2_all + (long long)beam_id * K2;

    int c1 = top1[qi * ef + l1_slot];
    if (c1 < 0) {
        for (int c2 = threadIdx.x; c2 < K2; c2 += blockDim.x) my_dists[c2] = INF;
        return;
    }

    extern __shared__ char shm_l2[];
    float* s_r1   = (float*)shm_l2;
    float* s_proj = s_r1   + d;
    float* s_dist = s_proj + d_proj;

    const float* q_ptr  = q       + (long long)qi * d;
    const float* c1_ptr = C1_full + (long long)c1 * d;
    float*       r1_ptr = r1_beam + (long long)beam_id * d;

    __shared__ float s_r1sq;
    if (threadIdx.x == 0) s_r1sq = 0.f;
    __syncthreads();

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = q_ptr[j] - c1_ptr[j];
        s_r1[j] = v;
        r1_ptr[j] = v;
    }
    __syncthreads();

    // ||r1||² needed for cross-parent global comparison
    float local_sq = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) local_sq += s_r1[j] * s_r1[j];
    for (int o = 16; o >= 1; o >>= 1) local_sq += __shfl_xor_sync(0xffffffff, local_sq, o);
    if (threadIdx.x % 32 == 0) atomicAdd(&s_r1sq, local_sq);
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
        // full ||r1 - c2||² for cross-parent ranking
        s_dist[c2] = s_r1sq + c2_norms_base[c2] - 2.f * dot;
    }
    __syncthreads();

    for (int c2 = threadIdx.x; c2 < K2; c2 += blockDim.x)
        my_dists[c2] = s_dist[c2];
}

// ── Global top-ef selection ───────────────────────────────────────────────────
// B blocks. Picks top-ef from ef*K distances per query.
// Shared mem layout: [n floats: s_dist | BLOCK floats: s_val | BLOCK ints: s_idx]

__global__ void pick_global_topk_kernel(
    const float* __restrict__ d_dists,
    int* d_sel,
    int ef, int K)
{
    const float INF   = __int_as_float(0x7F800000);
    const int   BLOCK = blockDim.x;
    const int   qi    = blockIdx.x;
    const int   n     = ef * K;
    const float* my_d = d_dists + (long long)qi * n;
    int*      my_sel  = d_sel   + qi * ef;

    extern __shared__ char shm_pk[];
    float* s_dist = (float*)shm_pk;
    float* s_val  = s_dist + n;
    int*   s_idx  = (int*)(s_val + BLOCK);

    for (int i = threadIdx.x; i < n; i += BLOCK) s_dist[i] = my_d[i];
    __syncthreads();

    for (int r = 0; r < ef; r++) {
        // Each thread finds local minimum over its stripe
        float lv = INF; int li = -1;
        for (int i = threadIdx.x; i < n; i += BLOCK)
            if (s_dist[i] < lv) { lv = s_dist[i]; li = i; }
        s_val[threadIdx.x] = lv;
        s_idx[threadIdx.x] = li;
        __syncthreads();

        // Parallel reduction to find global minimum
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && s_val[threadIdx.x + stride] < s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            int wi = s_idx[0];
            my_sel[r] = wi;
            if (wi >= 0) s_dist[wi] = INF;
        }
        __syncthreads();
    }
}

// ── L3 global kernel ──────────────────────────────────────────────────────────
// B*ef blocks. Each block handles one top2 selection (l2_slot).
// Loads r1 from the L1 slot of this top2 selection, computes r2 and K3 dists.

__global__ void route_l3_global_kernel(
    const float* __restrict__ r1_beam,   // [B*ef*d]
    const float* __restrict__ C2_full,   // [K1*K2*d]
    const float* __restrict__ Pi3,       // [d_proj*d]
    const float* __restrict__ C3_proj,   // [K1*K2*K3*d_proj]
    const float* __restrict__ C3_norms,  // [K1*K2*K3]
    const int*   __restrict__ top1,      // [B*ef]
    const int*   __restrict__ top2_sel,  // [B*ef] l1slot*K2+c2local
    float* dist3_all,                    // [B*ef*K3]
    int B, int d, int d_proj, int K2, int K3, int ef)
{
    const int beam_id = blockIdx.x;
    const int qi      = beam_id / ef;
    const int r       = beam_id % ef;
    if (qi >= B) return;

    const float INF   = __int_as_float(0x7F800000);
    float* my_dists   = dist3_all + (long long)beam_id * K3;

    int sel2 = top2_sel[qi * ef + r];
    if (sel2 < 0) {
        for (int c3 = threadIdx.x; c3 < K3; c3 += blockDim.x) my_dists[c3] = INF;
        return;
    }

    int l1_slot  = sel2 / K2;
    int c2_local = sel2 % K2;
    int c1       = top1[qi * ef + l1_slot];
    int c12      = c1 * K2 + c2_local;

    extern __shared__ char shm_l3[];
    float* s_r2   = (float*)shm_l3;
    float* s_proj = s_r2   + d;
    float* s_dist = s_proj + d_proj;

    __shared__ float s_r2sq;
    if (threadIdx.x == 0) s_r2sq = 0.f;
    __syncthreads();

    const float* r1_ptr = r1_beam + (long long)(qi * ef + l1_slot) * d;
    const float* c2_ptr = C2_full + (long long)c12 * d;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        s_r2[j] = r1_ptr[j] - c2_ptr[j];
    __syncthreads();

    // ||r2||² needed for cross-parent global comparison
    float local_sq = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) local_sq += s_r2[j] * s_r2[j];
    for (int o = 16; o >= 1; o >>= 1) local_sq += __shfl_xor_sync(0xffffffff, local_sq, o);
    if (threadIdx.x % 32 == 0) atomicAdd(&s_r2sq, local_sq);
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
        // full ||r2 - c3||² for cross-parent ranking
        s_dist[c3] = s_r2sq + c3_norms_base[c3] - 2.f * dot;
    }
    __syncthreads();

    for (int c3 = threadIdx.x; c3 < K3; c3 += blockDim.x)
        my_dists[c3] = s_dist[c3];
}

// ── Expand all K3 children of each selected L2 cell → global c123 ────────────
// Outputs B*ef*K3 cells (exhaustive L3, same semantics as v30).
// top3_cells[qi*ef*K3 + l2_slot*K3 + c3] = global c123

__global__ void expand_l3_all_kernel(
    const int* __restrict__ top1,          // [B*ef]
    const int* __restrict__ top2_localidx, // [B*ef]
    int* top3_cells,                        // [B*ef*K3]
    int B, int K2, int K3, int ef)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * ef * K3;
    if (idx >= total) return;

    int qi      = idx / (ef * K3);
    int rem     = idx % (ef * K3);
    int l2_slot = rem / K3;
    int c3      = rem % K3;

    int sel2 = top2_localidx[qi * ef + l2_slot];
    if (sel2 < 0) { top3_cells[idx] = -1; return; }
    int l1_slot  = sel2 / K2;
    int c2_local = sel2 % K2;

    int c1 = top1[qi * ef + l1_slot];
    if (c1 < 0) { top3_cells[idx] = -1; return; }

    top3_cells[idx] = (c1 * K2 + c2_local) * K3 + c3;
}

// ── Convert best top3 local index → global c123 (for PQ LUT only) ────────────
__global__ void top3_best_to_cell_kernel(
    const int* __restrict__ top1,          // [B*ef]
    const int* __restrict__ top2_localidx, // [B*ef]
    const int* __restrict__ top3_localidx, // [B*ef]  top-1 used
    int* best_cell,                         // [B]
    int B, int K2, int K3, int ef)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= B) return;
    int sel3 = top3_localidx[qi * ef + 0];
    if (sel3 < 0) { best_cell[qi] = -1; return; }
    int l2_slot  = sel3 / K3;
    int c3_local = sel3 % K3;
    int sel2 = top2_localidx[qi * ef + l2_slot];
    if (sel2 < 0) { best_cell[qi] = -1; return; }
    int l1_slot  = sel2 / K2;
    int c2_local = sel2 % K2;
    int c1 = top1[qi * ef + l1_slot];
    if (c1 < 0) { best_cell[qi] = -1; return; }
    best_cell[qi] = (c1 * K2 + c2_local) * K3 + c3_local;
}

// ── Fine residual for PQ LUT (best L3 cell) ──────────────────────────────────

__global__ void extract_best_r3_v32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ C1_full,
    const float* __restrict__ C2_full,
    const float* __restrict__ C3_full,
    const int*   __restrict__ best_cell, // [B] — nearest L3 cell per query
    float* d_q_r3,
    int B, int d, int K2, int K3, int ef)
{
    int qi = blockIdx.x; if (qi >= B) return;
    int c123 = best_cell[qi];

    const float* q_ptr  = q + (long long)qi * d;
    float*       r3_ptr = d_q_r3 + (long long)qi * d;

    if (c123 < 0) {
        for (int j = threadIdx.x; j < d; j += blockDim.x)
            r3_ptr[j] = q_ptr[j];
        return;
    }

    int c12 = c123 / K3;
    int c3  = c123 % K3;
    int c1  = c12  / K2;

    const float* c1_ptr = C1_full + (long long)c1   * d;
    const float* c2_ptr = C2_full + (long long)c12  * d;
    const float* c3_ptr = C3_full + (long long)c123 * d;
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

// ── Routing pipeline ──────────────────────────────────────────────────────────

void route_gpu_v32(
    cublasHandle_t cublas,
    const float* d_Pi1, const float* d_Pi2, const float* d_Pi3,
    const float* d_route1_cents_proj, const float* d_route1_cents_full, const float* d_route1_norms,
    const float* d_route2_cents_proj, const float* d_route2_cents_full, const float* d_route2_norms,
    const float* d_route3_cents_proj, const float* d_route3_cents_full, const float* d_route3_norms,
    const float* d_fine_c1d,
    const float* h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int K3, int Kr, int ef,
    int batch_size,
    SearchWorkspace& ws)
{
    if (nq > ws.batch_cap) throw std::runtime_error("hblock_v32: nq > batch_cap");
    const int B = ws.batch_cap, BLOCK = 128;
    const float one = 1.f, zero = 0.f;
    cudaStream_t s = ws.stream;

    std::memcpy(ws.h_q_pinned, h_queries, (long long)nq * d * sizeof(float));
    if (nq < B) std::memset(ws.h_q_pinned + (long long)nq*d, 0, (long long)(B-nq)*d*sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                               (long long)B*d*sizeof(float), cudaMemcpyHostToDevice, s));

    // L1: project queries, gemm, pick top-ef
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_proj, B, d, &one, d_Pi1, d, ws.d_q_batch, d, &zero, ws.d_q_proj1, d_proj));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             K1, B, d_proj, &one, d_route1_cents_proj, d_proj, ws.d_q_proj1, d_proj, &zero, ws.d_dots1, K1));
    {
        int shm1 = (K1 + 2*BLOCK) * (int)sizeof(float);
        select_route_topk_kernel<<<B, BLOCK, shm1, s>>>(
            ws.d_dots1, d_route1_norms, ws.d_top1_ids, K1, ef);
        CUDA_CHECK(cudaGetLastError());
    }

    // L2: B*ef blocks compute K2 dists, store r1 and dists
    {
        int shm2 = (d + d_proj + K2) * (int)sizeof(float);
        route_l2_global_kernel<<<B*ef, BLOCK, shm2, s>>>(
            ws.d_q_batch, d_route1_cents_full, d_Pi2,
            d_route2_cents_proj, d_route2_norms,
            ws.d_top1_ids, ws.d_r1_beam, ws.d_dist2_all,
            B, d, d_proj, K2, ef);
        CUDA_CHECK(cudaGetLastError());
    }
    // L2: pick global top-ef from ef*K2 per query
    {
        int shm_pk = (ef * K2 + 2 * BLOCK) * (int)sizeof(float);
        pick_global_topk_kernel<<<B, BLOCK, shm_pk, s>>>(
            ws.d_dist2_all, ws.d_top2_localidx, ef, K2);
        CUDA_CHECK(cudaGetLastError());
    }

    // L3: B*ef blocks compute K3 dists
    {
        int shm3 = (d + d_proj + K3) * (int)sizeof(float);
        route_l3_global_kernel<<<B*ef, BLOCK, shm3, s>>>(
            ws.d_r1_beam, d_route2_cents_full, d_Pi3,
            d_route3_cents_proj, d_route3_norms,
            ws.d_top1_ids, ws.d_top2_localidx, ws.d_dist3_all,
            B, d, d_proj, K2, K3, ef);
        CUDA_CHECK(cudaGetLastError());
    }
    // L3: pick top-1 from ef*K3 per query (for PQ LUT only)
    {
        int shm_pk = (ef * K3 + 2 * BLOCK) * (int)sizeof(float);
        pick_global_topk_kernel<<<B, BLOCK, shm_pk, s>>>(
            ws.d_dist3_all, ws.d_top3_localidx, 1, ef * K3);
        CUDA_CHECK(cudaGetLastError());
    }
    // PQ LUT: convert best localidx to global c123
    {
        top3_best_to_cell_kernel<<<(B+255)/256, 256, 0, s>>>(
            ws.d_top1_ids, ws.d_top2_localidx, ws.d_top3_localidx,
            ws.d_pq_best_cell, B, K2, K3, ef);
        CUDA_CHECK(cudaGetLastError());
    }
    // Expand all K3 children of each selected L2 cell → ef*K3 entry cells
    {
        int total = B * ef * K3;
        expand_l3_all_kernel<<<(total+255)/256, 256, 0, s>>>(
            ws.d_top1_ids, ws.d_top2_localidx,
            ws.d_top3_cells, B, K2, K3, ef);
        CUDA_CHECK(cudaGetLastError());
    }

    // Fine residual and PQ LUT
    extract_best_r3_v32_kernel<<<B, BLOCK, 0, s>>>(
        ws.d_q_batch, d_route1_cents_full, d_route2_cents_full, d_route3_cents_full,
        ws.d_pq_best_cell, ws.d_q_r3, B, d, K2, K3, ef);
    CUDA_CHECK(cudaGetLastError());
    {
        const int BLUT = 256;
        long long tot = (long long)B * d * Kr;
        int grid = (int)std::min((tot + BLUT - 1) / BLUT, (long long)65535);
        build_fine_lut_kernel<<<grid, BLUT, 0, s>>>(ws.d_q_r3, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ── Pair build ────────────────────────────────────────────────────────────────

__global__ void build_pairs_kernel_v32(
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

void gpu_build_and_sort_pairs_v32(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_cub_tmp, ws.cub_bytes, ws.d_leaf_cnt, ws.d_query_offsets, nq, s));
    build_pairs_kernel_v32<<<(nq+255)/256, 256, 0, s>>>(
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

// ── Block beam search with flat entry + HNSW natural termination ──────────────

__device__ __forceinline__ float
blk_qdist_v32(const float* __restrict__ s_q, float q_norm,
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
try_visit_v32(int* __restrict__ vis, int b, int n_blks)
{
    if (b < 0 || b >= n_blks) return false;
    int w = b >> 5, bit = b & 31;
    return !((atomicOr(&vis[w], 1 << bit) >> bit) & 1);
}

template <int SPT>
__global__ void block_search_fused_v32(
    const int*   __restrict__ d_top3_cells,  // [B*ef_cells] global c123 entry cells
    const int*   __restrict__ d_blk_start,
    const int*   __restrict__ d_blk_count,
    const int*   __restrict__ d_block_adj,
    const float* __restrict__ d_blk_proj,
    const float* __restrict__ d_blk_norm,
    const float* __restrict__ d_q_proj1,
    int* d_leaf_sel, int* d_leaf_cnt, int* d_visited,
    int n_blks, int d_proj,
    int ef_cells, int degree, int max_ls, int entry_per_cell, int bitmap_words)
{
    const int qi  = blockIdx.x;
    const int tid = threadIdx.x;
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

    // insert_if_better: check bmax BEFORE marking visited.
    auto insert_if_better = [&](float nd, int ni) {
        float lw; int ls; local_worst(lw, ls);
        float bmax = lw;
        for (int o = 16; o >= 1; o >>= 1) bmax = fmaxf(bmax, __shfl_xor_sync(FULL, bmax, o));
        if (nd >= bmax) return;
        bool fv = false;
        if (tid == 0) fv = try_visit_v32(vis, ni, n_blks);
        if (!(bool)__shfl_sync(FULL, (int)fv, 0)) return;
        unsigned em = __ballot_sync(FULL, lw == bmax);
        int el = __ffs(em) - 1;
        if (tid == el) { my_dist[ls] = nd; my_id[ls] = ni; my_exp[ls] = false; }
    };

    // Entry phase: all ef*K3 global L3 cells (exhaustive L3 expansion)
    for (int r = 0; r < ef_cells; r++) {
        int c123 = d_top3_cells[(long long)qi * ef_cells + r];
        if (c123 < 0) continue;
        int bs = d_blk_start[c123], bc = d_blk_count[c123];
        if (bc == 0) continue;
        for (int start = 0; start < bc; start += 32) {
            int bi = start + tid;
            float ck_d = 1e38f; int ck_id = -1;
            if (bi < bc) {
                ck_d  = blk_qdist_v32(s_q, q_norm, d_blk_proj, d_blk_norm, bs+bi, d_proj);
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
                insert_if_better(mn, ins_id);
            }
        }
    }

    // Graph expansion with HNSW natural termination
    int out_cnt = 0;
    for (int iter = 0; iter < max_ls; iter++) {
        float lb = 1e38f; int lbs = -1;
        for (int s = 0; s < SPT; s++)
            if (!my_exp[s] && my_id[s] >= 0 && my_dist[s] < lb) { lb = my_dist[s]; lbs = s; }
        float mn = lb;
        for (int o = 16; o >= 1; o >>= 1) mn = fminf(mn, __shfl_xor_sync(FULL, mn, o));
        if (mn >= 1e37f) break;

        // HNSW natural termination: if best unexpanded >= beam max, can't improve
        float lw_w; int ls_w; local_worst(lw_w, ls_w);
        float bmax = lw_w;
        for (int o = 16; o >= 1; o >>= 1) bmax = fmaxf(bmax, __shfl_xor_sync(FULL, bmax, o));
        if (mn >= bmax) break;

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
            if (nb >= 0 && nb < n_blks && !((vis[nb >> 5] >> (nb & 31)) & 1)) {
                new_d = blk_qdist_v32(s_q, q_norm, d_blk_proj, d_blk_norm, nb, d_proj);
                new_id = nb;
            }
        }
        for (int src = 0; src < 32; src++) {
            float nd = __shfl_sync(FULL, new_d,  src);
            int   ni = __shfl_sync(FULL, new_id, src);
            if (ni >= 0) insert_if_better(nd, ni);
        }
    }
    if (tid == 0) d_leaf_cnt[qi] = out_cnt;
}

#define LAUNCH_BEAM_V32(SPT_VAL) \
    block_search_fused_v32<SPT_VAL><<<B, 32, smem, ws.stream>>>( \
        ws.d_top3_cells, \
        d_pair_blk_start, d_pair_blk_count, \
        d_block_adj, d_blk_proj, d_blk_norm, \
        ws.d_q_proj1, \
        ws.d_leaf_sel, ws.d_leaf_cnt, ws.d_visited, \
        n_blks, d_proj, ef_cells, degree, max_ls, entry_per_cell, ws.bitmap_words)

void gpu_block_search_v32(
    int B, int n_blks, int d_proj,
    int ef_cells, int degree, int max_ls, int entry_per_cell,
    const int*   d_block_adj, const float* d_blk_proj, const float* d_blk_norm,
    const int*   d_pair_blk_start, const int* d_pair_blk_count,
    SearchWorkspace& ws)
{
    if (degree > 32) throw std::runtime_error("gpu_block_search_v32: graph_degree must be <= 32");
    int smem = d_proj * (int)sizeof(float);
    if      (ef_cells <= 32)  { LAUNCH_BEAM_V32(1); }
    else if (ef_cells <= 64)  { LAUNCH_BEAM_V32(2); }
    else                      { LAUNCH_BEAM_V32(4); }
    CUDA_CHECK(cudaGetLastError());
}
#undef LAUNCH_BEAM_V32

// ── Leaf flat + merge (identical to v30, different namespace) ─────────────────

__global__ void leaf_flat_kernel_v32(
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
    const int n_warps = leaf_size >> 5;

    const int leaf_blk = d_pair_leaf_ids[pi];
    const int qid      = d_pair_qids[pi];
    const int n_vecs   = leaf_sizes[leaf_blk];

    const float*   my_lut    = lut_fine   + (long long)qid      * d * Kr;
    const uint8_t* leaf_base = leaf_codes + (long long)leaf_blk * bpv * leaf_size;

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

    for (int k = 2; k <= 32; k <<= 1) {
        bool asc = ((lane & k) == 0);
        for (int j = k >> 1; j > 0; j >>= 1) {
            float od = __shfl_xor_sync(FULL, my_dist, j);
            int   oi = __shfl_xor_sync(FULL, my_id,   j);
            bool swap = asc ? (od < my_dist) : (od > my_dist);
            if (swap) { my_dist = od; my_id = oi; }
        }
    }

    extern __shared__ char shm_lf[];
    float* s_dist  = (float*)shm_lf;
    int*   s_id    = (int*)(s_dist + leaf_size);
    float* s_query = (float*)(s_id + leaf_size);
    float* s_wsum  = s_query + d;
    int*   s_sel   = (int*)(s_wsum + 4);
    float* s_exact = (float*)(s_sel + per_block_r);

    s_dist[tid] = my_dist;
    s_id  [tid] = my_id;
    __syncthreads();

    const float* q_ptr = d_q_batch + (long long)qid * d;
    for (int j = tid; j < d; j += blockDim.x)
        s_query[j] = q_ptr[j];
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
            s_sel[r] = (best_w >= 0 && best_d < INF) ? s_id[ptrs[best_w]] : -1;
            if (best_w >= 0) ptrs[best_w]++;
        }
        for (int r = (per_block_r < n_vecs ? per_block_r : n_vecs); r < per_block_r; r++) s_sel[r] = -1;
    }
    __syncthreads();

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
        for (int o = 16; o >= 1; o >>= 1) partial += __shfl_xor_sync(FULL, partial, o);
        __syncthreads();
        if (lane == 0) s_wsum[wid] = partial;
        __syncthreads();
        if (tid == 0) {
            float total = s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3];
            s_exact[r] = (cand_id >= 0) ? total : INF;
        }
    }
    __syncthreads();

    if (tid == 0) {
        long long out_base = (long long)pi * klocal;
        for (int r = 0; r < klocal; r++) { d_out_dists[out_base+r] = INF; d_out_ids[out_base+r] = -1; }
        bool used[32] = {};
        for (int slot = 0; slot < klocal; slot++) {
            float best = INF; int bi = -1;
            for (int r = 0; r < per_block_r; r++)
                if (!used[r] && s_sel[r] >= 0 && s_exact[r] < best) { best = s_exact[r]; bi = r; }
            if (bi >= 0) { d_out_dists[out_base+slot] = best; d_out_ids[out_base+slot] = s_sel[bi]; used[bi] = true; }
        }
    }
}

void launch_leaf_flat_v32(
    const int* d_pair_leaf_ids, const int* d_pair_qids,
    const uint8_t* d_leaf_codes, const int* d_leaf_ids_data, const int* d_leaf_sizes,
    const float* d_lut_fine, const float* d_base_vecs, const float* d_q_batch,
    float* d_out_dists, int* d_out_ids,
    int n_pairs, int d, int Kr, int Br, int bpv, int leaf_size,
    int per_block_r, int klocal, cudaStream_t stream)
{
    int smem = leaf_size * (int)(sizeof(float) + sizeof(int))
             + d         * (int)sizeof(float)
             + 4         * (int)sizeof(float)
             + per_block_r * (int)(sizeof(int) + sizeof(float));
    leaf_flat_kernel_v32<<<n_pairs, leaf_size, smem, stream>>>(
        d_pair_leaf_ids, d_pair_qids,
        d_leaf_sizes, d_leaf_codes, d_leaf_ids_data,
        d_lut_fine, d_base_vecs, d_q_batch,
        d_out_dists, d_out_ids,
        d, Kr, Br, bpv, leaf_size, per_block_r, klocal);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fill_iota_kernel_v32(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

__global__ void segmented_exact_topk_kernel_v32(
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

void launch_final_merge_v32(int nq, int n_pairs, int klocal, int k, SearchWorkspace& ws)
{
    cudaStream_t s = ws.stream;
    fill_iota_kernel_v32<<<(n_pairs+255)/256, 256, 0, s>>>(ws.d_pair_leaf_a, n_pairs);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        ws.d_cub_tmp, ws.cub_bytes,
        ws.d_pair_qid_b,  ws.d_pair_qid_a,
        ws.d_pair_leaf_a, ws.d_pair_leaf_b,
        n_pairs, 0, 32, s));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ws.d_cub_tmp, ws.cub_bytes, ws.d_leaf_cnt, ws.d_query_offsets, nq, s));
    int smem = 2 * 32 * k * (int)sizeof(float);
    segmented_exact_topk_kernel_v32<<<nq, 32, smem, s>>>(
        ws.d_pair_leaf_b, ws.d_query_offsets, ws.d_leaf_cnt,
        ws.d_out_dists, ws.d_out_ids,
        ws.d_final_dists, ws.d_final_ids,
        nq, k, klocal);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace hblock_v32
