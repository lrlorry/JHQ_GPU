#include "hblock_v4/search.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v4 {

// ── Route top-k selection ─────────────────────────────────────────────────────
__global__ void select_route_topk_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ cent_norms,
    int*                      topk_ids,
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

// ── Subtract best routing centroid ────────────────────────────────────────────
__global__ void subtract_best_cent_kernel(
    const float* __restrict__ d_in,
    const int*   __restrict__ top_ids,
    const float* __restrict__ d_cents,
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
__global__ void gather_leaf_blocks_kernel(
    const int* __restrict__ top1_ids,
    const int* __restrict__ top2_ids,
    const int* __restrict__ pair_start,
    const int* __restrict__ pair_cnt,
    int* leaf_sel,
    int* leaf_cnt,
    int B, int ck1, int ck2, int ck3, int K2, int leaf_cap_per_pair)
{
    int bqi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bqi >= B) return;

    const int* my1    = top1_ids + bqi * ck1;
    const int* my2    = top2_ids + bqi * ck2;
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

// ── Sort leaf_sel rows by leaf_idx ────────────────────────────────────────────
__global__ void sort_leaf_sel_kernel(
    int*       leaf_sel,
    const int* leaf_cnt,
    int ck3)
{
    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    int  n_sel = leaf_cnt[bqi];
    int* row   = leaf_sel + (long long)bqi * ck3;

    extern __shared__ int s[];
    s[tid] = (tid < n_sel) ? row[tid] : INT_MAX;
    __syncthreads();

    for (int k = 2; k <= ck3; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ij = tid ^ j;
            if (ij > tid) {
                bool asc = ((tid & k) == 0);
                if (asc ? (s[tid] > s[ij]) : (s[tid] < s[ij])) {
                    int tmp = s[tid]; s[tid] = s[ij]; s[ij] = tmp;
                }
            }
            __syncthreads();
        }
    }

    if (tid < n_sel) row[tid] = s[tid];
}

// ── Leaf fine compute ─────────────────────────────────────────────────────────
__global__ void leaf_fine_compute_kernel(
    const int*     __restrict__ leaf_sel,
    const int*     __restrict__ leaf_cnt,
    const int*     __restrict__ leaf_sizes,
    const uint8_t* __restrict__ leaf_codes,
    const int*     __restrict__ leaf_ids,
    const float*   __restrict__ lut_fine,
    float* fine_dists,
    int*   fine_ids,
    int ck3, int leaf_size, int bpv, int d, int Kr, int Br)
{
    const float INF = __int_as_float(0x7F800000);
    int bqi  = blockIdx.x;
    int slot = blockIdx.y;
    int v    = threadIdx.x;

    float* my_dists = fine_dists + ((long long)bqi * ck3 + slot) * leaf_size;
    int*   my_ids   = fine_ids   + ((long long)bqi * ck3 + slot) * leaf_size;

    int n_sel = leaf_cnt[bqi];
    if (slot >= n_sel) {
        my_dists[v] = INF; my_ids[v] = -1;
        return;
    }

    int leaf_idx = leaf_sel[(long long)bqi * ck3 + slot];
    int n_vecs   = leaf_sizes[leaf_idx];

    if (v >= n_vecs) {
        my_dists[v] = INF; my_ids[v] = -1;
        return;
    }

    const uint8_t* rc     = leaf_codes + ((long long)leaf_idx * leaf_size + v) * bpv;
    const float*   my_lut = lut_fine   + (long long)bqi * d * Kr;

    float fd = 0.f;
    for (int j = 0; j < d; ++j) {
        int ri = (Br == 4)
            ? ((j & 1) ? (rc[j >> 1] >> 4) : (rc[j >> 1] & 0x0F))
            : rc[j];
        fd += __ldg(&my_lut[j * Kr + ri]);
    }
    my_dists[v] = fd;
    my_ids[v]   = leaf_ids[(long long)leaf_idx * leaf_size + v];
}

// ── Final top-k ───────────────────────────────────────────────────────────────
__global__ void final_topk_kernel(
    float*       fine_dists,
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
            float val = my_d[c];
            if (val < bv) { bv = val; bi = c; }
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
            od[r] = (w >= 0) ? my_d[w]                                    : INF;
            oi[r] = (w >= 0) ? fine_ids[(long long)bqi * n_cands + w] : -1;
            if (w >= 0) my_d[w] = INF;
        }
        __syncthreads();
    }
}

// ── search_hblock (v3: PCA cascade routing) ───────────────────────────────────
void search_hblock(
    cublasHandle_t cublas,
    const float* d_P1, const float* d_P2,
    const float* d_C1_proj, const float* d_C1_proj_norms, const float* d_C1_full,
    const float* d_C2_proj, const float* d_C2_proj_norms, const float* d_C2_full,
    const float* d_fine_c1d,
    const int*   d_pair_blk_start, const int* d_pair_blk_count,
    const uint8_t* d_leaf_codes, const int* d_leaf_ids, const int* d_leaf_sizes,
    const float*   h_queries,
    int nq, int d, int K1, int K2, int k1, int k2,
    int Kr, int Br, int bpv,
    int leaf_size, int ck1, int ck2, int ck3, int k,
    int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists, int* h_out_ids)
{
    if (ws.batch_cap <= 0)
        throw std::runtime_error("hblock_v4: workspace not initialised");

    const int B       = ws.batch_cap;
    const int BLOCK   = 256;
    const int n_cands = ck3 * leaf_size;

    const float one = 1.f, zero = 0.f;

    const int route1_smem = (K1 + 2 * BLOCK) * (int)sizeof(float);
    const int route2_smem = (K2 + 2 * BLOCK) * (int)sizeof(float);
    const int topk_smem   = 2 * BLOCK * (int)sizeof(float);

    const int leaf_cap_per_pair = std::max(1, ck3 / (ck1 * ck2));

    // 11 steps → NE = 12 markers
    static const int NE = 12;
    cudaEvent_t e[NE];
    for (int i = 0; i < NE; i++) CUDA_CHECK(cudaEventCreate(&e[i]));
    float step_ms[NE - 1] = {};

    for (int qoff = 0; qoff < nq; qoff += batch_size) {
        int Bq = std::min(batch_size, nq - qoff);

        std::memcpy(ws.h_q_pinned,
                    h_queries + (long long)qoff * d,
                    (long long)Bq * d * sizeof(float));
        if (Bq < B)
            std::memset(ws.h_q_pinned + (long long)Bq * d, 0,
                        (long long)(B - Bq) * d * sizeof(float));

        cudaEventRecord(e[0], ws.stream);

        // H2D
        CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                                   (long long)B * d * sizeof(float),
                                   cudaMemcpyHostToDevice, ws.stream));
        cudaEventRecord(e[1], ws.stream);

        // L1 Proj+GEMM+topk
        // z1(k1×B) = P1(k1×d, row-major = d×k1 col-major) OP_T × q_batch(d×B)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 k1, B, d, &one,
                                 d_P1, d, ws.d_q_batch, d,
                                 &zero, ws.d_z1, k1));
        // dots1(K1×B) = C1_proj(K1×k1, row-major = k1×K1 col-major) OP_T × z1(k1×B)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K1, B, k1, &one,
                                 d_C1_proj, k1, ws.d_z1, k1,
                                 &zero, ws.d_dots1, K1));
        select_route_topk_kernel<<<B, BLOCK, route1_smem, ws.stream>>>(
            ws.d_dots1, d_C1_proj_norms, ws.d_top1_ids, K1, ck1);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[2], ws.stream);

        // L1 residual: r1 = q - C1_full[c1]  (in original d-dim space)
        {
            long long tot = (long long)B * d;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            subtract_best_cent_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_batch, ws.d_top1_ids, d_C1_full, ws.d_q_r1, d, B, ck1);
            CUDA_CHECK(cudaGetLastError());
        }
        cudaEventRecord(e[3], ws.stream);

        // L2 Proj+GEMM+topk
        // z2(k2×B) = P2(k2×d, row-major = d×k2 col-major) OP_T × r1(d×B)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 k2, B, d, &one,
                                 d_P2, d, ws.d_q_r1, d,
                                 &zero, ws.d_z2, k2));
        // dots2(K2×B) = C2_proj(K2×k2, row-major = k2×K2 col-major) OP_T × z2(k2×B)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K2, B, k2, &one,
                                 d_C2_proj, k2, ws.d_z2, k2,
                                 &zero, ws.d_dots2, K2));
        select_route_topk_kernel<<<B, BLOCK, route2_smem, ws.stream>>>(
            ws.d_dots2, d_C2_proj_norms, ws.d_top2_ids, K2, ck2);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[4], ws.stream);

        // L2 residual: r2 = r1 - C2_full[c2]
        {
            long long tot = (long long)B * d;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            subtract_best_cent_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_r1, ws.d_top2_ids, d_C2_full, ws.d_q_r2, d, B, ck2);
            CUDA_CHECK(cudaGetLastError());
        }
        cudaEventRecord(e[5], ws.stream);

        // Gather leaf blocks
        gather_leaf_blocks_kernel<<<(B + 63) / 64, 64, 0, ws.stream>>>(
            ws.d_top1_ids, ws.d_top2_ids,
            d_pair_blk_start, d_pair_blk_count,
            ws.d_leaf_sel, ws.d_leaf_cnt,
            B, ck1, ck2, ck3, K2, leaf_cap_per_pair);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[6], ws.stream);

        // Sort leaf_sel
        sort_leaf_sel_kernel<<<B, ck3, (long long)ck3 * sizeof(int), ws.stream>>>(
            ws.d_leaf_sel, ws.d_leaf_cnt, ck3);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[7], ws.stream);

        // Fine LUT
        {
            long long tot = (long long)B * d * Kr;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            build_fine_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_r2, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
            CUDA_CHECK(cudaGetLastError());
        }
        cudaEventRecord(e[8], ws.stream);

        // Leaf fine compute
        leaf_fine_compute_kernel<<<dim3(B, ck3), leaf_size, 0, ws.stream>>>(
            ws.d_leaf_sel, ws.d_leaf_cnt,
            d_leaf_sizes, d_leaf_codes, d_leaf_ids,
            ws.d_lut_fine,
            ws.d_fine_dists, ws.d_fine_ids,
            ck3, leaf_size, bpv, d, Kr, Br);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[9], ws.stream);

        // Final top-k
        final_topk_kernel<<<B, BLOCK, topk_smem, ws.stream>>>(
            ws.d_fine_dists, ws.d_fine_ids,
            ws.d_final_dists, ws.d_final_ids,
            n_cands, k, B);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[10], ws.stream);

        // D2H
        CUDA_CHECK(cudaMemcpyAsync(h_out_ids   + (long long)qoff * k,
                                   ws.d_final_ids,
                                   (long long)Bq * k * sizeof(int),
                                   cudaMemcpyDeviceToHost, ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_out_dists + (long long)qoff * k,
                                   ws.d_final_dists,
                                   (long long)Bq * k * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream));
        cudaEventRecord(e[11], ws.stream);
    }

    cudaError_t err;
    do { err = cudaStreamQuery(ws.stream); } while (err == cudaErrorNotReady);
    CUDA_CHECK(err);

    static const char* snames[] = {
        "H2D", "L1 Proj+GEMM+topk", "L1 residual",
        "L2 Proj+GEMM+topk", "L2 residual", "GatherLeaf",
        "SortLeaf", "FineLUT", "LeafFine", "FinalTopk", "D2H"
    };
    float total = 0.f;
    for (int i = 0; i < NE - 1; i++) {
        float ms; cudaEventElapsedTime(&ms, e[i], e[i + 1]);
        step_ms[i] += ms;
        printf("  %-22s %.3f ms\n", snames[i], step_ms[i]);
        total += step_ms[i];
    }
    printf("  %-22s %.3f ms\n", "--- TOTAL ---", total);

    for (int i = 0; i < NE; i++) cudaEventDestroy(e[i]);
}

} // namespace hblock_v4
