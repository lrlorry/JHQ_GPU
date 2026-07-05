#include "hblock_v5/search.cuh"
#include "common/cuda_utils.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hblock_v5 {

// ── Route top-k (unchanged from v2) ──────────────────────────────────────────
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

// ── Subtract best routing centroid (unchanged from v2) ────────────────────────
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

// ── Gather leaf blocks (unchanged from v2) ────────────────────────────────────
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

// ── Fine residual LUT (unchanged from v2) ────────────────────────────────────
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

// ── v5: Build (leaf_id, flat_idx) pairs for global sort ──────────────────────
// flat_idx = q * ck3 + slot encodes both query and slot.
__global__ void build_pairs_kernel(
    const int* __restrict__ leaf_sel,  // [B, ck3]
    const int* __restrict__ leaf_cnt,  // [B]
    int* pair_keys,                    // [B*ck3] output: leaf_block_id
    int* pair_vals,                    // [B*ck3] output: flat_idx
    int B, int ck3)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= B * ck3) return;
    int q    = flat / ck3;
    int slot = flat % ck3;
    // Mark invalid slots (beyond leaf_cnt) with INT_MAX so they sort to end
    int leaf = (slot < leaf_cnt[q]) ? leaf_sel[q * ck3 + slot] : INT_MAX;
    pair_keys[flat] = leaf;
    pair_vals[flat] = flat;
}

// ── v5: Transposed LeafFine kernel ────────────────────────────────────────────
// Grid: n_leaf_max blocks (= n_leaf_blocks_, upper bound on unique leaves).
// Block: leaf_size threads.
// Smem: leaf_size * bpv bytes (holds one leaf block's codes).
//
// Each block owns one unique leaf (from sorted+RLE output).
// It loads that leaf into shared memory ONCE, then processes every query
// that needs it sequentially. Eliminates B*ck3 random HBM reads → 1 per leaf.
__global__ void leaf_fine_trans_kernel(
    const int*     __restrict__ sorted_vals,   // [B*ck3] query flat_idx, sorted by leaf_id
    const int*     __restrict__ uniq_leaves,   // [n_leaf_max] unique leaf_block_ids
    const int*     __restrict__ seg_start,     // [n_leaf_max] start in sorted_vals
    const int*     __restrict__ seg_count,     // [n_leaf_max] # queries needing this leaf
    const uint8_t* __restrict__ leaf_codes,    // [n_leaf_blocks, leaf_size, bpv] AoS
    const int*     __restrict__ leaf_ids,      // [n_leaf_blocks, leaf_size]
    const int*     __restrict__ leaf_sizes,    // [n_leaf_blocks]
    const float*   __restrict__ lut_fine,      // [B, d, Kr]
    float* fine_dists,                         // [B, ck3 * leaf_size]
    int*   fine_ids,                           // [B, ck3 * leaf_size]
    int ck3, int leaf_size, int bpv, int d, int Kr, int Br, int n_leaf_max)
{
    int uid = blockIdx.x;
    int tid = threadIdx.x;

    int leaf_blk = uniq_leaves[uid];
    // Guard: invalid uid (uid >= n_unique), or INT_MAX sentinel for invalid pairs
    if ((unsigned)leaf_blk >= (unsigned)n_leaf_max) return;

    int sz    = leaf_sizes[leaf_blk];
    int start = seg_start[uid];
    int count = seg_count[uid];
    if (count <= 0) return;

    // Load leaf codes into shared memory (one coalesced block read from HBM)
    extern __shared__ uint8_t smem[];
    const uint8_t* src = leaf_codes + (long long)leaf_blk * leaf_size * bpv;
    for (int i = tid; i < sz * bpv; i += blockDim.x)
        smem[i] = src[i];
    __syncthreads();

    const float INF = __int_as_float(0x7F800000);
    const long long leaf_id_base = (long long)leaf_blk * leaf_size;

    // Process each query that needs this leaf sequentially.
    // Different queries write to disjoint output regions → no race conditions.
    for (int qi = 0; qi < count; qi++) {
        int flat = sorted_vals[start + qi];
        int q    = flat / ck3;
        int slot = flat % ck3;

        const float* lut  = lut_fine + (long long)q * d * Kr;
        long long    base = ((long long)q * ck3 + slot) * leaf_size;

        float fd = INF;
        int   fi = -1;
        if (tid < sz) {
            const uint8_t* rc = smem + (long long)tid * bpv;
            fd = 0.f;
            for (int j = 0; j < d; ++j) {
                int ri = (Br == 4)
                    ? ((j & 1) ? (rc[j >> 1] >> 4) : (rc[j >> 1] & 0x0F))
                    : rc[j];
                fd += __ldg(&lut[j * Kr + ri]);
            }
            fi = leaf_ids[leaf_id_base + tid];
        }
        fine_dists[base + tid] = fd;
        fine_ids  [base + tid] = fi;
    }
}

// ── Final top-k (unchanged from v2) ──────────────────────────────────────────
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
    int batch_size, int n_leaf_blocks,
    SearchWorkspace& ws,
    float* h_out_dists, int* h_out_ids)
{
    if (ws.batch_cap <= 0)
        throw std::runtime_error("hblock_v5: workspace not initialised");

    const int B       = ws.batch_cap;
    const int BLOCK   = 256;
    const int n_cands = ck3 * leaf_size;
    const int n_pairs = B * ck3;

    const float one = 1.f, zero = 0.f;

    const int route1_smem = (K1 + 2 * BLOCK) * (int)sizeof(float);
    const int route2_smem = (K2 + 2 * BLOCK) * (int)sizeof(float);
    const int topk_smem   = 2 * BLOCK * (int)sizeof(float);
    const int leaf_cap_per_pair = std::max(1, ck3 / (ck1 * ck2));

    // Shared memory for transposed kernel: leaf_size * bpv bytes
    const int trans_smem = leaf_size * bpv;

    // Set extended smem limit for transposed kernel (A100 supports up to 164KB)
    static bool smem_set = false;
    if (!smem_set) {
        cudaFuncSetAttribute(leaf_fine_trans_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             std::max(trans_smem, 65536));
        smem_set = true;
    }

    // 13 events for 12 steps (same count as v2 for bench script compatibility)
    // Steps: H2D, Rotate, L1 GEMM+topk, L1 residual, L2 GEMM+topk, L2 residual,
    //        GatherLeaf, Prepare(sort+RLE+scan), FineLUT, LeafFineTrans, FinalTopk, D2H
    static const int NE = 13;
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

        // Rotate: q_rot = Pi @ q_batch
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 d, B, d, &one,
                                 d_Pi, d, ws.d_q_batch, d,
                                 &zero, ws.d_q_rot, d));
        cudaEventRecord(e[2], ws.stream);

        // L1 GEMM + topk
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K1, B, d, &one,
                                 d_route1_cents, d, ws.d_q_rot, d,
                                 &zero, ws.d_dots1, K1));
        select_route_topk_kernel<<<B, BLOCK, route1_smem, ws.stream>>>(
            ws.d_dots1, d_route1_norms, ws.d_top1_ids, K1, ck1);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[3], ws.stream);

        // L1 residual
        {
            long long tot = (long long)B * d;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            subtract_best_cent_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_rot, ws.d_top1_ids, d_route1_cents, ws.d_q_r1, d, B, ck1);
            CUDA_CHECK(cudaGetLastError());
        }
        cudaEventRecord(e[4], ws.stream);

        // L2 GEMM + topk
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K2, B, d, &one,
                                 d_route2_cents, d, ws.d_q_r1, d,
                                 &zero, ws.d_dots2, K2));
        select_route_topk_kernel<<<B, BLOCK, route2_smem, ws.stream>>>(
            ws.d_dots2, d_route2_norms, ws.d_top2_ids, K2, ck2);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[5], ws.stream);

        // L2 residual
        {
            long long tot = (long long)B * d;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            subtract_best_cent_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_r1, ws.d_top2_ids, d_route2_cents, ws.d_q_r2, d, B, ck2);
            CUDA_CHECK(cudaGetLastError());
        }
        cudaEventRecord(e[6], ws.stream);

        // Gather leaf blocks
        gather_leaf_blocks_kernel<<<(B + 63) / 64, 64, 0, ws.stream>>>(
            ws.d_top1_ids, ws.d_top2_ids,
            d_pair_blk_start, d_pair_blk_count,
            ws.d_leaf_sel, ws.d_leaf_cnt,
            B, ck1, ck2, ck3, K2, leaf_cap_per_pair);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[7], ws.stream);

        // Prepare: build pairs + global sort by leaf_id + RLE + prefix scan
        // Pre-fill d_uniq_leaves with n_leaf_max sentinel so excess kernel blocks
        // return early; pre-fill d_seg_count with 0 for correct prefix scan.
        CUDA_CHECK(cudaMemsetAsync(ws.d_uniq_leaves, 0xFF,
                                   (long long)ws.n_leaf_max * sizeof(int), ws.stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_seg_count, 0x00,
                                   (long long)ws.n_leaf_max * sizeof(int), ws.stream));

        build_pairs_kernel<<<(n_pairs + BLOCK - 1) / BLOCK, BLOCK, 0, ws.stream>>>(
            ws.d_leaf_sel, ws.d_leaf_cnt,
            ws.d_pair_keys, ws.d_pair_vals, B, ck3);
        CUDA_CHECK(cudaGetLastError());

        // Sort pairs by leaf_id (30-bit key handles up to 1B leaf blocks)
        cub::DeviceRadixSort::SortPairs(
            ws.d_cub_tmp, ws.cub_bytes,
            ws.d_pair_keys,   ws.d_pair_keys_s,
            ws.d_pair_vals,   ws.d_pair_vals_s,
            n_pairs, 0, 30, ws.stream);

        // Run-length encode sorted keys → unique leaves + per-leaf query counts
        cub::DeviceRunLengthEncode::Encode(
            ws.d_cub_tmp, ws.cub_bytes,
            ws.d_pair_keys_s,
            ws.d_uniq_leaves, ws.d_seg_count, ws.d_n_uniq,
            n_pairs, ws.stream);

        // Exclusive prefix sum of seg_count → seg_start (scan ws.n_leaf_max elems;
        // the 0-padded tail produces correct offsets for all valid uid)
        cub::DeviceScan::ExclusiveSum(
            ws.d_cub_tmp, ws.cub_bytes,
            ws.d_seg_count, ws.d_seg_start,
            ws.n_leaf_max, ws.stream);

        cudaEventRecord(e[8], ws.stream);

        // Fine LUT
        {
            long long tot = (long long)B * d * Kr;
            int grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            build_fine_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_r2, d_fine_c1d, ws.d_lut_fine, B, d, Kr);
            CUDA_CHECK(cudaGetLastError());
        }
        cudaEventRecord(e[9], ws.stream);

        // Transposed LeafFine: n_leaf_max blocks, leaf_size threads, leaf_size*bpv smem
        // Blocks with uid >= n_unique return immediately (d_uniq_leaves[uid] = 0xFF…)
        leaf_fine_trans_kernel<<<ws.n_leaf_max, leaf_size, trans_smem, ws.stream>>>(
            ws.d_pair_vals_s,
            ws.d_uniq_leaves, ws.d_seg_start, ws.d_seg_count,
            d_leaf_codes, d_leaf_ids, d_leaf_sizes,
            ws.d_lut_fine,
            ws.d_fine_dists, ws.d_fine_ids,
            ck3, leaf_size, bpv, d, Kr, Br, ws.n_leaf_max);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[10], ws.stream);

        // Final top-k
        final_topk_kernel<<<B, BLOCK, topk_smem, ws.stream>>>(
            ws.d_fine_dists, ws.d_fine_ids,
            ws.d_final_dists, ws.d_final_ids,
            n_cands, k, B);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(e[11], ws.stream);

        // D2H
        CUDA_CHECK(cudaMemcpyAsync(h_out_ids   + (long long)qoff * k,
                                   ws.d_final_ids,
                                   (long long)Bq * k * sizeof(int),
                                   cudaMemcpyDeviceToHost, ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_out_dists + (long long)qoff * k,
                                   ws.d_final_dists,
                                   (long long)Bq * k * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream));
        cudaEventRecord(e[12], ws.stream);
    }

    cudaError_t err;
    do { err = cudaStreamQuery(ws.stream); } while (err == cudaErrorNotReady);
    CUDA_CHECK(err);

    static const char* snames[] = {
        "H2D", "Rotate", "L1 GEMM+topk", "L1 residual",
        "L2 GEMM+topk", "L2 residual", "GatherLeaf",
        "Prepare(sort+RLE)", "FineLUT", "LeafFineTrans", "FinalTopk", "D2H"
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

} // namespace hblock_v5
