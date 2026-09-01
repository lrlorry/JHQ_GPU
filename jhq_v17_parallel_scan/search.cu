#include "jhq_v17_parallel_scan/search.cuh"

// Per-thread candidate slots kept by scan_ivf_coalesced_kernel. Compile-time
// so ld[]/lp[] stay in registers.
//
// This is a lossy step: each thread keeps only its own best K_LOCAL, so if
// more than K_LOCAL of the true top-ck land in one thread's stride class, the
// rest are dropped before the block-wide selection ever sees them. Raise it to
// measure what that costs:  -DJHQ_K_LOCAL=8
//
// Shared memory is (2*K_LOCAL + 2) * BLOCK floats -- 10KB at 4, 34KB at 16,
// past the 48KB default at 32. capture_graph() checks before launching.
#ifndef JHQ_K_LOCAL
#define JHQ_K_LOCAL 4
#endif
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace jhq_gpu {

// ── select_probes_kernel (same as v10) ────────────────────────────────────────
__global__ void select_probes_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ cent_norms,
    const int*   __restrict__ list_offsets,
    int*                      probe_ids,
    int*                      probe_offsets,
    int*                      query_total,
    int nlist, int nprobe)
{
    extern __shared__ float s[];
    const int BLOCK = blockDim.x;
    float* red_val = s + nlist;
    int*   red_idx = (int*)(red_val + BLOCK);

    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    const float* row = dots + (long long)bqi * nlist;
    for (int c = tid; c < nlist; c += BLOCK)
        s[c] = cent_norms[c] - 2.0f * row[c];
    __syncthreads();

    int* my_ids  = probe_ids     + bqi * nprobe;
    int* my_offs = probe_offsets + bqi * (nprobe + 1);
    int  acc = 0;
    if (tid == 0) my_offs[0] = 0;

    for (int p = 0; p < nprobe; ++p) {
        float bv = __int_as_float(0x7F800000);
        int   bi = -1;
        for (int c = tid; c < nlist; c += BLOCK)
            if (s[c] < bv) { bv = s[c]; bi = c; }
        red_val[tid] = bv; red_idx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && red_val[tid+stride] < red_val[tid]) {
                red_val[tid] = red_val[tid+stride]; red_idx[tid] = red_idx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = red_idx[0];
            s[w] = __int_as_float(0x7F800000);
            my_ids[p] = w;
            acc += list_offsets[w+1] - list_offsets[w];
            my_offs[p+1] = acc;
        }
        __syncthreads();
    }
    if (tid == 0) query_total[bqi] = acc;
}

// ── build_byte_lut_kernel — PQ asymmetric distance table ────────────────────
// lut[b][m][c] = ||q_sub(b,m) - centroid[m][c]||^2, the same quantity the
// official compute_primary_distance_tables_flat() fills. Entries past K stay
// at +inf so a stale code byte can never look attractive.
//
// The scan kernel is untouched by the switch to a product quantizer: it was
// already doing sum_m lut[m][code[m]], which is exactly PQ's ADC.
__global__ void build_byte_lut_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_cent,      // [M][K][Ds]
    float*                    d_byte_lut,
    int B, int d, int M, int Ds, int K)
{
    long long total = (long long)B * M * 256;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)M * 256));
        int local = (int)(i % ((long long)M * 256));
        int m     = local / 256;
        int c     = local % 256;

        if (c >= K) { d_byte_lut[i] = __int_as_float(0x7F800000); continue; }

        const float* q_m = d_q_rot + (long long)bqi * d + (long long)m * Ds;
        const float* cc  = d_cent  + ((long long)m * K + c) * Ds;
        float sum = 0.0f;
        for (int j = 0; j < Ds; ++j) { float t = q_m[j] - cc[j]; sum += t * t; }
        d_byte_lut[i] = sum;
    }
}

// ── scan_ivf_coalesced_kernel (NEW) ───────────────────────────────────────────
//
// Uses list_primary_t in [M, N] layout instead of [N, M].
//
// Access pattern for 32 threads in a warp at outer-candidate step j:
//   abs_pos values are consecutive (abs_pos_0, abs_pos_0+1, ..., abs_pos_0+31)
//   At codeword m: list_primary_t[m*N + abs_pos_0 .. abs_pos_0+31]
//                = 32 consecutive bytes = 1 cache line (32× vs v10)
//
// Byte LUT lookup: unchanged from v10 (96 L2 reads per candidate).
// No __syncthreads() in the inner loops — full warp scheduling flexibility.
//
// ── Two-stage scan ───────────────────────────────────────────────────────────
// v16 launched one block per query: 256 blocks against this card's 170 SMs, so
// about 1.5 blocks per SM, each running a serial loop over every candidate of
// its query. Nothing was left to hide that loop's latency behind, and the
// measured bandwidth was 6.8% of peak against cuVS IVF-PQ's 52% on the same
// data at the same recall. The gap also shrank from 39x to 5x as nprobe grew,
// which is what a launch-bound kernel looks like: more work per block finally
// amortises the fixed cost.
//
// Stage 1 gives every (query, probe) pair its own block -- B*nprobe of them,
// 32,768 at B=256, nprobe=128 -- and each block reduces its own list to its
// best ckb candidates. Stage 2 merges the nprobe partial lists per query into
// the global top-ck.
//
// ckb is chosen so the split cannot lose a candidate the single-block version
// would have kept: with nprobe lists sharing ck slots, a list holds ck/nprobe
// of them on average, and ckb keeps four times that with a floor of 32. At
// nprobe=1 the formula degenerates to ckb=ck, which is exactly the old
// behaviour.
__host__ __device__ __forceinline__ int scan_ckb(int ck, int nprobe) {
    if (nprobe <= 1) return ck;
    int per = (ck + nprobe - 1) / nprobe;
    int v   = per * 4;
    if (v < 32) v = 32;
    return v < ck ? v : ck;
}

__global__ void scan_ivf_partial_kernel(
    const float*   __restrict__ d_byte_lut,     // [B, M, 256]
    const int*     __restrict__ probe_ids,      // [B, nprobe]
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary_t, // [M, N] transposed
    float*                      part_dist,      // [B, nprobe, ckb]
    int*                        part_pos,
    int nprobe, int M, int N, int ckb)
{
    constexpr int K_LOCAL = JHQ_K_LOCAL;
    const float   INF     = __int_as_float(0x7F800000);
    const int     BLOCK   = blockDim.x;
    const int     bqi     = blockIdx.x;         // query within the batch
    const int     p       = blockIdx.y;         // probe within the query
    const int     tid     = threadIdx.x;

    extern __shared__ char shm[];
    float* s_cdist   = (float*)shm;
    int*   s_cpos    = (int*)(s_cdist + K_LOCAL * BLOCK);
    float* s_red_val = (float*)(s_cpos + K_LOCAL * BLOCK);
    int*   s_red_idx = (int*)(s_red_val + BLOCK);

    const float* my_lut = d_byte_lut + (long long)bqi * M * 256;
    const int    lid    = probe_ids[bqi * nprobe + p];
    const int    beg    = list_offsets[lid];
    const int    end    = list_offsets[lid + 1];

    float ld[K_LOCAL]; int lp[K_LOCAL];
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) { ld[i] = INF; lp[i] = -1; }

    // One list, contiguous: abs_pos needs no per-candidate probe search, unlike
    // v16 where every candidate walked the probe-offset array to find its list.
    for (int abs_pos = beg + tid; abs_pos < end; abs_pos += BLOCK) {
        float dist = 0.0f;
        #pragma unroll 4
        for (int m = 0; m < M; ++m) {
            uint8_t cm = __ldg(&list_primary_t[(long long)m * N + abs_pos]);
            dist += my_lut[m * 256 + cm];
        }
        if (dist < ld[K_LOCAL - 1]) {
            ld[K_LOCAL - 1] = dist;
            lp[K_LOCAL - 1] = abs_pos;
            #pragma unroll
            for (int i = K_LOCAL - 1; i > 0 && ld[i] < ld[i-1]; --i) {
                float td = ld[i-1]; ld[i-1] = ld[i]; ld[i] = td;
                int   tp = lp[i-1]; lp[i-1] = lp[i]; lp[i] = tp;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) {
        s_cdist[tid * K_LOCAL + i] = ld[i];
        s_cpos [tid * K_LOCAL + i] = lp[i];
    }
    __syncthreads();

    const int n_cands = K_LOCAL * BLOCK;
    const long long base = ((long long)bqi * nprobe + p) * ckb;
    for (int c = 0; c < ckb; ++c) {
        float bv = INF; int bi = -1;
        for (int ci = tid; ci < n_cands; ci += BLOCK)
            if (s_cdist[ci] < bv) { bv = s_cdist[ci]; bi = ci; }
        s_red_val[tid] = bv; s_red_idx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_red_val[tid+stride] < s_red_val[tid]) {
                s_red_val[tid] = s_red_val[tid+stride];
                s_red_idx[tid] = s_red_idx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = s_red_idx[0];
            part_dist[base + c] = (w >= 0) ? s_cdist[w] : INF;
            part_pos [base + c] = (w >= 0) ? s_cpos [w] : -1;
            if (w >= 0) s_cdist[w] = INF;
        }
        __syncthreads();
    }
}

// Stage 2: nprobe*ckb partial candidates per query -> global top-ck.
__global__ void merge_partial_kernel(
    const float* __restrict__ part_dist,
    const int*   __restrict__ part_pos,
    float*                    topck_primary,
    int*                      topck_pos,
    int nprobe, int ckb, int ck)
{
    const float INF   = __int_as_float(0x7F800000);
    const int   BLOCK = blockDim.x;
    const int   bqi   = blockIdx.x;
    const int   tid   = threadIdx.x;
    const int   n     = nprobe * ckb;

    extern __shared__ char shm[];
    float* s_val     = (float*)shm;              // n entries
    int*   s_idx     = (int*)(s_val + n);
    float* s_red_val = (float*)(s_idx + n);      // BLOCK
    int*   s_red_idx = (int*)(s_red_val + BLOCK);

    const long long base = (long long)bqi * n;
    for (int i = tid; i < n; i += BLOCK) {
        s_val[i] = part_dist[base + i];
        s_idx[i] = part_pos [base + i];
    }
    __syncthreads();

    float* out_primary = topck_primary + (long long)bqi * ck;
    int*   out_pos     = topck_pos     + (long long)bqi * ck;
    for (int c = 0; c < ck; ++c) {
        float bv = INF; int bi = -1;
        for (int i = tid; i < n; i += BLOCK)
            if (s_val[i] < bv) { bv = s_val[i]; bi = i; }
        s_red_val[tid] = bv; s_red_idx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && s_red_val[tid+stride] < s_red_val[tid]) {
                s_red_val[tid] = s_red_val[tid+stride];
                s_red_idx[tid] = s_red_idx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = s_red_idx[0];
            out_primary[c] = (w >= 0) ? s_val[w] : INF;
            out_pos    [c] = (w >= 0) ? s_idx[w] : -1;
            if (w >= 0) s_val[w] = INF;
        }
        __syncthreads();
    }
}

// ── Residual / final-topk kernels (same as v10) ───────────────────────────────
// d_res_c1d is [M][Kr] -- one scalar codebook per subspace, as in the official
// get_scalar_codebook_ptr(subspace_idx, level). Dimension `dim` belongs to
// subspace dim/Ds.
__global__ void build_residual_lut_batched_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_res_c1d,
    float*                    d_lut_r,
    int B, int d, int Ds, int Kr)
{
    long long total = (long long)B * d * Kr;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        int j     = local % Kr;
        int dim   = local / Kr;
        float diff = d_q_rot[(long long)bqi * d + dim]
                   - d_res_c1d[(long long)(dim / Ds) * Kr + j];
        d_lut_r[i] = diff * diff;
    }
}

__global__ void residual_refine_batched_kernel(
    const int*     __restrict__ topck_pos,
    const float*   __restrict__ topck_primary,
    const float*   __restrict__ lut_r,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int ck, int d, int Kr, int Br, int bpv, int B)
{
    long long total = (long long)B * ck;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int pos = topck_pos[i];
        if (pos < 0) { comp_dists[i] = __int_as_float(0x7F800000); continue; }
        int bqi = (int)(i / ck);
        const float*   my_lut_r = lut_r + (long long)bqi * d * Kr;
        const uint8_t* rc       = list_res + (long long)pos * bpv;
        float d_res = 0.0f;
        for (int j = 0; j < d; ++j) {
            int ri = (Br == 4)
                ? ((j % 2 == 0) ? (rc[j/2] & 0x0F) : (rc[j/2] >> 4))
                : rc[j];
            d_res += my_lut_r[(long long)j * Kr + ri];
        }
        comp_dists[i] = topck_primary[i] + d_res + list_corr[pos];
    }
}

__global__ void batched_topk_final_kernel(
    const float* __restrict__ comp_dists,
    const int*   __restrict__ topck_pos,
    const int*   __restrict__ list_ids,
    float*                    final_dists,
    int*                      final_ids,
    int ck, int k, int B)
{
    const int   BLOCK = blockDim.x;
    const float INF   = __int_as_float(0x7F800000);
    int bqi = blockIdx.x;
    int tid = threadIdx.x;
    if (bqi >= B) return;

    extern __shared__ char shm[];
    float* s_dists = (float*)shm;
    int*   s_pos   = (int*)(s_dists + ck);
    float* red_val = (float*)(s_pos + ck);
    int*   red_idx = (int*)(red_val + BLOCK);

    const float* my_dists = comp_dists + (long long)bqi * ck;
    const int*   my_pos   = topck_pos  + (long long)bqi * ck;
    for (int i = tid; i < ck; i += BLOCK) { s_dists[i] = my_dists[i]; s_pos[i] = my_pos[i]; }
    __syncthreads();

    float* out_dists = final_dists + (long long)bqi * k;
    int*   out_ids   = final_ids   + (long long)bqi * k;

    for (int r = 0; r < k; ++r) {
        float bv = INF; int bi = -1;
        for (int i = tid; i < ck; i += BLOCK)
            if (s_dists[i] < bv) { bv = s_dists[i]; bi = i; }
        red_val[tid] = bv; red_idx[tid] = bi; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride && red_val[tid+stride] < red_val[tid]) {
                red_val[tid] = red_val[tid+stride]; red_idx[tid] = red_idx[tid+stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = red_idx[0]; int pos = (w >= 0) ? s_pos[w] : -1;
            out_ids[r]   = (pos >= 0) ? list_ids[pos] : -1;
            out_dists[r] = (w  >= 0)  ? s_dists[w]   : INF;
            if (w >= 0) s_dists[w] = INF;
        }
        __syncthreads();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static void realloc_ck_buffers(SearchWorkspace& ws, int batch_cap, int ck, int k,
                               int nprobe) {
    // Stage-1 partials: batch * nprobe * ckb. At batch 256, nprobe 128 and
    // the ckb floor of 32 that is 8.4 MB, and it shrinks as nprobe drops
    // because ckb grows only to ck.
    {
        const int ckb = scan_ckb(ck, nprobe);
        const long long need = (long long)batch_cap * nprobe * ckb;
        if (need > ws.part_cap) {
            cudaFree(ws.d_part_dist); ws.d_part_dist = nullptr;
            cudaFree(ws.d_part_pos);  ws.d_part_pos  = nullptr;
            CUDA_CHECK(cudaMalloc(&ws.d_part_dist, need * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&ws.d_part_pos,  need * sizeof(int)));
            ws.part_cap = (int)need;
        }
    }
    if (ck > ws.ck_cap) {
        cudaFree(ws.d_topck_pos);     ws.d_topck_pos     = nullptr;
        cudaFree(ws.d_topck_primary); ws.d_topck_primary = nullptr;
        cudaFree(ws.d_comp_dists);    ws.d_comp_dists    = nullptr;
        long long n = (long long)batch_cap * ck;
        CUDA_CHECK(cudaMalloc(&ws.d_topck_pos,     n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_topck_primary, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_comp_dists,    n * sizeof(float)));
        ws.ck_cap = ck;
    }
    if (k > ws.k_cap) {
        cudaFree(ws.d_final_ids);   ws.d_final_ids   = nullptr;
        cudaFree(ws.d_final_dists); ws.d_final_dists = nullptr;
        long long n = (long long)batch_cap * k;
        CUDA_CHECK(cudaMalloc(&ws.d_final_ids,   n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws.d_final_dists, n * sizeof(float)));
        ws.k_cap = k;
    }
}

static void capture_graph(
    SearchWorkspace& ws,
    cublasHandle_t cublas,
    const float*   d_Pi, const float* d_cent, const float* d_res_c1d,
    const float*   d_centroids, const float* d_cent_norms,
    const int*     d_list_offsets, const int* d_list_ids,
    const uint8_t* d_list_primary_t, const uint8_t* d_list_res,
    const float*   d_list_corr,
    int B, int d, int M, int Ds, int K, int Kr,
    int nlist, int nprobe, int Br, int bpv,
    int ck, int k, int ntotal)
{
    if (ws.graph_exec) { cudaGraphExecDestroy(ws.graph_exec); ws.graph_exec = nullptr; }
    if (ws.graph)      { cudaGraphDestroy(ws.graph);          ws.graph      = nullptr; }

    const float one = 1.0f, zero = 0.0f;
    const int   BLOCK = 256;
    // Must track JHQ_K_LOCAL exactly -- the kernel indexes s_cdist/s_cpos as
    // K_LOCAL*BLOCK each, so a stale literal here silently corrupts memory
    // rather than failing to build.
    const int   scan_smem = (2 * JHQ_K_LOCAL * BLOCK + 2 * BLOCK) * (int)sizeof(float);
    const int   topk_smem = (2 * ck + 2 * BLOCK) * (int)sizeof(float);

    {
        int smem_max = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&smem_max,
                       cudaDevAttrMaxSharedMemoryPerBlock, 0));
        const int ckb_chk = scan_ckb(ck, nprobe);
        const long long merge_need =
            (2LL * nprobe * ckb_chk + 2LL * BLOCK) * (long long)sizeof(float);
        if (merge_need > smem_max)
            throw std::runtime_error(
                "merge_partial_kernel needs " + std::to_string(merge_need) +
                " B of shared memory (nprobe=" + std::to_string(nprobe) +
                ", ckb=" + std::to_string(ckb_chk) + ") but the device allows " +
                std::to_string(smem_max) + " B per block.");
        if (scan_smem > smem_max)
            throw std::runtime_error(
                "scan_ivf_coalesced_kernel needs " + std::to_string(scan_smem) +
                " B of shared memory (JHQ_K_LOCAL=" + std::to_string(JHQ_K_LOCAL) +
                ") but the device allows " + std::to_string(smem_max) +
                " B per block without opt-in.");
    }

    CUDA_CHECK(cudaStreamBeginCapture(ws.stream, cudaStreamCaptureModeGlobal));

    // 1. Rotate
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             d, B, d, &one, d_Pi, d, ws.d_q_batch, d, &zero, ws.d_q_rot, d));
    // 2. Centroid dots
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             nlist, B, d, &one, d_centroids, d, ws.d_q_rot, d, &zero, ws.d_dots, nlist));
    // 3. Select probes
    select_probes_kernel<<<B, BLOCK, (nlist + 2*BLOCK) * (int)sizeof(float), ws.stream>>>(
        ws.d_dots, d_cent_norms, d_list_offsets,
        ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total, nlist, nprobe);
    // 4. Build byte LUT
    {
        long long tot  = (long long)B * M * 256;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_byte_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_cent, ws.d_byte_lut, B, d, M, Ds, K);
    }
    // 5. Scan IVF, split over (query, probe) so the grid fills the device
    {
        const int ckb = scan_ckb(ck, nprobe);
        scan_ivf_partial_kernel<<<dim3(B, nprobe), BLOCK, scan_smem, ws.stream>>>(
            ws.d_byte_lut, ws.d_probe_ids, d_list_offsets,
            d_list_primary_t, ws.d_part_dist, ws.d_part_pos,
            nprobe, M, ntotal, ckb);
        const int merge_smem =
            (int)((2 * (long long)nprobe * ckb + 2 * BLOCK) * (long long)sizeof(float));
        merge_partial_kernel<<<B, BLOCK, merge_smem, ws.stream>>>(
            ws.d_part_dist, ws.d_part_pos,
            ws.d_topck_primary, ws.d_topck_pos, nprobe, ckb, ck);
    }
    // 6. Residual LUT
    {
        long long tot  = (long long)B * d * Kr;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_residual_lut_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_res_c1d, ws.d_lut_r, B, d, Ds, Kr);
    }
    // 7. Residual refine
    {
        long long tot  = (long long)B * ck;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        residual_refine_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_topck_pos, ws.d_topck_primary,
            ws.d_lut_r, d_list_res, d_list_corr,
            ws.d_comp_dists, ck, d, Kr, Br, bpv, B);
    }
    // 8. Final top-k
    batched_topk_final_kernel<<<B, BLOCK, topk_smem, ws.stream>>>(
        ws.d_comp_dists, ws.d_topck_pos, d_list_ids,
        ws.d_final_dists, ws.d_final_ids, ck, k, B);

    CUDA_CHECK(cudaStreamEndCapture(ws.stream, &ws.graph));
    CUDA_CHECK(cudaGraphInstantiate(&ws.graph_exec, ws.graph, nullptr, nullptr, 0));

    ws.graph_ck     = ck;
    ws.graph_nprobe = nprobe;
}

void search_gpu(
    cublasHandle_t cublas,
    const float* d_Pi, const float* d_cent, const float* d_res_c1d,
    const float* d_centroids, const float* d_cent_norms,
    const int* d_list_offsets, const int* d_list_ids,
    const uint8_t* d_list_primary_t, const uint8_t* d_list_res,
    const float* d_list_corr,
    const float* h_queries,
    int nq, int d, int M, int Ds, int K, int Kr,
    int nlist, int nprobe, int Br, int bpv,
    float alpha, int k, int batch_size,
    int ntotal,
    SearchWorkspace& ws,
    float* h_out_dists, int* h_out_ids)
{
    if (ws.batch_cap <= 0)
        throw std::runtime_error("v12: workspace not initialised — call add() first");
    if (!ws.stream || !ws.h_q_pinned)
        throw std::runtime_error("v12: CUDA stream / pinned buffer not created");

    int ck = std::max(k, (int)std::ceil(alpha * (float)k));
    realloc_ck_buffers(ws, ws.batch_cap, ck, k, nprobe);

    const int B_full = ws.batch_cap;
    if (!ws.graph_exec || ws.graph_ck != ck || ws.graph_nprobe != nprobe) {
        capture_graph(ws, cublas,
                      d_Pi, d_cent, d_res_c1d, d_centroids, d_cent_norms,
                      d_list_offsets, d_list_ids, d_list_primary_t, d_list_res, d_list_corr,
                      B_full, d, M, Ds, K, Kr, nlist, nprobe, Br, bpv,
                      ck, k, ntotal);
    }

    for (int qoff = 0; qoff < nq; qoff += batch_size) {
        int B = std::min(batch_size, nq - qoff);
        std::memcpy(ws.h_q_pinned,
                    h_queries + (long long)qoff * d, (long long)B * d * sizeof(float));
        if (B < B_full)
            std::memset(ws.h_q_pinned + (long long)B * d, 0,
                        (long long)(B_full - B) * d * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                                   (long long)B_full * d * sizeof(float),
                                   cudaMemcpyHostToDevice, ws.stream));
        CUDA_CHECK(cudaGraphLaunch(ws.graph_exec, ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_out_ids   + (long long)qoff * k,
                                   ws.d_final_ids,
                                   (long long)B * k * sizeof(int),
                                   cudaMemcpyDeviceToHost, ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_out_dists + (long long)qoff * k,
                                   ws.d_final_dists,
                                   (long long)B * k * sizeof(float),
                                   cudaMemcpyDeviceToHost, ws.stream));
    }

    cudaError_t err;
    do { err = cudaStreamQuery(ws.stream); } while (err == cudaErrorNotReady);
    CUDA_CHECK(err);
}

} // namespace jhq_gpu
