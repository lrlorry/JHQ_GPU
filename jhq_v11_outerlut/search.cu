#include "jhq_v11_outerlut/search.cuh"
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

// ── build_byte_lut_kernel (same as v10) ──────────────────────────────────────
__global__ void build_byte_lut_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_c1d,
    float*                    d_byte_lut,
    int B, int d, int M, int Ds, int K1D, int bits_per_dim)
{
    long long total = (long long)B * M * 256;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi      = (int)(i / ((long long)M * 256));
        int local    = (int)(i % ((long long)M * 256));
        int m        = local / 256;
        int byte_val = local % 256;

        const float* q_m  = d_q_rot + (long long)bqi * d + (long long)m * Ds;
        const int    kmask = K1D - 1;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= Ds) break;
            int   j    = (byte_val >> (k * bits_per_dim)) & kmask;
            float diff = q_m[k] - d_c1d[j];
            sum += diff * diff;
        }
        d_byte_lut[i] = sum;
    }
}

// ── scan_ivf_outerlut_kernel (NEW) ────────────────────────────────────────────
//
// Outer loop: m = 0..M-1 (codewords), chunked over candidates.
//
// For each chunk of up to MAX_CANDS candidates per thread:
//   1. Pre-compute abs_pos[MAX_CANDS] in registers.
//   2. Outer-m loop: 256 threads cooperatively load 256-entry sub-table (1 KB)
//      into shared memory, then all candidates in this chunk look up from shared
//      (1-cycle latency vs ~30-cycle L2 in v10, zero bank conflict).
//   3. Update global top-K_LOCAL from the chunk's partial distances.
//
// Chunking handles arbitrary nprobe/list sizes correctly: even if a query
// probes a very large list (total > MAX_CANDS × BLOCK), the outer chunk loop
// repeats until all candidates are covered.  For nprobe=8 with typical IVF
// balance, one chunk is enough, so the overhead is zero.
//
template <int MAX_CANDS>
__global__ void scan_ivf_outerlut_kernel(
    const float*   __restrict__ d_byte_lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary,
    const int*     __restrict__ query_total,
    float*                      topck_primary,
    int*                        topck_pos,
    int nprobe, int M, int ck)
{
    constexpr int K_LOCAL = 4;
    const float   INF     = __int_as_float(0x7F800000);
    const int     BLOCK   = blockDim.x;  // must be 256 (== sub-table size)
    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory layout (10 KB, same as v10):
    //   Scan phase:   s_sub[256]  — 1 KB (first 256 floats)
    //   Reduce phase: s_cdist[K_LOCAL*BLOCK], s_cpos[...], s_red_val[BLOCK], s_red_idx[BLOCK]
    // The scan phase uses only s_sub (1 KB), reused as s_cdist/s_cpos after.
    extern __shared__ char shm[];
    float* s_sub     = (float*)shm;
    float* s_cdist   = (float*)shm;                          // alias s_sub
    int*   s_cpos    = (int*)(s_cdist + K_LOCAL * BLOCK);
    float* s_red_val = (float*)(s_cpos + K_LOCAL * BLOCK);
    int*   s_red_idx = (int*)(s_red_val + BLOCK);

    const float* my_lut = d_byte_lut + (long long)bqi * M * 256;
    int total = query_total[bqi];
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);

    // Global top-K_LOCAL state — persists across all chunks.
    float ld[K_LOCAL]; int lp[K_LOCAL];
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) { ld[i] = INF; lp[i] = -1; }

    // ── Chunked scan: process MAX_CANDS candidates per thread per chunk ────
    //
    // Chunk c covers local_t in [c*MAX_CANDS*BLOCK, (c+1)*MAX_CANDS*BLOCK).
    // Thread tid handles local_t = chunk_base + j*BLOCK + tid, j=0..MAX_CANDS-1.
    //
    for (int chunk_base = 0; chunk_base < total; chunk_base += MAX_CANDS * BLOCK) {

        // Phase A: pre-compute abs_pos + init partial for this chunk
        int   my_abs_pos[MAX_CANDS];
        float partial[MAX_CANDS];
        int   n_my = 0;

        for (int j = 0; j < MAX_CANDS; ++j) {
            int local_t = chunk_base + j * BLOCK + tid;
            if (local_t >= total) break;
            int p = 0;
            while (p + 1 < nprobe && local_t >= my_offs[p + 1]) ++p;
            my_abs_pos[n_my] = list_offsets[my_ids[p]] + (local_t - my_offs[p]);
            partial[n_my]    = 0.0f;
            ++n_my;
        }

        // Phase B: outer-m loop with cooperative shared sub-table load
        for (int m = 0; m < M; ++m) {
            s_sub[tid] = my_lut[(long long)m * 256 + tid];
            __syncthreads();

            for (int j = 0; j < n_my; ++j) {
                uint8_t cm = list_primary[(long long)my_abs_pos[j] * M + m];
                partial[j] += s_sub[cm];
            }
            __syncthreads();
        }

        // Phase C: update global top-K_LOCAL from this chunk's partial distances
        for (int j = 0; j < n_my; ++j) {
            if (partial[j] < ld[K_LOCAL - 1]) {
                ld[K_LOCAL - 1] = partial[j];
                lp[K_LOCAL - 1] = my_abs_pos[j];
                #pragma unroll
                for (int i = K_LOCAL - 1; i > 0 && ld[i] < ld[i-1]; --i) {
                    float td = ld[i-1]; ld[i-1] = ld[i]; ld[i] = td;
                    int   tp = lp[i-1]; lp[i-1] = lp[i]; lp[i] = tp;
                }
            }
        }
    }

    // ── Write per-thread top-K_LOCAL to shared; global top-ck selection ───
    // Last __syncthreads() in the m-loop ensures s_sub is safe to reuse.
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) {
        s_cdist[tid * K_LOCAL + i] = ld[i];
        s_cpos [tid * K_LOCAL + i] = lp[i];
    }
    __syncthreads();

    const int n_cands = K_LOCAL * BLOCK;
    float* out_primary = topck_primary + (long long)bqi * ck;
    int*   out_pos     = topck_pos     + (long long)bqi * ck;

    for (int c = 0; c < ck; ++c) {
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
            out_primary[c] = (w >= 0) ? s_cdist[w] : INF;
            out_pos    [c] = (w >= 0) ? s_cpos [w] : -1;
            if (w >= 0) s_cdist[w] = INF;
        }
        __syncthreads();
    }
}

// ── scan_ivf_bytelut_kernel (v10 fallback for large nprobe) ──────────────────
__global__ void scan_ivf_bytelut_kernel(
    const float*   __restrict__ d_byte_lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary,
    const int*     __restrict__ query_total,
    float*                      topck_primary,
    int*                        topck_pos,
    int nprobe, int M, int ck)
{
    constexpr int K_LOCAL = 4;
    const float   INF     = __int_as_float(0x7F800000);
    const int     BLOCK   = blockDim.x;
    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ char shm[];
    float* s_cdist   = (float*)shm;
    int*   s_cpos    = (int*)(s_cdist + K_LOCAL * BLOCK);
    float* s_red_val = (float*)(s_cpos + K_LOCAL * BLOCK);
    int*   s_red_idx = (int*)(s_red_val + BLOCK);

    const float* my_lut = d_byte_lut + (long long)bqi * M * 256;

    int total      = query_total[bqi];
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);

    float ld[K_LOCAL]; int lp[K_LOCAL];
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) { ld[i] = INF; lp[i] = -1; }

    for (int local_t = tid; local_t < total; local_t += BLOCK) {
        int p = 0;
        while (p + 1 < nprobe && local_t >= my_offs[p + 1]) ++p;
        int abs_pos = list_offsets[my_ids[p]] + (local_t - my_offs[p]);

        const uint32_t* code4 = (const uint32_t*)(list_primary + (long long)abs_pos * M);
        float dist = 0.0f;
        #pragma unroll 4
        for (int m4 = 0; m4 < M / 4; ++m4) {
            uint32_t chunk = code4[m4];
            int base = m4 * 4;
            dist += my_lut[ base      * 256 + ( chunk        & 0xFF)];
            dist += my_lut[(base + 1) * 256 + ((chunk >>  8) & 0xFF)];
            dist += my_lut[(base + 2) * 256 + ((chunk >> 16) & 0xFF)];
            dist += my_lut[(base + 3) * 256 + ((chunk >> 24)       )];
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
    float* out_primary = topck_primary + (long long)bqi * ck;
    int*   out_pos     = topck_pos     + (long long)bqi * ck;

    for (int c = 0; c < ck; ++c) {
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
            out_primary[c] = (w >= 0) ? s_cdist[w] : INF;
            out_pos    [c] = (w >= 0) ? s_cpos [w] : -1;
            if (w >= 0) s_cdist[w] = INF;
        }
        __syncthreads();
    }
}

// ── Residual / final-topk kernels (same as v10) ───────────────────────────────
__global__ void build_residual_lut_batched_kernel(
    const float* __restrict__ d_q_rot,
    const float* __restrict__ d_res_c1d,
    float*                    d_lut_r,
    int B, int d, int Kr)
{
    long long total = (long long)B * d * Kr;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)d * Kr));
        int local = (int)(i % ((long long)d * Kr));
        int j     = local % Kr;
        int dim   = local / Kr;
        float diff = d_q_rot[(long long)bqi * d + dim] - d_res_c1d[j];
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
static void realloc_ck_buffers(SearchWorkspace& ws, int batch_cap, int ck, int k) {
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

// MAX_CANDS per chunk: controls the register footprint and L1 pressure.
//
// With BLOCK=256 and nprobe=8 (total≈7280): ~29 candidates per thread,
// so ceil(29/MAX_CANDS) chunks are processed.
//
// Trade-off:
//   Large MAX_CANDS → fewer chunks (less cooperative-load overhead) but
//     more concurrent cache-line footprint in L1 per block, causing
//     list_primary L1 thrash and lower SM occupancy.
//   Small MAX_CANDS → more chunks but each chunk's L1 footprint is tiny,
//     registers drop to ~36/thread → 7 blocks/SM (same as v10).
//
// With MAX_CANDS=8:
//   - chunks=4 for nprobe=8  (≤29 cands/thread)
//   - registers ≈ 36/thread  → 7 blocks/SM, 112 KB L1 pressure < 128 KB ✓
//   - lut HBM traffic ≈ 5× less than v10 (96 KB per query loaded once into L2,
//     reused across all m-iterations within a chunk)
static constexpr int MAX_CANDS = 8;

// ── capture_graph ─────────────────────────────────────────────────────────────
static void capture_graph(
    SearchWorkspace& ws,
    cublasHandle_t cublas,
    const float*   d_Pi, const float* d_c1d, const float* d_res_c1d,
    const float*   d_centroids, const float* d_cent_norms,
    const int*     d_list_offsets, const int* d_list_ids,
    const uint8_t* d_list_primary, const uint8_t* d_list_res,
    const float*   d_list_corr,
    int B, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe, int Br, int bpv, int bits_per_dim,
    int ck, int k)
{
    if (ws.graph_exec) { cudaGraphExecDestroy(ws.graph_exec); ws.graph_exec = nullptr; }
    if (ws.graph)      { cudaGraphDestroy(ws.graph);          ws.graph      = nullptr; }

    const float one = 1.0f, zero = 0.0f;
    const int   BLOCK = 256;
    // Shared for scan: max(s_sub=1KB, reduction=10KB) = 10KB (same as v10)
    const int   scan_smem = (2 * 4 * BLOCK + 2 * BLOCK) * (int)sizeof(float); // 10240
    const int   topk_smem = (2 * ck + 2 * BLOCK) * (int)sizeof(float);

    CUDA_CHECK(cudaStreamBeginCapture(ws.stream, cudaStreamCaptureModeGlobal));

    // 1. Rotate queries
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             d, B, d, &one, d_Pi, d, ws.d_q_batch, d, &zero, ws.d_q_rot, d));
    // 2. Centroid dots
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             nlist, B, d, &one, d_centroids, d, ws.d_q_rot, d, &zero, ws.d_dots, nlist));
    // 3. Select probes
    select_probes_kernel<<<B, BLOCK, (nlist + 2*BLOCK) * (int)sizeof(float), ws.stream>>>(
        ws.d_dots, d_cent_norms, d_list_offsets,
        ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total, nlist, nprobe);
    // 4. Build byte LUT  [B × M × 256]
    {
        long long tot  = (long long)B * M * 256;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_byte_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_c1d, ws.d_byte_lut, B, d, M, Ds, K1D, bits_per_dim);
    }
    // 5. Scan IVF — chunked outer-m, correct for any nprobe / list size
    scan_ivf_outerlut_kernel<MAX_CANDS><<<B, BLOCK, scan_smem, ws.stream>>>(
        ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
        d_list_primary, ws.d_query_total,
        ws.d_topck_primary, ws.d_topck_pos,
        nprobe, M, ck);
    // 6. Residual LUT
    {
        long long tot  = (long long)B * d * Kr;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_residual_lut_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_res_c1d, ws.d_lut_r, B, d, Kr);
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

// ── search_gpu ────────────────────────────────────────────────────────────────
void search_gpu(
    cublasHandle_t cublas,
    const float* d_Pi, const float* d_c1d, const float* d_res_c1d,
    const float* d_centroids, const float* d_cent_norms,
    const int* d_list_offsets, const int* d_list_ids,
    const uint8_t* d_list_primary, const uint8_t* d_list_res,
    const float* d_list_corr,
    const float* h_queries,
    int nq, int d, int M, int Ds, int K1D, int Kr,
    int nlist, int nprobe, int Br, int bpv, int bits_per_dim,
    float alpha, int k, int batch_size,
    SearchWorkspace& ws,
    float* h_out_dists, int* h_out_ids)
{
    if (ws.batch_cap <= 0)
        throw std::runtime_error("v11: workspace not initialised — call add() first");
    if (!ws.stream || !ws.h_q_pinned)
        throw std::runtime_error("v11: CUDA stream / pinned buffer not created");

    int ck = std::max(k, (int)std::ceil(alpha * (float)k));
    realloc_ck_buffers(ws, ws.batch_cap, ck, k);

    const int B_full = ws.batch_cap;
    if (!ws.graph_exec || ws.graph_ck != ck || ws.graph_nprobe != nprobe) {
        capture_graph(ws, cublas,
                      d_Pi, d_c1d, d_res_c1d, d_centroids, d_cent_norms,
                      d_list_offsets, d_list_ids, d_list_primary, d_list_res, d_list_corr,
                      B_full, d, M, Ds, K1D, Kr, nlist, nprobe, Br, bpv, bits_per_dim, ck, k);
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
