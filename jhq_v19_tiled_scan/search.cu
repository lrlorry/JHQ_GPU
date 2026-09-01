#include "jhq_v19_tiled_scan/search.cuh"
#include <cuda_fp16.h>
#include <cstdio>

// Stage timing is compiled out by default; -DJHQ_STEP_TIMING=1 turns it on.
#ifndef JHQ_STEP_TIMING
#define JHQ_STEP_TIMING 0
#endif

// Ablation switch, not a correctness option: skips the in-scan top-ck
// selection and writes the first ck per-thread candidates instead. Results are
// wrong with it on. It exists to price that selection, which the stage timing
// implicates -- fitting t = R + c*per to the two nprobe points puts 15.77 ms
// per batch in a term that does not grow with the candidate count, 95% of the
// time at nprobe=8 and 56% at nprobe=128, while the scan itself costs only
// 0.41 ns per candidate (235 GB/s at 96 B each).
//   -DJHQ_ABLATE_INSCAN_TOPK=1
#ifndef JHQ_ABLATE_INSCAN_TOPK
#define JHQ_ABLATE_INSCAN_TOPK 0
#endif

// Inner-loop ablations. All of them produce wrong results; they exist to price
// the three things the loop does per candidate, one at a time, after the
// in-scan top-k turned out to cost nothing (removing it changed 16.53 -> 16.84
// and 27.98 -> 28.56 ms, i.e. nothing).
//   1  JHQ_ABLATE_LUT      table lookup replaced by a constant, codes still read
//   2  JHQ_ABLATE_ACC      no lookup and no accumulate, codes still read
//   3  JHQ_ABLATE_INSERT   per-thread top-4 insertion sort removed
// Tile shape, swept in v20. TILE_C independent accumulate chains per thread,
// TILE_M subspaces per pass over them.
#ifndef JHQ_TILE_C
#define JHQ_TILE_C 4
#endif
#ifndef JHQ_TILE_M
#define JHQ_TILE_M 8
#endif
#ifndef JHQ_ABLATE_LUT
#define JHQ_ABLATE_LUT 0
#endif
#ifndef JHQ_ABLATE_ACC
#define JHQ_ABLATE_ACC 0
#endif
#ifndef JHQ_ABLATE_INSERT
#define JHQ_ABLATE_INSERT 0
#endif

// Ablating the table lookup alone cut the scan 47% (16.54 -> 8.87 ms at
// nprobe=8, 28.41 -> 14.95 at 128) while also dropping the accumulate saved
// only 2% more, so the cost is the indirect access, not the dependent chain.
// The table is already in shared memory at M=96, which leaves bank conflicts:
// the 32 lanes of a warp hold unrelated code bytes and index the same 512-byte
// half table at random, and 2-byte elements put two lanes per bank to begin
// with.
//
// This switch stores the table as float in shared instead. Four-byte elements
// map one per bank, so a conflict needs two lanes on the same bank rather than
// merely nearby. It doubles the shared footprint, which is why half was chosen;
// if the scan gets faster despite that, conflicts are the mechanism.
//   -DJHQ_SMEM_LUT_FLOAT=1
#ifndef JHQ_SMEM_LUT_FLOAT
#define JHQ_SMEM_LUT_FLOAT 0
#endif

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
    __half*                   d_byte_lut,
    int B, int d, int M, int Ds, int K)
{
    long long total = (long long)B * M * 256;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)M * 256));
        int local = (int)(i % ((long long)M * 256));
        int m     = local / 256;
        int c     = local % 256;

        if (c >= K) { d_byte_lut[i] = __float2half(65504.0f); continue; }  // half max

        const float* q_m = d_q_rot + (long long)bqi * d + (long long)m * Ds;
        const float* cc  = d_cent  + ((long long)m * K + c) * Ds;
        float sum = 0.0f;
        for (int j = 0; j < Ds; ++j) { float t = q_m[j] - cc[j]; sum += t * t; }
        d_byte_lut[i] = __float2half(sum);
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
// The lookup table is 96*256 half = 48 KB per query, and v16 re-read it from
// global memory once per subspace per candidate: 96 lookups of 4 B against
// 96 B of codes, so 384 of every 480 bytes moved were the same table being
// fetched again. At nprobe=128 that is 56 MB per query rather than 11.6, which
// is why the measured 9.4% of peak was misleading -- the device was near
// bandwidth saturation, just spending it on repetition.
//
// Staging the table in shared memory once per block leaves 96 B per candidate.
// In half it fits: 48 KB for M=96 plus the candidate scratch, opted into via
// cudaFuncSetAttribute. When it does not fit (large M), s_lut is null and the
// kernel reads from global as before.
#if JHQ_SMEM_LUT_FLOAT
#define LUT_AT(p, i) ((p)[(i)])
#else
#define LUT_AT(p, i) __half2float((p)[(i)])
#endif

__global__ void scan_ivf_coalesced_kernel(
    const __half*  __restrict__ d_byte_lut,     // [B, M, 256]
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary_t, // [M, N] transposed
    const int*     __restrict__ query_total,
    float*                      topck_primary,
    int*                        topck_pos,
    int nprobe, int M, int N, int ck, bool lut_in_smem)
{
    constexpr int K_LOCAL = JHQ_K_LOCAL;
    const float   INF     = __int_as_float(0x7F800000);
    const int     BLOCK   = blockDim.x;
    int bqi = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ char shm[];
    float* s_cdist   = (float*)shm;
    int*   s_cpos    = (int*)(s_cdist + K_LOCAL * BLOCK);
    float* s_red_val = (float*)(s_cpos + K_LOCAL * BLOCK);
    int*   s_red_idx = (int*)(s_red_val + BLOCK);
#if JHQ_SMEM_LUT_FLOAT
    float* s_lut = (float*)(s_red_idx + BLOCK);
#else
    __half* s_lut = (__half*)(s_red_idx + BLOCK);
#endif

    const __half* g_lut = d_byte_lut + (long long)bqi * M * 256;
    if (lut_in_smem) {
        for (int i = tid; i < M * 256; i += BLOCK)
#if JHQ_SMEM_LUT_FLOAT
            s_lut[i] = __half2float(g_lut[i]);
#else
            s_lut[i] = g_lut[i];
#endif
        __syncthreads();
    }
    // Whichever copy is live; LUT_AT converts if it is half.
#if JHQ_SMEM_LUT_FLOAT
    const float*  my_lut_s = s_lut;
#else
    const __half* my_lut_s = s_lut;
#endif

    int total      = query_total[bqi];
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);

    float ld[K_LOCAL]; int lp[K_LOCAL];
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) { ld[i] = INF; lp[i] = -1; }

    // Tiled over subspaces, several candidates at a time.
    //
    // v18 walked all M subspaces for one candidate before moving to the next,
    // so every iteration issued an independent random index into the whole
    // 48 KB table and the next accumulate waited on it. Ablating that lookup
    // cut the scan 47% (16.54 -> 8.87 ms at nprobe=8); ablating the accumulate
    // as well saved 2% more, so the cost is the access, not the chain. Storing
    // the table as float to widen the bank stride needs 106 KB against this
    // device's 99 KB opt-in limit, so that route is closed at M=96 and shut
    // entirely at larger M.
    //
    // Here a thread holds TILE_C candidates and sweeps TILE_M subspaces at a
    // time. Within a tile every lane indexes the same TILE_M*256 slice -- 4 KB
    // at TILE_M=8 -- instead of the full table, and the TILE_C partial sums are
    // independent, so their loads overlap rather than serialise. That is also
    // the answer to the other half of the measurement: 16x the candidates cost
    // only 1.69x the time because at nprobe=8 each thread ran 28 iterations,
    // too few to fill the pipeline. TILE_C independent chains fill it at any
    // nprobe.
    constexpr int TILE_C = JHQ_TILE_C;
    constexpr int TILE_M = JHQ_TILE_M;

    for (int base = tid * TILE_C; base < total; base += BLOCK * TILE_C) {
        float acc[TILE_C];
        int   pos[TILE_C];
        int   live = 0;
        #pragma unroll
        for (int t = 0; t < TILE_C; ++t) {
            acc[t] = 0.0f;
            const int lt = base + t;
            if (lt < total) {
                // Same probe walk as v18, just done once per candidate up
                // front rather than inside the subspace loop.
                int p = 0;
                while (p + 1 < nprobe && lt >= my_offs[p + 1]) ++p;
                pos[t] = list_offsets[my_ids[p]] + (lt - my_offs[p]);
                live = t + 1;
            } else {
                pos[t] = -1;
            }
        }

        for (int m0 = 0; m0 < M; m0 += TILE_M) {
            const int mhi = (m0 + TILE_M < M) ? (m0 + TILE_M) : M;
            #pragma unroll
            for (int t = 0; t < TILE_C; ++t) {
                if (t >= live) break;
                float a = acc[t];
                for (int m = m0; m < mhi; ++m) {
                    uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos[t]]);
                    a += lut_in_smem ? LUT_AT(my_lut_s, m * 256 + cm)
                                     : __half2float(g_lut[m * 256 + cm]);
                }
                acc[t] = a;
            }
        }

        #pragma unroll
        for (int t = 0; t < TILE_C; ++t) {
            if (t >= live) break;
            float dist = acc[t];
            if (dist < ld[K_LOCAL - 1]) {
                ld[K_LOCAL - 1] = dist;
                lp[K_LOCAL - 1] = pos[t];
                #pragma unroll
                for (int i = K_LOCAL - 1; i > 0 && ld[i] < ld[i-1]; --i) {
                    float td = ld[i-1]; ld[i-1] = ld[i]; ld[i] = td;
                    int   tp = lp[i-1]; lp[i-1] = lp[i]; lp[i] = tp;
                }
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) {
        s_cdist[tid * K_LOCAL + i] = ld[i];
        s_cpos [tid * K_LOCAL + i] = lp[i];
    }
    __syncthreads();

    const int n_cands = K_LOCAL * BLOCK;
    float* out_primary = topck_primary + (long long)bqi * ck;
    int*   out_pos     = topck_pos     + (long long)bqi * ck;

#if JHQ_ABLATE_INSCAN_TOPK
    for (int c = tid; c < ck; c += BLOCK) {
        out_primary[c] = s_cdist[c % n_cands];
        out_pos    [c] = s_cpos [c % n_cands];
    }
    return;
#endif
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
    // Candidate scratch, then the staged lookup table if it fits. Blackwell
    // allows well past the 48 KB default per block, but only after opting in
    // per kernel, so ask for the limit first and fall back to the global-memory
    // path when even that is not enough.
    const int   scan_base = (2 * JHQ_K_LOCAL * BLOCK + 2 * BLOCK) * (int)sizeof(float);
    const int   lut_bytes = M * 256 * (int)sizeof(__half);
    int         smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin,
                   cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const bool  lut_in_smem = (scan_base + lut_bytes) <= smem_optin;
    const int   scan_smem = lut_in_smem ? (scan_base + lut_bytes) : scan_base;
    if (lut_in_smem)
        CUDA_CHECK(cudaFuncSetAttribute(scan_ivf_coalesced_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, scan_smem));
    const int   topk_smem = (2 * ck + 2 * BLOCK) * (int)sizeof(float);

    {
        int smem_max = smem_optin;
        if (scan_smem > smem_max)
            throw std::runtime_error(
                "scan_ivf_coalesced_kernel needs " + std::to_string(scan_smem) +
                " B of shared memory (JHQ_K_LOCAL=" + std::to_string(JHQ_K_LOCAL) +
                ") but the device allows " + std::to_string(smem_max) +
                " B per block without opt-in.");
    }

#if JHQ_STEP_TIMING
    // ncu cannot read performance counters in this container (ERR_NVGPUCTRPERM),
    // so stage attribution comes from events instead. They cannot be read back
    // from inside a capture, so the timed build skips the graph and launches
    // each batch directly -- it loses graph launch overhead, which is the same
    // for every stage and so does not distort the shares.
    for (int i = 0; i < SearchWorkspace::N_STEP_EVENTS; ++i)
        if (!ws.ev_step[i]) CUDA_CHECK(cudaEventCreate(&ws.ev_step[i]));
    CUDA_CHECK(cudaEventRecord(ws.ev_step[0], ws.stream));
#else
    CUDA_CHECK(cudaStreamBeginCapture(ws.stream, cudaStreamCaptureModeGlobal));
#endif

    // 1. Rotate
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             d, B, d, &one, d_Pi, d, ws.d_q_batch, d, &zero, ws.d_q_rot, d));
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[1], ws.stream));
#endif
    // 2. Centroid dots
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             nlist, B, d, &one, d_centroids, d, ws.d_q_rot, d, &zero, ws.d_dots, nlist));
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[2], ws.stream));
#endif
    // 3. Select probes
    select_probes_kernel<<<B, BLOCK, (nlist + 2*BLOCK) * (int)sizeof(float), ws.stream>>>(
        ws.d_dots, d_cent_norms, d_list_offsets,
        ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total, nlist, nprobe);
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[3], ws.stream));
#endif
    // 4. Build byte LUT
    {
        long long tot  = (long long)B * M * 256;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_byte_lut_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_cent, ws.d_byte_lut, B, d, M, Ds, K);
    }
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[4], ws.stream));
#endif
    // 5. Scan IVF — coalesced via [M, N] list_primary_t
    scan_ivf_coalesced_kernel<<<B, BLOCK, scan_smem, ws.stream>>>(
        ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
        d_list_primary_t, ws.d_query_total,
        ws.d_topck_primary, ws.d_topck_pos,
        nprobe, M, ntotal, ck, lut_in_smem);
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[5], ws.stream));
#endif
    // 6. Residual LUT
    {
        long long tot  = (long long)B * d * Kr;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        build_residual_lut_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_q_rot, d_res_c1d, ws.d_lut_r, B, d, Ds, Kr);
    }
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[6], ws.stream));
#endif
    // 7. Residual refine
    {
        long long tot  = (long long)B * ck;
        int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
        residual_refine_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
            ws.d_topck_pos, ws.d_topck_primary,
            ws.d_lut_r, d_list_res, d_list_corr,
            ws.d_comp_dists, ck, d, Kr, Br, bpv, B);
    }
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[7], ws.stream));
#endif
    // 8. Final top-k
    batched_topk_final_kernel<<<B, BLOCK, topk_smem, ws.stream>>>(
        ws.d_comp_dists, ws.d_topck_pos, d_list_ids,
        ws.d_final_dists, ws.d_final_ids, ck, k, B);

#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[8], ws.stream));
    CUDA_CHECK(cudaEventSynchronize(ws.ev_step[8]));
    for (int i = 0; i < SearchWorkspace::N_STEP_EVENTS - 1; ++i) {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ws.ev_step[i], ws.ev_step[i + 1]));
        ws.acc_ms[i] += ms;
    }
    ws.timing_count++;
#else
    CUDA_CHECK(cudaStreamEndCapture(ws.stream, &ws.graph));
    CUDA_CHECK(cudaGraphInstantiate(&ws.graph_exec, ws.graph, nullptr, nullptr, 0));
#endif

    ws.graph_ck     = ck;
    ws.graph_nprobe = nprobe;
}

#if JHQ_STEP_TIMING
static void report_step_timing(SearchWorkspace& ws, int B, int nprobe, int ck) {
    if (!ws.timing_count) return;
    static const char* names[] = {
        "gemm_rotate", "gemm_centroid", "select_probes", "build_byte_lut",
        "scan_ivf", "build_lut_r", "residual_refine", "topk_final"};
    double tot = 0.0;
    for (int i = 0; i < SearchWorkspace::N_STEP_EVENTS - 1; ++i) tot += ws.acc_ms[i];
    std::fprintf(stderr,
        "[STEP-TIMING  B=%d nprobe=%d ck=%d  batches=%d]\n", B, nprobe, ck,
        ws.timing_count);
    for (int i = 0; i < SearchWorkspace::N_STEP_EVENTS - 1; ++i)
        std::fprintf(stderr, "  %-16s %8.3f ms/batch  %5.1f%%\n",
                     names[i], ws.acc_ms[i] / ws.timing_count,
                     tot > 0 ? 100.0 * ws.acc_ms[i] / tot : 0.0);
    std::fprintf(stderr, "  %-16s %8.3f ms/batch\n", "TOTAL",
                 tot / ws.timing_count);
    for (int i = 0; i < SearchWorkspace::N_STEP_EVENTS - 1; ++i) ws.acc_ms[i] = 0.0;
    ws.timing_count = 0;
}
#endif

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
    realloc_ck_buffers(ws, ws.batch_cap, ck, k);

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
#if JHQ_STEP_TIMING
        // No graph to launch in the timed build: re-run the sequence so the
        // events land on this batch.
        capture_graph(ws, cublas,
                      d_Pi, d_cent, d_res_c1d, d_centroids, d_cent_norms,
                      d_list_offsets, d_list_ids, d_list_primary_t, d_list_res,
                      d_list_corr, B_full, d, M, Ds, K, Kr, nlist, nprobe,
                      Br, bpv, ck, k, ntotal);
#else
        CUDA_CHECK(cudaGraphLaunch(ws.graph_exec, ws.stream));
#endif
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
#if JHQ_STEP_TIMING
    report_step_timing(ws, B_full, nprobe, ck);
#endif
}

} // namespace jhq_gpu
