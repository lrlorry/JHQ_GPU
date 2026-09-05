#include "jhq_v37_host_resid/search.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <string>

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
// Cascaded scan (v21).
//
// v19 pays all M subspaces for every candidate. But the primary code is a
// product quantiser: any prefix of the subspaces is already an unbiased,
// lower-variance estimate of the full distance, so most candidates can be
// rejected long before the last subspace is read. Ablating the table lookup
// cut the scan 47%, so the lookups are the cost and fewer of them is the win.
//
// Pass 1 sweeps subspaces [0, PM) over every candidate and keeps the best
// JHQ_PREFIX_KEEP per thread by that partial sum. Pass 2 completes only those,
// resuming from the partial rather than restarting. At PM = M/4, KEEP = 8 and
// the measured candidate counts (7168 per query at nprobe=8, 114700 at 128)
// that is 319k lookups against 688k, and 2.9M against 11.0M -- 2.2x and 3.8x.
//
// PM is a fraction of M so it tracks the codebook: NUM/DEN, default a quarter.
// Setting NUM == DEN turns the cascade off and restores v19 exactly.
#ifndef JHQ_PREFIX_NUM
#define JHQ_PREFIX_NUM 1
#endif
#ifndef JHQ_PREFIX_DEN
#define JHQ_PREFIX_DEN 4
#endif
#ifndef JHQ_PREFIX_KEEP
#define JHQ_PREFIX_KEEP (2 * JHQ_K_LOCAL)
#endif
#ifndef JHQ_TILE_C
#define JHQ_TILE_C 1
#endif
#ifndef JHQ_TILE_M
#define JHQ_TILE_M 8
#endif
// Measured neutral. Striding the tile does keep the recall that a contiguous
// tile loses (0.9411 at TILE_C=2 against 0.9411 at TILE_C=1), so the
// explanation for that loss was right -- adjacent candidates landing on one
// thread -- but it buys no throughput either: 46192 QPS against 46322 at
// nprobe=128. At BLOCK=1024 the occupancy already fills the pipeline, so the
// extra independent chains have nothing to hide. Kept for the record; TILE_C
// stays 1.
#ifndef JHQ_NO_RESIDUAL
#define JHQ_NO_RESIDUAL 0
#endif
#ifndef JHQ_TILE_STRIDED
#define JHQ_TILE_STRIDED 0
#endif
// On by default: sorting the candidate array once holds recall to four
// decimal places against the ck-pass selection it replaces (0.7617 -> 0.7616,
// 0.8956 -> 0.8956, 0.9411 -> 0.9413) and lifts QPS 46% at nprobe=128, 77% at
// 32 and 90% at 8. The gain is largest where there is least to scan because
// the cost it removes -- ck block-wide reductions -- does not scale with
// nprobe. Set to 0 to measure against the old path.
#ifndef JHQ_BITONIC_SELECT
#define JHQ_BITONIC_SELECT 1
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
// Element type of the primary lookup table.
//
// Equation 6 sums the per-subspace distances d(q^m, yhat^(m)) exactly. Storing
// each as a half rounds it to eleven mantissa bits before it is ever added --
// an error term the paper's bound does not carry. Half stays the default: it
// halves the table traffic and lets M*256 entries sit in shared memory. But it
// is a departure from the paper's arithmetic, so it gets a switch and a
// measurement rather than silence. JHQ_LUT32=1 builds the table in float; at
// M=96 that is 98 KB, past shared memory, so the scan reads it from L2 and the
// cost of exactness shows up in QPS.
typedef jhq_gpu::jhq_lut_t lut_t;
#if JHQ_LUT32
#define LUT_STORE(x) (x)
#define LUT_LOAD(x)  (x)
#else
#define LUT_STORE(x) __float2half(x)
#define LUT_LOAD(x)  __half2float(x)
#endif

// Per-thread candidate slots kept by scan_ivf_coalesced_kernel. Compile-time
// so ld[]/lp[] stay in registers.
//
// This is a lossy step, and here is what it costs. Each thread keeps only its
// own best K_LOCAL, so when more than K_LOCAL of the true top-ck land in one
// thread's stride class the rest are dropped before the block-wide selection
// sees them.
//
// Measured by diffing the returned ids, not by watching recall: recall is an
// aggregate over a thousand queries and two different result sets can share
// one, so a flat recall across depths proves nothing. With the index pinned
// (JHQ_ENCODE_GROUPED_OFF=1, JHQ_INDEX_CACHE) and BLOCK held at 512, depth 4
// against depth 8 on vogue at nprobe=128 differs in 11 of 10,000 id positions,
// touching 4 queries in 1000, of which 1 gets a genuinely different set. So
// the top-alpha*k this produces is not always the global top-alpha*k.
//
// BLOCK=512 is the harder case on purpose: the expected number of the true
// top-ck per thread is ck/BLOCK, 1.95 there against 0.98 at the production
// BLOCK=1024. The production setting cannot be measured the same way, because
// depth is capped by registers, not by shared memory -- pd/pp/ld/lp are
// 4*K_LOCAL registers per thread against the 64 available when 1024 of them
// are resident, so K_LOCAL=8 does not launch at BLOCK=1024 at all. Four is not
// a tuning choice there; it is the only value that fits.
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
#include <cstdlib>
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
    lut_t*                    d_byte_lut,
    int B, int d, int M, int Ds, int K)
{
    long long total = (long long)B * M * 256;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        int bqi   = (int)(i / ((long long)M * 256));
        int local = (int)(i % ((long long)M * 256));
        int m     = local / 256;
        int c     = local % 256;

        if (c >= K) { d_byte_lut[i] = LUT_STORE(65504.0f); continue; }  // half max

        const float* q_m = d_q_rot + (long long)bqi * d + (long long)m * Ds;
        const float* cc  = d_cent  + ((long long)m * K + c) * Ds;
        float sum = 0.0f;
        for (int j = 0; j < Ds; ++j) { float t = q_m[j] - cc[j]; sum += t * t; }
        d_byte_lut[i] = LUT_STORE(sum);
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
#define LUT_AT(p, i) LUT_LOAD((p)[(i)])

// Exact top-ck selection, independent of how candidates fall across threads.
//
// Algorithm 1 line 4 selects the top-alpha*k of 𝒵 by complete primary distance,
// a global selection. The per-thread retention this replaces is not that: a
// thread holding more than K_LOCAL of the true top-ck drops the excess before
// the block ever merges, and whether that happens is a property of the data.
// Measured on vogue, depth 4 against depth 8 at BLOCK=512 differs in 1 query's
// result set per 1000.
//
// This keeps one shared buffer of capacity 2*ck for the whole block. A
// candidate is appended when its distance is below the current threshold; when
// the buffer fills, it is sorted, the best ck are kept, and the threshold
// becomes the ck-th value.
//
// Correctness: a candidate is discarded only if its distance is >= the ck-th
// best seen so far. The threshold is non-increasing as more candidates arrive,
// so it is always >= the ck-th best over the whole set. A discarded candidate
// therefore has at least ck candidates strictly better than it and cannot be
// in the final top-ck. Nothing here depends on the distribution over threads,
// on K_LOCAL, or on the order candidates are visited in.
//
// Cost: 2*ck*(4+4) bytes of shared memory -- 16 KB at ck=1000 -- and a sort
// each time the buffer fills. The buffer only fills when ck candidates have
// beaten the threshold, so on a set where most candidates are poor it fills
// rarely.
__device__ __forceinline__ void jhq_compact_topck(
    float* s_val, int* s_pos, int* s_cnt, float* s_thresh, int ck, int cap, int tid, int BLOCK)
{
    // Bitonic sort of the whole buffer, then keep the best ck.
    const int n = *s_cnt;
    const float INF = __int_as_float(0x7F800000);
    for (int i = n + tid; i < cap; i += BLOCK) { s_val[i] = INF; s_pos[i] = -1; }
    __syncthreads();
    for (int k2 = 2; k2 <= cap; k2 <<= 1)
        for (int j = k2 >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < cap; i += BLOCK) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool up = ((i & k2) == 0);
                    if ((s_val[i] > s_val[ixj]) == up) {
                        float t = s_val[i]; s_val[i] = s_val[ixj]; s_val[ixj] = t;
                        int   p = s_pos[i]; s_pos[i] = s_pos[ixj]; s_pos[ixj] = p;
                    }
                }
            }
            __syncthreads();
        }
    if (tid == 0) {
        *s_cnt    = (n < ck) ? n : ck;
        *s_thresh = (n >= ck) ? s_val[ck - 1] : INF;
    }
    __syncthreads();
}

// Debug dump for the differential test: every candidate's complete primary
// distance for one query, written in scan order. The test recomputes the exact
// top-ck from this and compares it against what the selector returned, so the
// selection is checked against the same distances the selector saw. A
// disagreement is then the selection, not arithmetic.
__global__ void dump_primary_distances_kernel(
    const lut_t*   __restrict__ d_byte_lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary_t,
    const int*     __restrict__ query_total,
    float* out_dist, int* out_pos, int cap_out,
    int bqi, int nprobe, int M, int N)
{
    const int total = query_total[bqi];
    const lut_t* g_lut = d_byte_lut + (long long)bqi * M * 256;
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);
    for (int lt = blockIdx.x * blockDim.x + threadIdx.x; lt < total;
         lt += gridDim.x * blockDim.x) {
        if (lt >= cap_out) return;
        int p = 0;
        while (p + 1 < nprobe && lt >= my_offs[p + 1]) ++p;
        const int pos = list_offsets[my_ids[p]] + (lt - my_offs[p]);
        float a = 0.0f;
        for (int m = 0; m < M; ++m) {
            const uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos]);
            a += LUT_LOAD(g_lut[m * 256 + cm]);
        }
        out_dist[lt] = a;
        out_pos[lt]  = pos;
    }
}

// The faithful scan: Algorithm 1 lines 2-4 with nothing between them.
//
// Every candidate gets its complete primary distance over all M subspaces --
// no prefix, no early discard -- and the top-ck is selected exactly by the
// threshold-compaction buffer above.
//
// The buffer holds ck + BLOCK entries, rounded up to a power of two for the
// sort. That size makes overflow structurally impossible rather than merely
// unlikely: the block processes candidates BLOCK at a time and compacts
// whenever the count exceeds ck, so at the start of a chunk the count is at
// most ck, at most BLOCK are appended, and the total cannot exceed ck + BLOCK.
// No candidate is ever dropped for want of room.
__global__ void scan_ivf_exact_kernel(
    const lut_t*   __restrict__ d_byte_lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary_t,
    const int*     __restrict__ query_total,
    float*                      topck_primary,
    int*                        topck_pos,
    int nprobe, int M, int N, int ck, int cap, bool lut_in_smem, int tile_m)
{
    const float INF   = __int_as_float(0x7F800000);
    const int   BLOCK = blockDim.x;
    const int   bqi   = blockIdx.x;
    const int   tid   = threadIdx.x;

    extern __shared__ char shm[];
    float* s_val    = (float*)shm;
    int*   s_pos    = (int*)(s_val + cap);
    int*   s_cnt    = (int*)(s_pos + cap);
    float* s_thresh = (float*)(s_cnt + 1);
    lut_t* s_lut    = (lut_t*)(s_thresh + 1);

    const lut_t* g_lut = d_byte_lut + (long long)bqi * M * 256;
    if (lut_in_smem) {
        for (int i = tid; i < M * 256; i += BLOCK) s_lut[i] = g_lut[i];
    }
    if (tid == 0) { *s_cnt = 0; *s_thresh = INF; }
    __syncthreads();

    const int  total   = query_total[bqi];
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);
    const int  TILE_M  = tile_m;

    // The probe a candidate belongs to only moves forward as lt grows, so the
    // search for it is carried across chunks instead of restarting at 0 for
    // every candidate.
    int p = 0;
    for (int chunk = 0; chunk < total; chunk += BLOCK) {
        const int lt = chunk + tid;
        float dist = INF;
        int   pos  = -1;
        if (lt < total) {
            int pp = p;
            while (pp + 1 < nprobe && lt >= my_offs[pp + 1]) ++pp;
            pos = list_offsets[my_ids[pp]] + (lt - my_offs[pp]);
            float a = 0.0f;
            for (int m0 = 0; m0 < M; m0 += TILE_M) {
                const int mhi = (m0 + TILE_M < M) ? (m0 + TILE_M) : M;
                for (int m = m0; m < mhi; ++m) {
                    const uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos]);
                    a += lut_in_smem ? LUT_AT(s_lut, m * 256 + cm)
                                     : LUT_LOAD(g_lut[m * 256 + cm]);
                }
            }
            dist = a;
        }
        // Advance the shared probe cursor to where this chunk ended.
        if (tid == BLOCK - 1 || lt == total - 1) {
            int pp = p;
            const int last = (total - 1 < chunk + BLOCK - 1) ? total - 1 : chunk + BLOCK - 1;
            while (pp + 1 < nprobe && last >= my_offs[pp + 1]) ++pp;
            p = pp;
        }
        __syncthreads();

        if (lt < total && dist < *s_thresh) {
            const int slot = atomicAdd(s_cnt, 1);
            s_val[slot] = dist;      // slot < cap by the sizing argument above
            s_pos[slot] = pos;
        }
        __syncthreads();

        if (*s_cnt > ck)
            jhq_compact_topck(s_val, s_pos, s_cnt, s_thresh, ck, cap, tid, BLOCK);
    }

    // One final ordering so the output is sorted by primary distance.
    jhq_compact_topck(s_val, s_pos, s_cnt, s_thresh, ck, cap, tid, BLOCK);

    const int  n   = *s_cnt;
    float* out_pri = topck_primary + (long long)bqi * ck;
    int*   out_pos = topck_pos     + (long long)bqi * ck;
    for (int c = tid; c < ck; c += BLOCK) {
        const bool have = (c < n);
        out_pri[c] = have ? s_val[c] : INF;
        out_pos[c] = have ? s_pos[c] : -1;
    }
}

__global__ void scan_ivf_coalesced_kernel(
    const lut_t*   __restrict__ d_byte_lut,     // [B, M, 256]
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary_t, // [M, N] transposed
    const int*     __restrict__ query_total,
    float*                      topck_primary,
    int*                        topck_pos,
    int nprobe, int M, int N, int ck, bool lut_in_smem,
    int pfx_num, int pfx_den, int tile_m)
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
    lut_t* s_lut = (lut_t*)(s_red_idx + BLOCK);

    const lut_t* g_lut = d_byte_lut + (long long)bqi * M * 256;
    if (lut_in_smem) {
        for (int i = tid; i < M * 256; i += BLOCK)
            s_lut[i] = g_lut[i];
        __syncthreads();
    }
    // LUT_AT converts when lut_t is half and passes the value through when
    // JHQ_LUT32 makes it float.
    const lut_t* my_lut_s = s_lut;

    int total      = query_total[bqi];
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);

    constexpr int TILE_C = JHQ_TILE_C;
    const     int TILE_M = tile_m;
    constexpr int KEEP   = JHQ_PREFIX_KEEP;

    // Prefix length. Do NOT round it to a whole tile: at TILE_M = 96 that sent
    // PM = 24 to 0 and then clamped it back up to M, silently turning the
    // cascade off. The tile loop already clamps its upper bound to PM, so a
    // partial last tile is handled. The fraction is read from the environment
    // so one binary sweeps it; only KEEP has to be a build-time constant,
    // because it sizes an array.
    int PM = (M * pfx_num) / pfx_den;
    if (PM < 1) PM = 1;
    if (PM > M) PM = M;

    // Best KEEP candidates by the prefix, with the partial sum kept so pass 2
    // resumes instead of rescanning.
    float pd[KEEP]; int pp[KEEP];
    #pragma unroll
    for (int i = 0; i < KEEP; i++) { pd[i] = INF; pp[i] = -1; }

    // ── pass 1: subspaces [0, PM) over every candidate ───────────────────────
    // Candidate assignment. Contiguous (lt = base + t) hands one thread a run of
    // adjacent candidates, and adjacency means the same IVF list region: the v20
    // grid lost 0.7328 -> 0.6933 recall going from TILE_C 1 to 16 because true
    // neighbours cluster and then compete for one thread's K_LOCAL slots.
    // Striding by BLOCK gives each thread the same spread-out sample TILE_C=1
    // sees, while still running TILE_C independent accumulate chains.
#if JHQ_TILE_STRIDED
    for (int base = tid; base < total; base += BLOCK * TILE_C) {
#else
    for (int base = tid * TILE_C; base < total; base += BLOCK * TILE_C) {
#endif
        float acc[TILE_C];
        int   pos[TILE_C];
        int   live = 0;
        #pragma unroll
        for (int t = 0; t < TILE_C; ++t) {
            acc[t] = 0.0f;
#if JHQ_TILE_STRIDED
            const int lt = base + t * BLOCK;
#else
            const int lt = base + t;
#endif
            if (lt < total) {
                int p = 0;
                while (p + 1 < nprobe && lt >= my_offs[p + 1]) ++p;
                pos[t] = list_offsets[my_ids[p]] + (lt - my_offs[p]);
                live = t + 1;
            } else {
                pos[t] = -1;
            }
        }

        for (int m0 = 0; m0 < PM; m0 += TILE_M) {
            const int mhi = (m0 + TILE_M < PM) ? (m0 + TILE_M) : PM;
            #pragma unroll
            for (int t = 0; t < TILE_C; ++t) {
                if (t >= live) break;
                float a = acc[t];
                for (int m = m0; m < mhi; ++m) {
                    uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos[t]]);
                    a += lut_in_smem ? LUT_AT(my_lut_s, m * 256 + cm)
                                     : LUT_LOAD(g_lut[m * 256 + cm]);
                }
                acc[t] = a;
            }
        }

        #pragma unroll
        for (int t = 0; t < TILE_C; ++t) {
            if (t >= live) break;
            if (acc[t] < pd[KEEP - 1]) {
                pd[KEEP - 1] = acc[t];
                pp[KEEP - 1] = pos[t];
                #pragma unroll
                for (int i = KEEP - 1; i > 0 && pd[i] < pd[i-1]; --i) {
                    float td = pd[i-1]; pd[i-1] = pd[i]; pd[i] = td;
                    int   tp = pp[i-1]; pp[i-1] = pp[i]; pp[i] = tp;
                }
            }
        }
    }

    // ── pass 2: finish [PM, M) for the survivors only ────────────────────────
    float ld[K_LOCAL]; int lp[K_LOCAL];
    #pragma unroll
    for (int i = 0; i < K_LOCAL; i++) { ld[i] = INF; lp[i] = -1; }

    for (int s0 = 0; s0 < KEEP; s0 += TILE_C) {
        float acc[TILE_C];
        int   pos[TILE_C];
        int   live = 0;
        #pragma unroll
        for (int t = 0; t < TILE_C; ++t) {
            const int si = s0 + t;
            if (si < KEEP && pp[si] >= 0) { acc[t] = pd[si]; pos[t] = pp[si]; live = t + 1; }
            else                          { acc[t] = INF;    pos[t] = -1; }
        }
        if (live == 0) break;   // pd/pp are sorted, so a gap ends the list

        for (int m0 = PM; m0 < M; m0 += TILE_M) {
            const int mhi = (m0 + TILE_M < M) ? (m0 + TILE_M) : M;
            #pragma unroll
            for (int t = 0; t < TILE_C; ++t) {
                if (t >= live) break;
                float a = acc[t];
                for (int m = m0; m < mhi; ++m) {
                    uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos[t]]);
                    a += lut_in_smem ? LUT_AT(my_lut_s, m * 256 + cm)
                                     : LUT_LOAD(g_lut[m * 256 + cm]);
                }
                acc[t] = a;
            }
        }

        #pragma unroll
        for (int t = 0; t < TILE_C; ++t) {
            if (t >= live) break;
            if (acc[t] < ld[K_LOCAL - 1]) {
                ld[K_LOCAL - 1] = acc[t];
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
#if JHQ_BITONIC_SELECT
    // ck is ceil(alpha*k) -- 1000 at the alpha=100 operating point -- and the
    // loop below extracts one candidate per pass with a whole-block reduction:
    // log2(BLOCK)+2 barriers each, so ~12000 barriers per block at BLOCK=1024,
    // with 1023 of 1024 threads idle for every write. None of that scales with
    // nprobe, which is why the scan stays expensive when there is almost
    // nothing to scan: at nprobe=8 each thread holds only 7 candidates.
    //
    // Sorting the whole array once instead costs log2(n)*(log2(n)+1)/2 passes
    // -- 78 at n_cands = 4096 -- one barrier each, with every thread working.
    // n_cands is K_LOCAL*BLOCK, a power of two whenever both are.
    for (int k2 = 2; k2 <= n_cands; k2 <<= 1) {
        for (int j = k2 >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n_cands; i += BLOCK) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool up = ((i & k2) == 0);
                    if ((s_cdist[i] > s_cdist[ixj]) == up) {
                        float td = s_cdist[i]; s_cdist[i] = s_cdist[ixj]; s_cdist[ixj] = td;
                        int   tp = s_cpos [i]; s_cpos [i] = s_cpos [ixj]; s_cpos [ixj] = tp;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int c = tid; c < ck; c += BLOCK) {
        const bool have = (c < n_cands);
        out_primary[c] = have ? s_cdist[c] : INF;
        out_pos    [c] = have ? s_cpos [c] : -1;
    }
#else
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
#endif
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
#if JHQ_NO_RESIDUAL
        // JQ-GPU: the same primary quantiser, IVF and GPU scan, with the
        // residual level switched off. This is the hierarchy ablation -- it
        // separates what the second level of the quantiser contributes from
        // what the kernel work contributes, which a comparison against the
        // published CPU numbers cannot do.
        (void)d_res; (void)list_corr;
        comp_dists[i] = topck_primary[i];
#else
        comp_dists[i] = topck_primary[i] + d_res + list_corr[pos];
#endif
    }
}

// The residual level without materialising its lookup tables.
//
// Equation 8 sums one lookup per dimension, T_R^{mj}[q'_mj], and the paper
// builds M*Ds = d such tables per query in O(d*Kr). Cheap for one query on a
// CPU; batched here it is B*d*Kr floats -- 3.0 GB at B=1024, d=3072, Kr=256 --
// written in full and read back, for values that cost two instructions.
//
// Lemma 5 makes the recompute cheap: residual dimensions are independent, so
// one scalar codebook C_R^m is shared across every dimension of subspace m and
// the codebooks are only O(M*Kr). An entry is T_R^{mj}[k] = (q^{mj} - c^m_Rk)^2,
// so with the rotated query and the codebooks in shared memory it is a subtract
// and a multiply. One block per query: the query's d floats are read once for
// all of its alpha*k candidates.
//
// __fmul_rn keeps the square from contracting into the accumulate, so each term
// rounds where it rounded going through the table. That is what makes the
// A/B against JHQ_RESID_LUT=1 a like-for-like comparison.
__global__ void residual_refine_fused_kernel(
    const int*     __restrict__ topck_pos,
    const float*   __restrict__ topck_primary,
    const float*   __restrict__ q_rot,
    const float*   __restrict__ res_c1d,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int ck, int d, int Ds, int Kr, int M, int Br, int bpv, int B,
    int cb_in_smem)
{
    const int bqi = blockIdx.x;
    if (bqi >= B) return;
    extern __shared__ char shm[];
    float* s_q  = (float*)shm;
    float* s_cb = s_q + d;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        s_q[i] = q_rot[(long long)bqi * d + i];
    if (cb_in_smem)
        for (int i = threadIdx.x; i < M * Kr; i += blockDim.x)
            s_cb[i] = res_c1d[i];
    __syncthreads();
    const float* cb = cb_in_smem ? s_cb : res_c1d;
    const long long base = (long long)bqi * ck;
    for (int c = threadIdx.x; c < ck; c += blockDim.x) {
        const int pos = topck_pos[base + c];
        if (pos < 0) { comp_dists[base + c] = __int_as_float(0x7F800000); continue; }
        const uint8_t* rc = list_res + (long long)pos * bpv;
        float d_res = 0.0f;
        // Walk subspaces, not dimensions: the codebook row is fixed across the
        // Ds dimensions of a subspace, so the j/Ds that picks it belongs
        // outside the inner loop rather than costing a division per dimension.
        int j = 0;
        for (int m = 0; m < M; ++m) {
            const float* cbm = cb + (long long)m * Kr;
            for (int jj = 0; jj < Ds; ++jj, ++j) {
                const int ri = (Br == 4)
                    ? ((j & 1) == 0 ? (rc[j >> 1] & 0x0F) : (rc[j >> 1] >> 4))
                    : rc[j];
                const float diff = s_q[j] - cbm[ri];
                d_res += __fmul_rn(diff, diff);
            }
        }
#if JHQ_NO_RESIDUAL
        (void)d_res; (void)list_corr;
        comp_dists[base + c] = topck_primary[base + c];
#else
        comp_dists[base + c] = topck_primary[base + c] + d_res + list_corr[pos];
#endif
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
    int*   red_id  = (int*)(red_idx + BLOCK);

    const float* my_dists = comp_dists + (long long)bqi * ck;
    const int*   my_pos   = topck_pos  + (long long)bqi * ck;
    for (int i = tid; i < ck; i += BLOCK) { s_dists[i] = my_dists[i]; s_pos[i] = my_pos[i]; }
    __syncthreads();

    float* out_dists = final_dists + (long long)bqi * k;
    int*   out_ids   = final_ids   + (long long)bqi * k;

    // Ties are broken by database id, in both the per-thread scan and the
    // block reduction. Without it the winner of an exact tie is whichever
    // candidate the reduction happened to reach first, which varies with block
    // size and with the order candidates were scanned: on vogue at nprobe=32
    // the tenth and eleventh composite distances are bit-identical
    // (1.7044189) and the two residual paths returned different ids for the
    // same tie. The reference breaks ties the same way, so the two agree.
    for (int r = 0; r < k; ++r) {
        float bv = INF; int bi = -1; int bid = 0x7FFFFFFF;
        for (int i = tid; i < ck; i += BLOCK) {
            const float dv = s_dists[i];
            if (dv > bv) continue;
            const int p = s_pos[i];
            const int id = (p >= 0) ? list_ids[p] : 0x7FFFFFFF;
            if (dv < bv || id < bid) { bv = dv; bi = i; bid = id; }
        }
        red_val[tid] = bv; red_idx[tid] = bi; red_id[tid] = bid; __syncthreads();
        for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const bool take = (red_val[tid+stride] <  red_val[tid]) ||
                                  (red_val[tid+stride] == red_val[tid] &&
                                   red_id [tid+stride] <  red_id [tid]);
                if (take) {
                    red_val[tid] = red_val[tid+stride];
                    red_idx[tid] = red_idx[tid+stride];
                    red_id [tid] = red_id [tid+stride];
                }
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
// Materialise the residual lookup tables, as the paper's O(d*Kr) per-query
// construction does, instead of recomputing entries in the refine kernel.
// Off by default; the switch exists so the two can be diffed.
// JHQ_PAPER_FAITHFUL locks the semantics the fidelity tests verified. The
// point is not to pick good defaults -- those already existed -- but to make
// the properties unreachable from the environment, so a stray variable in a
// shell cannot turn the binary the paper cites into a different algorithm.
//
// Forbidden rather than ignored: silently discarding a variable someone set on
// purpose hides a misunderstanding, and the cost of stopping is one clear
// message against a table of numbers that are not what they claim to be.
#if JHQ_PAPER_FAITHFUL
static void jhq_faithful_guard() {
    static const char* const forbidden[] = {
        "JHQ_PFX_NUM",      // prefix cascade: prunes before the full primary
        "JHQ_PFX_DEN",      //   distance exists (Algorithm 1 line 3)
        "JHQ_EXACT_TOPCK",  // would restore the lossy retention buffer
        "JHQ_TF32",         // 10-bit mantissa can reorder candidates
        "JHQ_LUT32",        // half table rounds Eq. 6 terms before summing
        "JHQ_ASSIGN_OUT32", // coarse assignment precision
    };
    for (const char* v : forbidden)
        if (std::getenv(v))
            throw std::runtime_error(
                std::string("the paper-faithful target does not read ") + v +
                ". It is built with the verified semantics compiled in: full "
                "primary distance over all M subspaces, exact global "
                "top-alpha*k, no cascade, fp32 primary table, TF32 off. Use "
                "demo_jhq_v23_cascade or another named variant to vary them, "
                "and do not report the result as JHQ-GPU.");
    if (const char* pc = std::getenv("JHQ_PAPER_CODEBOOK"))
        if (pc[0] == '0')
            throw std::runtime_error(
                "JHQ_PAPER_CODEBOOK=0 selects the Lloyd-refined product "
                "quantiser, not the paper's Cartesian construction of "
                "equation 4. The faithful target builds equation 4.");
}
#endif

// Exact global top-alpha*k instead of the per-thread retention buffer. On by
// default: Algorithm 1 line 4 is a global selection, and the retention buffer
// is only equal to it when no thread happens to hold more than K_LOCAL of the
// answer. JHQ_EXACT_TOPCK=0 selects the old path for measurement.
static bool exact_topck() {
#if JHQ_PAPER_FAITHFUL
    return true;                       // compiled in, not defaulted
#else
    const char* e = std::getenv("JHQ_EXACT_TOPCK");
    return !(e && e[0] == '0');
#endif
}

static bool resid_lut_materialised() {
    const char* e = std::getenv("JHQ_RESID_LUT");
    return e && e[0] == '1';
}

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
    // The scan block carries the 48 KB table plus (2*K_LOCAL+2)*BLOCK floats of
    // scratch, so at 256 threads one block holds ~58 KB and the SM fits exactly
    // one of them: 256 of 1536 resident threads, 17% occupancy. 1024 threads
    // costs 89 KB, still one block, but four times the threads. Runtime rather
    // than build-time so one binary sweeps it.
    // Cascade prefix as a fraction of M; 1/1 disables it and restores v19.
#if JHQ_PAPER_FAITHFUL
    // PM = M. The exact scan takes no prefix arguments at all, so these only
    // reach the legacy kernel, which this target never launches; they are
    // pinned anyway so the values cannot be misread from a log.
    static const int pfx_num = 1, pfx_den = 1;
#else
    static const int pfx_num = [] { const char* e = getenv("JHQ_PFX_NUM"); int v = e ? atoi(e) : JHQ_PREFIX_NUM; return v > 0 ? v : 1; }();
    static const int pfx_den = [] { const char* e = getenv("JHQ_PFX_DEN"); int v = e ? atoi(e) : JHQ_PREFIX_DEN; return v > 0 ? v : 4; }();
#endif

    // The v20 grid put TILE_C=1, TILE_M=96 ahead of every tiled shape at equal
    // recall (10.089 ms vs 11.491 at TILE_C=4/TILE_M=8), and TILE_C >= 8 lost
    // recall outright -- 0.7192 at 8, 0.6933 at 16 against 0.7328 -- because a
    // larger tile hands one thread a run of spatially adjacent candidates that
    // then compete for its K_LOCAL slots. So TILE_M is swept at runtime and
    // TILE_C defaults to 1.
    static const int tile_m = [] { const char* e = getenv("JHQ_TILE_M_RT"); int v = e ? atoi(e) : 96; return v > 0 ? v : 96; }();

    static const int BLOCK = [] {
        const char* e = getenv("JHQ_BLOCK");
        const int   v = e ? atoi(e) : 256;
        return (v == 128 || v == 256 || v == 512 || v == 1024) ? v : 256;
    }();
    // Must track JHQ_K_LOCAL exactly -- the kernel indexes s_cdist/s_cpos as
    // K_LOCAL*BLOCK each, so a stale literal here silently corrupts memory
    // rather than failing to build.
    // Candidate scratch, then the staged lookup table if it fits. Blackwell
    // allows well past the 48 KB default per block, but only after opting in
    // per kernel, so ask for the limit first and fall back to the global-memory
    // path when even that is not enough.
    const int   scan_base = (2 * JHQ_K_LOCAL * BLOCK + 2 * BLOCK) * (int)sizeof(float);
    const int   lut_bytes = M * 256 * (int)sizeof(lut_t);
    int         smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin,
                   cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const bool  lut_in_smem = (scan_base + lut_bytes) <= smem_optin;
    const int   scan_smem = lut_in_smem ? (scan_base + lut_bytes) : scan_base;
    if (lut_in_smem)
        CUDA_CHECK(cudaFuncSetAttribute(scan_ivf_coalesced_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, scan_smem));
    // s_dists[ck] + s_pos[ck] + red_val[BLOCK] + red_idx[BLOCK] + red_id[BLOCK]
    const int   topk_smem = (2 * ck + 3 * BLOCK) * (int)sizeof(float);

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
    // Opt in before launching: this block asks for (nlist + 2*BLOCK) floats,
    // which is 40 KB at nlist=8192 but 72 KB at 16384 -- past the 48 KB a
    // kernel gets without asking. Without this the launch fails, and because
    // the sequence is being captured into a graph the failure surfaces later
    // as "operation failed due to a previous error during capture", pointing
    // at whichever check runs next rather than at this line. It only shows up
    // on datasets large enough to want that many lists: 10M vectors at
    // nlist=8192 were fine, 17.8M at 16384 were not.
    {
        const int probe_smem = (nlist + 2 * BLOCK) * (int)sizeof(float);
        if (probe_smem > 48 * 1024) {
            int optin = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(
                &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
            if (probe_smem > optin)
                throw std::runtime_error(
                    "select_probes_kernel needs " + std::to_string(probe_smem) +
                    " B of shared memory for nlist=" + std::to_string(nlist) +
                    " but the device allows " + std::to_string(optin) +
                    "; lower nlist or raise the block size");
            CUDA_CHECK(cudaFuncSetAttribute(
                select_probes_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, probe_smem));
        }
    }
    select_probes_kernel<<<B, BLOCK, (nlist + 2*BLOCK) * (int)sizeof(float), ws.stream>>>(
        ws.d_dots, d_cent_norms, d_list_offsets,
        ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total, nlist, nprobe);
    CUDA_CHECK(cudaGetLastError());
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
    if (exact_topck()) {
        // cap = ck + BLOCK rounded up to a power of two, which is what makes
        // overflow impossible and what the bitonic sort needs.
        int cap = 1; while (cap < ck + BLOCK) cap <<= 1;
        const int ex_base = cap * (int)(sizeof(float) + sizeof(int))
                          + (int)sizeof(int) + (int)sizeof(float);
        const bool ex_lut = (ex_base + lut_bytes) <= smem_optin;
        const int  ex_smem = ex_lut ? (ex_base + lut_bytes) : ex_base;
        if (ex_smem > smem_optin)
            throw std::runtime_error(
                "scan_ivf_exact_kernel needs " + std::to_string(ex_smem) +
                " B of shared memory for ck=" + std::to_string(ck) +
                " at BLOCK=" + std::to_string(BLOCK) +
                "; the device allows " + std::to_string(smem_optin) +
                ". Lower alpha or BLOCK.");
        CUDA_CHECK(cudaFuncSetAttribute(scan_ivf_exact_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, ex_smem));
        scan_ivf_exact_kernel<<<B, BLOCK, ex_smem, ws.stream>>>(
            ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
            d_list_primary_t, ws.d_query_total,
            ws.d_topck_primary, ws.d_topck_pos,
            nprobe, M, ntotal, ck, cap, ex_lut, tile_m);
        CUDA_CHECK(cudaGetLastError());

    } else
    scan_ivf_coalesced_kernel<<<B, BLOCK, scan_smem, ws.stream>>>(
        ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
        d_list_primary_t, ws.d_query_total,
        ws.d_topck_primary, ws.d_topck_pos,
        nprobe, M, ntotal, ck, lut_in_smem, pfx_num, pfx_den, tile_m);
    // A launch that fails for resources is otherwise silent here: the stream
    // carries on, every distance stays at its initial value and the run reports
    // recall 0.0000 next to a QPS in the hundreds of thousands, which reads
    // like a data point. KEEP=16 at BLOCK=1024 does exactly that -- pd[KEEP]
    // and pp[KEEP] alone are 32 registers against the 64 per thread available
    // when 1024 of them are resident.
    CUDA_CHECK(cudaGetLastError());
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[5], ws.stream));
#endif
    // 6-7. Residual level. The fused kernel recomputes each table entry from
    // the query and the shared codebooks instead of reading it out of a
    // B*d*Kr buffer; JHQ_RESID_LUT=1 restores the materialised path.
    // The buffer is allocated in add(), which reads JHQ_RESID_LUT then, while
    // this reads it now. They agree within one process, but a null here would
    // otherwise be a device-side fault several kernels later rather than a
    // message, so say so where it can still be understood.
    if (resid_lut_materialised() && !ws.d_lut_r)
        throw std::runtime_error(
            "JHQ_RESID_LUT asks for the materialised residual tables but the "
            "buffer was not allocated: it is reserved in add(), so the variable "
            "has to be set before the index is built, not just before the search.");
    if (!resid_lut_materialised()) {
#if JHQ_STEP_TIMING
        CUDA_CHECK(cudaEventRecord(ws.ev_step[6], ws.stream));
#endif
        const size_t q_bytes  = (size_t)d * sizeof(float);
        const size_t cb_bytes = (size_t)M * Kr * sizeof(float);
        const int    cb_smem  = (q_bytes + cb_bytes <= 40960) ? 1 : 0;
        residual_refine_fused_kernel<<<B, 256,
            q_bytes + (cb_smem ? cb_bytes : 0), ws.stream>>>(
            ws.d_topck_pos, ws.d_topck_primary, ws.d_q_rot, d_res_c1d,
            d_list_res, d_list_corr, ws.d_comp_dists,
            ck, d, Ds, Kr, M, Br, bpv, B, cb_smem);
        CUDA_CHECK(cudaGetLastError());
    } else {
        {
            long long tot  = (long long)B * d * Kr;
            int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            build_residual_lut_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_q_rot, d_res_c1d, ws.d_lut_r, B, d, Ds, Kr);
        }
#if JHQ_STEP_TIMING
        CUDA_CHECK(cudaEventRecord(ws.ev_step[6], ws.stream));
#endif
        {
            long long tot  = (long long)B * ck;
            int       grid = (int)std::min((tot + BLOCK - 1) / BLOCK, (long long)65535);
            residual_refine_batched_kernel<<<grid, BLOCK, 0, ws.stream>>>(
                ws.d_topck_pos, ws.d_topck_primary,
                ws.d_lut_r, d_list_res, d_list_corr,
                ws.d_comp_dists, ck, d, Kr, Br, bpv, B);
        }
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
#if JHQ_PAPER_FAITHFUL
    jhq_faithful_guard();
#endif
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

    // JHQ_DUMP_TOPCK=<path>: after the work has run and the stream is idle --
    // a graph capture permits neither allocation nor synchronisation, which is
    // why this cannot sit beside the kernel it inspects. Writes every
    // candidate's complete primary distance for query 0 and the ck the
    // selector returned, for the differential test to compare.
    if (const char* dp = std::getenv("JHQ_DUMP_TOPCK")) {
        int h_total = 0;
        CUDA_CHECK(cudaMemcpy(&h_total, ws.d_query_total, sizeof(int),
                              cudaMemcpyDeviceToHost));
        float* d_ad = nullptr; int* d_ap = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ad, (size_t)h_total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ap, (size_t)h_total * sizeof(int)));
        dump_primary_distances_kernel<<<256, 256>>>(
            ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
            d_list_primary_t, ws.d_query_total, d_ad, d_ap, h_total,
            0, nprobe, M, ntotal);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> ad(h_total), sp(ck);
        std::vector<int>   ap(h_total), sq(ck);
        CUDA_CHECK(cudaMemcpy(ad.data(), d_ad, ad.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ap.data(), d_ap, ap.size()*sizeof(int),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sp.data(), ws.d_topck_primary, sp.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sq.data(), ws.d_topck_pos,     sq.size()*sizeof(int),   cudaMemcpyDeviceToHost));
        if (FILE* f = std::fopen(dp, "wb")) {
            std::fwrite(&h_total, sizeof(int), 1, f);
            std::fwrite(&ck,      sizeof(int), 1, f);
            std::fwrite(ad.data(), sizeof(float), ad.size(), f);
            std::fwrite(ap.data(), sizeof(int),   ap.size(), f);
            std::fwrite(sp.data(), sizeof(float), sp.size(), f);
            std::fwrite(sq.data(), sizeof(int),   sq.size(), f);
            // Stages F and G: the composite distances the refinement produced
            // for those ck, and the k ids the final selection returned. The
            // test checks the second is the exact top-k of the first.
            std::vector<float> cd(ck), fd(k);
            std::vector<int>   fi(k);
            CUDA_CHECK(cudaMemcpy(cd.data(), ws.d_comp_dists,  cd.size()*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(fd.data(), ws.d_final_dists, fd.size()*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(fi.data(), ws.d_final_ids,   fi.size()*sizeof(int),   cudaMemcpyDeviceToHost));
            std::fwrite(&k, sizeof(int), 1, f);
            std::fwrite(cd.data(), sizeof(float), cd.size(), f);
            std::fwrite(fd.data(), sizeof(float), fd.size(), f);
            std::fwrite(fi.data(), sizeof(int),   fi.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "[dump] %d candidates, ck=%d -> %s\n", h_total, ck, dp);
        }
        cudaFree(d_ad); cudaFree(d_ap);
    }
#if JHQ_STEP_TIMING
    report_step_timing(ws, B_full, nprobe, ck);
#endif
}

} // namespace jhq_gpu
