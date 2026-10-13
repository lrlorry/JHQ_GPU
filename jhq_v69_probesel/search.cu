#include "jhq_v69_probesel/search.cuh"
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
// Element type of the primary lookup table: float, and no longer a choice.
//
// Equation 6 sums the per-subspace distances exactly, and storing each as a
// half rounded it to eleven mantissa bits before it was ever added -- an error
// term the paper's bound does not carry. That was tolerated because 256
// entries a subspace only fit in shared memory as halves. Factorised, a
// subspace is 32 entries, which fits in float at every M here, so the reason
// for the half is gone and so is the switch. search.cuh has the identity.
typedef jhq_gpu::jhq_lut_t lut_t;

// The factorised table is float, so store and load are identity. Kept as
// names because the scan reads them in three places and a bare index there
// hides which array is meant.
#define LUT_STORE(x) (x)
#define LUT_LOAD(x)  (x)

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

SearchDiag& search_diag() { static SearchDiag g; return g; }

static bool diag_on() {
    static const int v = [] {
        const char* e = std::getenv("JHQ_DIAG");
        return e ? std::atoi(e) : 0;
    }();
    return v != 0;
}

// ── select_probes_kernel — exact top-nprobe by radix select (v69) ────────────
//
// v65's kernel streamed the centroid distances into a buffer of cap = pow2 >=
// nprobe + BLOCK and compacted it with a full bitonic sort whenever it filled:
// 1024 or 2048 elements, 55-66 barrier-separated stages, three or four times
// per query, one block per query. At b = 8 that is eight blocks on the whole
// card doing serial sorting work: 53-78 us of a 420-480 us search
// (experiments/math_ablation/b0_results/stages).
//
// v69 keeps the one-block-per-query shape and returns exactly the same probes
// in exactly the same order -- the nprobe smallest by (distance, list id), in
// that order, the smaller id winning a tie -- but finds them by selection:
//   1. radix select on an order-preserving 32-bit key of the distance, four
//      8-bit passes, gives the key K* of the k-th smallest (k = min(nprobe,
//      nlist)) and krem, how many keys equal to K* belong to the selection.
//      The histogram has one copy per lane, hist[bin][lane]: the first byte
//      of a key is the sign and the top of the exponent, which most distances
//      share, so the first pass sends most centroids to a few bins, and with
//      one shared copy those atomics would serialise up to 32 ways. With a
//      copy per lane no two lanes of a warp touch the same word or the same
//      bank. 32 KB, independent of nlist and BLOCK.
//   2. one pass collects every key < K*; keys == K* are taken in increasing
//      list id until krem are taken.
//   3. each of the k is ranked by counting how many of the others precede it
//      on (distance, id): two barriers instead of a sorting network.
//   4. the list lengths are summed by a block scan instead of one thread.
// Every pass computes the distance with the one function below, so all of them
// see the same bits; 2*row[c] is exact, so they are also v65's bits.
__device__ __forceinline__ bool probe_before(float va, int ia, float vb, int ib) {
    return (va < vb) || (va == vb && ia < ib);
}

__device__ __forceinline__ float probe_dist(const float* __restrict__ cent_norms,
                                            const float* __restrict__ row, int c) {
    return __fmaf_rn(-2.0f, row[c], cent_norms[c]);
}

// key(a) < key(b) iff a < b, and key(a) == key(b) iff a == b: the two zeros,
// which compare equal, share a key.
__device__ __forceinline__ unsigned int probe_key(float v) {
    unsigned int u = __float_as_uint(v);
    if ((u << 1) == 0u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

constexpr int PROBE_BINS = 256;

// dynamic shared memory of select_probes_kernel
inline int probe_select_smem(int nprobe) {
    return (PROBE_BINS * 32 + PROBE_BINS) * (int)sizeof(unsigned int)
         + 3 * nprobe * (int)sizeof(int) + 40 * (int)sizeof(int);
}

__global__ void select_probes_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ cent_norms,
    const int*   __restrict__ list_offsets,
    int*                      probe_ids,
    int*                      probe_offsets,
    int*                      query_total,
    int nlist, int nprobe)
{
    extern __shared__ char shm[];
    unsigned int* s_hist = (unsigned int*)shm;                 // [PROBE_BINS][32]
    unsigned int* s_bin  = s_hist + PROBE_BINS * 32;           // [PROBE_BINS]
    float*        s_val  = (float*)(s_bin + PROBE_BINS);       // [nprobe] selected, unordered
    int*          s_pos  = (int*)(s_val + nprobe);             // [nprobe]
    int*          s_ord  = s_pos + nprobe;                     // [nprobe] selected ids, ordered
    int*          s_misc = s_ord + nprobe;                     // [8]
    int*          s_wsum = s_misc + 8;                         // [32]

    const int    BLOCK = blockDim.x;                           // a multiple of 32
    const int    bqi   = blockIdx.x;
    const int    tid   = threadIdx.x;
    const int    lane  = tid & 31, wid = tid >> 5, nw = BLOCK >> 5;
    const float* row   = dots + (long long)bqi * nlist;
    const int    k     = (nprobe < nlist) ? nprobe : nlist;

    if (tid == 0) { s_misc[0] = 0; s_misc[1] = 0; }

    // 1. radix select: prefix ends as K*, krem as the number of keys equal to K* that are selected
    unsigned int prefix = 0u, mask = 0u;
    int krem = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < PROBE_BINS * 32; i += BLOCK) s_hist[i] = 0u;
        __syncthreads();
        #pragma unroll 4
        for (int c = tid; c < nlist; c += BLOCK) {
            const unsigned int key = probe_key(probe_dist(cent_norms, row, c));
            if ((key & mask) == prefix) atomicAdd(&s_hist[((key >> shift) & 255u) * 32 + lane], 1u);
        }
        __syncthreads();
        for (int bin = wid; bin < PROBE_BINS; bin += nw) {
            const unsigned int x = __reduce_add_sync(0xffffffffu, s_hist[bin * 32 + lane]);
            if (lane == 0) s_bin[bin] = x;
        }
        __syncthreads();
        if (wid == 0) {
            // the bin holding the krem-th smallest of the keys that still match: lane l scans bins 8l..8l+7
            unsigned int loc[8], sum = 0u;
            #pragma unroll
            for (int j = 0; j < 8; ++j) { loc[j] = s_bin[lane * 8 + j]; sum += loc[j]; }
            unsigned int inc = sum;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned int t = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) inc += t;
            }
            const unsigned int exc = inc - sum;
            if (exc < (unsigned int)krem && (unsigned int)krem <= inc) {
                unsigned int acc = exc;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (acc + loc[j] >= (unsigned int)krem) { s_misc[2] = lane * 8 + j; s_misc[3] = krem - (int)acc; break; }
                    acc += loc[j];
                }
            }
        }
        __syncthreads();
        prefix |= (unsigned int)s_misc[2] << shift;
        mask   |= 255u << shift;
        krem    = s_misc[3];
    }

    // 2. collect every key below K*, then krem of the keys equal to it
    const int nlt = k - krem;
    for (int c = tid; c < nlist; c += BLOCK) {
        const float        v   = probe_dist(cent_norms, row, c);
        const unsigned int key = probe_key(v);
        if (key < prefix) {
            const int slot = atomicAdd(&s_misc[0], 1);
            s_val[slot] = v; s_pos[slot] = c;
        } else if (key == prefix) {
            const int slot = atomicAdd(&s_misc[1], 1);
            if (slot < krem) { s_val[nlt + slot] = v; s_pos[nlt + slot] = c; }
        }
    }
    __syncthreads();
    if (s_misc[1] > krem) {
        // More keys equal K* than there are slots: the ones above were taken in arrival
        // order, so take them again in increasing list id. Block-uniform branch.
        int taken = 0;
        for (int c0 = 0; c0 < nlist && taken < krem; c0 += BLOCK) {
            const int  c = c0 + tid;
            float      v = 0.0f;
            bool       f = false;
            if (c < nlist) { v = probe_dist(cent_norms, row, c); f = (probe_key(v) == prefix); }
            const unsigned int bal = __ballot_sync(0xffffffffu, f);
            if (lane == 0) s_wsum[wid] = __popc(bal);
            __syncthreads();
            int wpre = 0, tot = 0;
            for (int w = 0; w < nw; ++w) { const int x = s_wsum[w]; if (w < wid) wpre += x; tot += x; }
            const int r = taken + wpre + __popc(bal & ((1u << lane) - 1u));
            if (f && r < krem) { s_val[nlt + r] = v; s_pos[nlt + r] = c; }
            taken += tot;
            __syncthreads();
        }
    }

    // 3. rank each selected entry on (distance, id); G adjacent lanes share one entry's count
    {
        int G = 1;
        while (G < 32 && 2 * G * k <= BLOCK) G <<= 1;
        const int g = tid & (G - 1);
        for (int i0 = 0; i0 < k; i0 += BLOCK / G) {
            const int i = i0 + tid / G;
            int cnt = 0, pi = 0;
            if (i < k) {
                const float vi = s_val[i];
                pi = s_pos[i];
                for (int j = g; j < k; j += G) cnt += probe_before(s_val[j], s_pos[j], vi, pi) ? 1 : 0;
            }
            for (int o = G >> 1; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
            if (i < k && g == 0) s_ord[cnt] = pi;
        }
    }
    __syncthreads();

    // 4. probe ids and the exclusive prefix of their list lengths. Fewer lists than
    // nprobe only when nlist < nprobe: the repeated id then contributes zero length.
    int* my_ids  = probe_ids     + bqi * nprobe;
    int* my_offs = probe_offsets + bqi * (nprobe + 1);
    const int per = (nprobe + BLOCK - 1) / BLOCK;               // consecutive entries per thread
    const int p0  = tid * per;
    int tsum = 0;
    for (int p = p0; p < p0 + per && p < nprobe; ++p) {
        const int id = s_ord[p < k ? p : k - 1];
        my_ids[p] = id;
        if (p < k) tsum += list_offsets[id + 1] - list_offsets[id];
    }
    int inc = tsum;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += t;
    }
    if (lane == 31) s_wsum[wid] = inc;
    __syncthreads();
    int acc = inc - tsum, total = 0;                            // exclusive prefix of this thread's first entry
    for (int w = 0; w < nw; ++w) { const int x = s_wsum[w]; if (w < wid) acc += x; total += x; }
    if (tid == 0) my_offs[0] = 0;
    for (int p = p0; p < p0 + per && p < nprobe; ++p) {
        if (p < k) { const int id = s_ord[p]; acc += list_offsets[id + 1] - list_offsets[id]; }
        my_offs[p + 1] = acc;
    }
    if (tid == 0) query_total[bqi] = total;
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
    lut_t*                    d_byte_lut,  // [B][M][32]
    int B, int d, int M, int Ds, int K)
{
    // Entry h < 16 is the high nibble's half-distance, h >= 16 the low
    // nibble's. Both are read off the full centroid table rather than from
    // the level array: code h<<4 has exactly h as its first Ds/2 digits and
    // code h-16 has exactly h-16 as its last Ds/2, so the representative
    // centroid already carries the right per-dimension values and the
    // Cartesian property -- verified once at train time -- is what makes the
    // representative stand for all sixteen codes that share that half.
    const int H = Ds >> 1;
    const long long total = (long long)B * M * JHQ_SPLIT_LUT;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long long)gridDim.x * blockDim.x) {
        const int bqi   = (int)(i / ((long long)M * JHQ_SPLIT_LUT));
        const int local = (int)(i % ((long long)M * JHQ_SPLIT_LUT));
        const int m     = local / JHQ_SPLIT_LUT;
        const int h     = local % JHQ_SPLIT_LUT;

        const float* q_m = d_q_rot + (long long)bqi * d + (long long)m * Ds;
        const int    c   = (h < 16) ? (h << 4) : (h - 16);
        const float* cc  = d_cent + ((long long)m * K + c) * Ds;
        const int    j0  = (h < 16) ? 0 : H;
        const int    j1  = (h < 16) ? H : Ds;
        float sum = 0.0f;
        for (int j = j0; j < j1; ++j) { const float t = q_m[j] - cc[j]; sum += t * t; }
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

// v62: the order every list-group stage selects and merges by -- primary
// distance, then position in the inverted lists. Positions are unique within
// a query, so this is a strict total order: the ck smallest candidates under
// it are one definite set, whatever the grouping, the block size or the
// order candidates are visited in. (jhq_compact_topck orders by distance
// alone, so among candidates tied exactly at the ck-th distance it keeps
// whichever the scan happened to reach first.)
__device__ __forceinline__ bool key_less(float va, int pa, float vb, int pb) {
    return va < vb || (va == vb && pa < pb);
}

// v63: key_less as one 64-bit integer, for atomicMin. Primary distances are
// sums of squared differences, so >= +0 (or +INF), whose float bits order as
// the floats do; positions are >= 0. So a < b here exactly when key_less.
__device__ __forceinline__ unsigned long long key64(float v, int p) {
    return ((unsigned long long)__float_as_uint(v) << 32) | (unsigned)p;
}

// jhq_compact_topck under key_less: the same buffer and the same argument,
// with the threshold a (distance, position) pair. Padding sorts last.
__device__ __forceinline__ void jhq_compact_topck_lex(
    float* s_val, int* s_pos, int* s_cnt, float* s_thresh, int* s_tpos,
    int ck, int cap, int tid, int BLOCK)
{
    const int n = *s_cnt;
    const float INF = __int_as_float(0x7F800000);
    for (int i = n + tid; i < cap; i += BLOCK) { s_val[i] = INF; s_pos[i] = 0x7FFFFFFF; }
    __syncthreads();
    for (int k2 = 2; k2 <= cap; k2 <<= 1)
        for (int j = k2 >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < cap; i += BLOCK) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool up = ((i & k2) == 0);
                    if (key_less(s_val[ixj], s_pos[ixj], s_val[i], s_pos[i]) == up) {
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
        *s_tpos   = (n >= ck) ? s_pos[ck - 1] : 0x7FFFFFFF;
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
    const lut_t* g_lut = d_byte_lut + (long long)bqi * M * JHQ_SPLIT_LUT;
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
            a += LUT_LOAD(g_lut[m * JHQ_SPLIT_LUT + (cm >> 4)])
               + LUT_LOAD(g_lut[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
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
    int nprobe, int M, int N, int ck, int cap, bool lut_in_smem, int tile_m,
    // JHQ_FUSE_REFINE only. Null and zero on the unfused path.
    const float*   __restrict__ q_rot,
    const float*   __restrict__ res_c1d,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int d, int Ds, int Kr, int Br, int bpv)
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
    float* s_suf    = (float*)(s_lut + M * JHQ_SPLIT_LUT);   // JHQ_EXIT_BOUND only
    // The rotated query, staged once a block for the fused refinement. The
    // residual codebook stays in global: M*Kr*4 is 131 KB at M=128 and does
    // not fit beside the scan buffers, which is the same reason the standalone
    // refine kernel reads it from global at M >= 128.
    float* s_q      = s_suf + (JHQ_EXIT_BOUND ? (M + 1) : 0);

    const lut_t* g_lut = d_byte_lut + (long long)bqi * M * JHQ_SPLIT_LUT;
    if (lut_in_smem) {
        for (int i = tid; i < M * JHQ_SPLIT_LUT; i += BLOCK) s_lut[i] = g_lut[i];
    }
    if (tid == 0) { *s_cnt = 0; *s_thresh = INF; }
    __syncthreads();

#if JHQ_EXIT_BOUND
    // s_suf[m] = sum over subspaces at or after m of the smallest distance
    // that subspace can contribute for any code. Because the table is
    // factorised, that smallest entry is
    //     min_c T_m[c] = min_h Thi_m[h] + min_h Tlo_m[h]
    // so it costs 32 reads a subspace rather than 256. The factorisation pays
    // for the bound as well as for the lookup.
    {
        const lut_t* t = lut_in_smem ? s_lut : g_lut;
        for (int m = tid; m < M; m += BLOCK) {
            float hi = INF, lo = INF;
            for (int h = 0; h < 16; ++h) {
                const float a = LUT_LOAD(t[m * JHQ_SPLIT_LUT + h]);
                const float b = LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + h]);
                if (a < hi) hi = a;
                if (b < lo) lo = b;
            }
            s_suf[m] = hi + lo;
        }
        __syncthreads();
        // Serial suffix sum: M is a few hundred and this runs once a block,
        // against the millions of candidate lookups it guards.
        if (tid == 0) {
            float acc = 0.0f;
            s_suf[M] = 0.0f;
            for (int m = M - 1; m >= 0; --m) { acc += s_suf[m]; s_suf[m] = acc; }
        }
        __syncthreads();
    }
#endif

    const int  total   = query_total[bqi];
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);
    const int  TILE_M  = tile_m;

    // The probe a candidate belongs to only moves forward as lt grows, so the
    // search for it is carried across chunks instead of restarting at 0 for
    // every candidate.
    //
    // It was not, until now. `p` is a per-thread register and only the thread
    // holding the last candidate of a chunk wrote to it, so every other thread
    // kept p = 0 for the whole scan and re-walked the probe boundaries from
    // zero on every chunk. The comment above described the intent; the code
    // did the opposite, which is why nobody looked.
    //
    // Each thread owns lt = chunk + tid, which grows by BLOCK a chunk, so its
    // own probe index is monotone and a private cursor is exact. The work per
    // thread over the whole scan drops from about (nprobe/2) per chunk to
    // nprobe in total: 14,308 boundary tests to 128 on stella at nprobe=128,
    // which is roughly 30% of the inner loop's instruction count.
    int p = 0;
    for (int chunk = 0; chunk < total; chunk += BLOCK) {
        const int  lt   = chunk + tid;
        const bool live = (lt < total);
        float dist = INF;
        int   pos  = -1;
        if (live) {
#if JHQ_PROBE_CURSOR
            while (p + 1 < nprobe && lt >= my_offs[p + 1]) ++p;
            pos = list_offsets[my_ids[p]] + (lt - my_offs[p]);
#else
            int pp = p;
            while (pp + 1 < nprobe && lt >= my_offs[pp + 1]) ++pp;
            pos = list_offsets[my_ids[pp]] + (lt - my_offs[pp]);
#endif
        }
#if JHQ_EXIT_BOUND
        {
            // Every table entry is a squared distance, so the subspaces not
            // yet visited contribute at least s_suf[mhi] and
            //     complete >= a + s_suf[mhi].
            // Past the ck-th best that candidate cannot enter the top-ck, and
            // the rest of its subspaces need be neither gathered nor looked
            // up. Exact -- the same argument the compaction makes, one group
            // of subspaces at a time.
            //
            // v48 bounded the remainder by zero instead, the loosest bound
            // available, and lost 1-3%: the exit is per warp, so it fires only
            // when all 32 lanes are finished, and the maximum over 32 lanes is
            // far above the mean. A tighter bound moves every lane at once,
            // which is what the vote needs.
            //
            // The threshold is read once, unsynchronised, as the append below
            // already does; *s_thresh is the ck-th best seen so far and so
            // never below the final cut, which makes either value safe.
            const float  thr = *s_thresh;
            const lut_t* t   = lut_in_smem ? s_lut : g_lut;
            float a      = 0.0f;
            bool  done   = !live;
            bool  pruned = false;
            for (int m0 = 0; m0 < M; m0 += JHQ_EXIT_EVERY) {
                if (!done) {
                    const int mhi = (m0 + JHQ_EXIT_EVERY < M) ? (m0 + JHQ_EXIT_EVERY) : M;
#if JHQ_WORD_LAYOUT
                    {
                        const uint32_t* __restrict__ pw =
                            reinterpret_cast<const uint32_t*>(list_primary_t);
                        for (int mm = m0; mm < mhi; mm += 4) {
                            const uint32_t word = __ldg(&pw[(long long)(mm >> 2) * N + pos]);
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const uint32_t cm = (word >> (8 * i)) & 0xFFu;
                                const int m = mm + i;
                                a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                                   + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                            }
                        }
                    }
#else
                    for (int m = m0; m < mhi; ++m) {
                        const uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos]);
                        a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                           + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                    }
#endif
                    if (a + s_suf[mhi] > thr) { done = true; pruned = true; }
                }
                // Uniform across the warp, which is why the loop sits outside
                // the live test: every lane has to reach the vote, including
                // those past `total` in the final chunk, which enter done.
                if (__all_sync(0xffffffffu, done)) break;
            }
            // A pruned candidate must not carry its partial sum out of here.
            //
            // v48 could: its bound was `a > thr` on its own, so the partial
            // sum was already past the threshold and the append below rejected
            // it for the same reason the exit fired. That reasoning does not
            // survive a tighter bound. Here only `a + s_suf[mhi]` is known to
            // exceed thr, and `a` alone may sit well under it -- so writing
            // the partial sum out let pruned candidates into the buffer at a
            // distance that understated them, displacing real ones. Measured:
            // recall 0.08 against 0.98.
            if (live) dist = pruned ? INF : a;
        }
#else
        if (live) {
            const lut_t* t = lut_in_smem ? s_lut : g_lut;
            float a = 0.0f;
#if JHQ_WORD_LAYOUT
            // One 32-bit word covers four subspaces, and the warp's 32 threads
            // read 32 consecutive words: a single 128-byte transaction with
            // nothing discarded, where the byte layout used a quarter of one.
            const uint32_t* __restrict__ pw =
                reinterpret_cast<const uint32_t*>(list_primary_t);
            for (int m0 = 0; m0 < M; m0 += 4) {
                const uint32_t word = __ldg(&pw[(long long)(m0 >> 2) * N + pos]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t cm = (word >> (8 * i)) & 0xFFu;
                    const int m = m0 + i;
                    a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                       + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                }
            }
#else
            for (int m0 = 0; m0 < M; m0 += TILE_M) {
                const int mhi = (m0 + TILE_M < M) ? (m0 + TILE_M) : M;
                for (int m = m0; m < mhi; ++m) {
                    const uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos]);
                    a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                       + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                }
            }
#endif
            dist = a;
        }
#endif
#if !JHQ_PROBE_CURSOR
        // The old advance: one thread updates its own register, which does
        // nothing for the other 1023. Kept so the two can be built and timed
        // against each other.
        if (tid == BLOCK - 1 || lt == total - 1) {
            int pp = p;
            const int last = (total - 1 < chunk + BLOCK - 1) ? total - 1 : chunk + BLOCK - 1;
            while (pp + 1 < nprobe && last >= my_offs[pp + 1]) ++pp;
            p = pp;
        }
#endif
        __syncthreads();

        if (live && dist < *s_thresh) {
            const int slot = atomicAdd(s_cnt, 1);
            s_val[slot] = dist;      // slot < cap by the sizing argument above
            s_pos[slot] = pos;
        }
        __syncthreads();

        // Same correction as select_probes above. cap is the power of two at
        // or above ck + BLOCK, and at ck=1000 with BLOCK=1024 that is 2048:
        // the old test fired as soon as the buffer held 1001, so every chunk
        // after the first paid a 2048-element sort to take in a handful of
        // new candidates. results/block_sweep/ measured that cost at 19-31%
        // of the scan and rising with the block size on the large sets, which
        // is what four times the threads contending for the same 24 free
        // slots looks like.
        if (*s_cnt + BLOCK > cap)
            jhq_compact_topck(s_val, s_pos, s_cnt, s_thresh, ck, cap, tid, BLOCK);
    }

    // One final ordering so the output is sorted by primary distance.
    jhq_compact_topck(s_val, s_pos, s_cnt, s_thresh, ck, cap, tid, BLOCK);

    const int  n   = *s_cnt;

#if JHQ_REGROUP
    // Reorder the survivors by their position in the inverted lists before the
    // residual codes are read.
    //
    // The top-ck comes out of a bitonic sort on distance, so the positions are
    // in no order at all, and the refinement reads bpv bytes at each of them:
    // 1000 scattered 1 KB reads a query at d=1024. Sorting on position makes
    // those reads run forward through list_res instead.
    //
    // It cannot change the answer. Which ck survive was already decided, each
    // composite distance is computed from that candidate alone -- so not even
    // the summation order moves -- and the final top-k re-sorts on the
    // composite distance regardless. Algorithm 1 fixes the set and the
    // arithmetic; it says nothing about the order they are visited in.
    //
    // The key is (position, distance) rather than position alone, so that two
    // candidates at one position -- which cannot happen, but the sort must be
    // total for the network to be deterministic -- still order the same way.
    {
        const float INFF = __int_as_float(0x7F800000);
        for (int i = n + tid; i < cap; i += BLOCK) { s_pos[i] = 0x7FFFFFFF; s_val[i] = INFF; }
        __syncthreads();
        for (int k2 = 2; k2 <= cap; k2 <<= 1)
            for (int j = k2 >> 1; j > 0; j >>= 1) {
                for (int i = tid; i < cap; i += BLOCK) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
                        const bool up = ((i & k2) == 0);
                        if ((s_pos[i] > s_pos[ixj]) == up) {
                            int   p = s_pos[i]; s_pos[i] = s_pos[ixj]; s_pos[ixj] = p;
                            float t = s_val[i]; s_val[i] = s_val[ixj]; s_val[ixj] = t;
                        }
                    }
                }
                __syncthreads();
            }
    }
#endif

#if JHQ_FUSE_REFINE
    // Refine here rather than in a kernel of its own. The unfused path writes
    // ck positions and ck primary distances to global memory and reads them
    // straight back -- 16 KB a query at ck=1000, plus a launch -- and the
    // residual codes have to come from global either way. The selection and
    // every distance are unchanged; only where the work happens moves.
    {
        for (int i = tid; i < d; i += BLOCK)
            s_q[i] = q_rot[(long long)bqi * d + i];
        __syncthreads();
        float* out_comp = comp_dists + (long long)bqi * ck;
        for (int c = tid; c < ck; c += BLOCK) {
            if (c >= n) { out_comp[c] = INF; continue; }
            const int pos = s_pos[c];
            if (pos < 0)  { out_comp[c] = INF; continue; }
            const uint8_t* rc = list_res + (long long)pos * bpv;
            float d_res = 0.0f;
            int j = 0;
            for (int m = 0; m < M; ++m) {
                const float* cbm = res_c1d + (long long)m * Kr;
                for (int jj = 0; jj < Ds; ++jj, ++j) {
                    const int ri = (Br == 4)
                        ? ((j & 1) == 0 ? (rc[j >> 1] & 0x0F) : (rc[j >> 1] >> 4))
                        : rc[j];
                    const float diff = s_q[j] - cbm[ri];
                    d_res += __fmul_rn(diff, diff);
                }
            }
            out_comp[c] = s_val[c] + d_res + list_corr[pos];
        }
    }
    // The final top-k kernel reads positions to map back to ids, so they still
    // have to be written; the primary distances do not, since the composite
    // ones are already here.
    {
        int* out_pos = topck_pos + (long long)bqi * ck;
        for (int c = tid; c < ck; c += BLOCK)
            out_pos[c] = (c < n) ? s_pos[c] : -1;
    }
#else
    float* out_pri = topck_primary + (long long)bqi * ck;
    int*   out_pos = topck_pos     + (long long)bqi * ck;
    for (int c = tid; c < ck; c += BLOCK) {
        const bool have = (c < n);
        out_pri[c] = have ? s_val[c] : INF;
        out_pos[c] = have ? s_pos[c] : -1;
    }
#endif
}

// v62: list-group primary screening. Grid (B, G): block (q, g) owns an equal
// share of query q's candidates -- the probed lists laid end to end in probe
// order, the way scan_ivf_exact_kernel walks them, cut at
// [g*total/G, (g+1)*total/G) -- so every candidate belongs to exactly one
// group, the shares differ by at most one candidate, and a long list may be
// split between two groups. The cut is computed here from query_total on the
// device; nothing waits on the host.
//
// Each candidate's primary distance is computed by the same code, in the same
// order, as scan_ivf_exact_kernel -- the same table, the same word loads, the
// same additions -- so the distances are bit-identical to v61's. The block
// keeps its group's exact top-ck under key_less with the threshold-compaction
// buffer, and writes it sorted under key_less to
//   out_val/out_pos [(q*G + g)*ck ...], out_cnt[q*G + g] = how many are valid,
// or, when G == 1 (out_cnt null), straight into the query's top-ck arrays with
// the unused tail padded (INF, -1), which is the layout the refinement reads.
//
// Not implemented here, by design: JHQ_EXIT_BOUND, JHQ_FUSE_REFINE and
// JHQ_REGROUP. v62 measures the parallel exact pipeline alone; the launch
// refuses G >= 1 in a build that sets any of them.
__global__ void scan_ivf_group_kernel(
    const lut_t*   __restrict__ d_byte_lut,
    const int*     __restrict__ probe_ids,
    const int*     __restrict__ probe_offsets,
    const int*     __restrict__ list_offsets,
    const uint8_t* __restrict__ list_primary_t,
    const int*     __restrict__ query_total,
    float* out_val, int* out_pos, int* out_cnt,
    unsigned long long* qkey,     // v63: the queries' shared bounds [B], or null (G == 1, or off)
    int nprobe, int M, int N, int ck, int cap, bool lut_in_smem, int tile_m, int G)
{
    const float INF   = __int_as_float(0x7F800000);
    const int   BLOCK = blockDim.x;
    const int   bqi   = blockIdx.x;
    const int   g     = blockIdx.y;
    const int   tid   = threadIdx.x;

    extern __shared__ char shm[];
    float* s_val    = (float*)shm;
    int*   s_pos    = (int*)(s_val + cap);
    int*   s_cnt    = (int*)(s_pos + cap);
    float* s_thresh = (float*)(s_cnt + 1);
    int*   s_tpos   = (int*)(s_thresh + 1);
    unsigned long long* s_shared = (unsigned long long*)(shm + (size_t)cap * 8 + 16);
    lut_t* s_lut    = (lut_t*)(shm + (size_t)cap * 8 + 24);

    const lut_t* g_lut = d_byte_lut + (long long)bqi * M * JHQ_SPLIT_LUT;
    if (lut_in_smem)
        for (int i = tid; i < M * JHQ_SPLIT_LUT; i += BLOCK) s_lut[i] = g_lut[i];
    if (tid == 0) { *s_cnt = 0; *s_thresh = INF; *s_tpos = 0x7FFFFFFF; *s_shared = ~0ull; }
    __syncthreads();

    const int  total   = query_total[bqi];
    const int  lt0     = (int)(((long long)total * g) / G);
    const int  lt1     = (int)(((long long)total * (g + 1)) / G);
    const int* my_ids  = probe_ids     + bqi * nprobe;
    const int* my_offs = probe_offsets + bqi * (nprobe + 1);
    (void)tile_m;

    // The probe holding this thread's first candidate: the last p with
    // my_offs[p] <= lt. After that the cursor only moves forward, as in the
    // query-centric scan.
    int p = 0;
    {
        const int lt = lt0 + tid;
        int lo = 0, hi = nprobe - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (my_offs[mid] <= lt) lo = mid; else hi = mid - 1;
        }
        p = lo;
    }
    for (int chunk = lt0; chunk < lt1; chunk += BLOCK) {
        // v63: the query's shared bound as of this chunk, for the admission
        // below (the __syncthreads after the distances publishes it)
        if (qkey && tid == 0) *s_shared = *(volatile unsigned long long*)&qkey[bqi];
        const int  lt   = chunk + tid;
        const bool live = (lt < lt1);
        float dist = INF;
        int   pos  = -1;
        if (live) {
            while (p + 1 < nprobe && lt >= my_offs[p + 1]) ++p;
            pos = list_offsets[my_ids[p]] + (lt - my_offs[p]);
            const lut_t* t = lut_in_smem ? s_lut : g_lut;
            float a = 0.0f;
#if JHQ_WORD_LAYOUT
            const uint32_t* __restrict__ pw = reinterpret_cast<const uint32_t*>(list_primary_t);
            for (int m0 = 0; m0 < M; m0 += 4) {
                const uint32_t word = __ldg(&pw[(long long)(m0 >> 2) * N + pos]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t cm = (word >> (8 * i)) & 0xFFu;
                    const int m = m0 + i;
                    a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                       + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                }
            }
#else
            for (int m0 = 0; m0 < M; m0 += tile_m) {
                const int mhi = (m0 + tile_m < M) ? (m0 + tile_m) : M;
                for (int m = m0; m < mhi; ++m) {
                    const uint8_t cm = __ldg(&list_primary_t[(long long)m * N + pos]);
                    a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                       + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                }
            }
#endif
            dist = a;
        }
        __syncthreads();
        if (live) {
            const float thr = *s_thresh;
            const int   tps = *s_tpos;
            // v63: and not after the shared bound -- some group already holds
            // ck candidates at or before it
            if (key_less(dist, pos, thr, tps) && (!qkey || key64(dist, pos) <= *s_shared)) {
                const int slot = atomicAdd(s_cnt, 1);
                s_val[slot] = dist;           // slot < cap: at most ck kept + BLOCK new
                s_pos[slot] = pos;
            }
        }
        __syncthreads();
        if (*s_cnt + BLOCK > cap) {
            jhq_compact_topck_lex(s_val, s_pos, s_cnt, s_thresh, s_tpos, ck, cap, tid, BLOCK);
            // v63: publish this group's ck-th key, a bound on the query's
            if (qkey && tid == 0 && *s_cnt >= ck) atomicMin(&qkey[bqi], key64(*s_thresh, *s_tpos));
        }
    }
    jhq_compact_topck_lex(s_val, s_pos, s_cnt, s_thresh, s_tpos, ck, cap, tid, BLOCK);

    int n = *s_cnt;
    if (qkey) {
        // v63: publish, then keep only the sorted prefix at or before the
        // bound as it now stands; everything after it is outside the global
        // top-ck. The element that set the bound is at it, and is kept.
        if (tid == 0) {
            if (n >= ck) atomicMin(&qkey[bqi], key64(*s_thresh, *s_tpos));
            const unsigned long long sk = *(volatile unsigned long long*)&qkey[bqi];
            int lo = 0, hi = n;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (key64(s_val[mid], s_pos[mid]) <= sk) lo = mid + 1; else hi = mid;
            }
            *s_shared = (unsigned long long)lo;   // free: last read before the final compaction
        }
        __syncthreads();
        n = (int)*s_shared;
    }
    if (out_cnt) {
        const long long base = ((long long)bqi * G + g) * ck;
        for (int c = tid; c < n; c += BLOCK) { out_val[base + c] = s_val[c]; out_pos[base + c] = s_pos[c]; }
        if (tid == 0) out_cnt[bqi * G + g] = n;
    } else {
        const long long base = (long long)bqi * ck;
        for (int c = tid; c < ck; c += BLOCK) {
            const bool have = (c < n);
            out_val[base + c] = have ? s_val[c] : INF;
            out_pos[base + c] = have ? s_pos[c] : -1;
        }
    }
}

// v62: the global top-ck of each query from its G sorted local lists.
//
// One block per query, log2(G) rounds; each round merges the lists in pairs,
// keeping the first min(ck, a + b) of each merged pair, until one list is
// left, which goes to the query's top-ck arrays with the tail padded (INF, -1).
// A pair is merged by T/pairs threads, each producing an equal slice of the
// output: merge path finds where the slice starts with a binary search along
// the diagonal, then the thread merges sequentially. All lists live in global
// memory -- grp for even rounds, mrg for odd, the two sides of a ping-pong --
// so shared memory holds only the 2*G list lengths, whatever ck and G are.
//
// Exact: every element of the query's global top-ck is in its group's local
// top-ck (fewer than ck candidates precede it overall, so fewer than ck do in
// its group), and the first min(ck, a+b) of a merge of two sorted lists are the
// smallest min(ck, a+b) of their union. Ties cannot occur: key_less is total.
__global__ void merge_group_topk_kernel(
    float* grp_val, int* grp_pos, const int* __restrict__ grp_cnt,
    float* mrg_val, int* mrg_pos, float* topck_primary, int* topck_pos, int ck, int G)
{
    const float INF = __int_as_float(0x7F800000);
    const int q = blockIdx.x, tid = threadIdx.x, T = blockDim.x;
    __shared__ int s_n[2][8];
    if (tid < G) s_n[0][tid] = grp_cnt[q * G + tid];
    __syncthreads();
    float* gv = grp_val + (long long)q * G * ck;       int* gp = grp_pos + (long long)q * G * ck;
    float* mv = mrg_val + (long long)q * (G / 2) * ck; int* mp = mrg_pos + (long long)q * (G / 2) * ck;
    float* ov = topck_primary + (long long)q * ck;     int* op = topck_pos + (long long)q * ck;
    int nl = G, cur = 0;
    while (nl > 1) {
        const int  pairs = nl >> 1;
        const bool last  = (pairs == 1);
        const float* sv = cur ? mv : gv;  const int* sp = cur ? mp : gp;
        float*       dv = last ? ov : (cur ? gv : mv);
        int*         dp = last ? op : (cur ? gp : mp);
        const int tpp = T / pairs;
        const int pr  = tid / tpp, t = tid - pr * tpp;
        if (pr < pairs) {
            const float* Av = sv + (long long)(2 * pr) * ck;     const int* Ap = sp + (long long)(2 * pr) * ck;
            const float* Bv = sv + (long long)(2 * pr + 1) * ck; const int* Bp = sp + (long long)(2 * pr + 1) * ck;
            const int a = s_n[cur][2 * pr], b = s_n[cur][2 * pr + 1];
            const int o = (a + b < ck) ? a + b : ck;
            const int i0 = (int)(((long long)o * t) / tpp);
            const int i1 = (int)(((long long)o * (t + 1)) / tpp);
            // merge path: how many of the first i0 outputs come from A
            int lo = (i0 - b > 0) ? i0 - b : 0, hi = (i0 < a) ? i0 : a;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (key_less(Av[mid], Ap[mid], Bv[i0 - mid - 1], Bp[i0 - mid - 1])) lo = mid + 1;
                else hi = mid;
            }
            int ia = lo, ib = i0 - lo;
            float* Dv = dv + (long long)pr * ck; int* Dp = dp + (long long)pr * ck;
            for (int i = i0; i < i1; ++i) {
                const bool takeA = (ib >= b) || (ia < a && key_less(Av[ia], Ap[ia], Bv[ib], Bp[ib]));
                if (takeA) { Dv[i] = Av[ia]; Dp[i] = Ap[ia]; ++ia; }
                else       { Dv[i] = Bv[ib]; Dp[i] = Bp[ib]; ++ib; }
            }
            if (t == 0) s_n[1 - cur][pr] = o;
            if (last)
                for (int i = o + t; i < ck; i += tpp) { Dv[i] = INF; Dp[i] = -1; }
        }
        __syncthreads();          // this round's lists are complete before the next reads them
        cur = 1 - cur;
        nl  = pairs;
    }
}

__global__ void scan_ivf_coalesced_kernel(
    const lut_t*   __restrict__ d_byte_lut,     // [B, M, JHQ_SPLIT_LUT]
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

    const lut_t* g_lut = d_byte_lut + (long long)bqi * M * JHQ_SPLIT_LUT;
    if (lut_in_smem) {
        for (int i = tid; i < M * JHQ_SPLIT_LUT; i += BLOCK)
            s_lut[i] = g_lut[i];
        __syncthreads();
    }
    // The table is float; LUT_LOAD passes it through.
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
                    {
                        const lut_t* t = lut_in_smem ? my_lut_s : g_lut;
                        a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                           + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                    }
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
                    {
                        const lut_t* t = lut_in_smem ? my_lut_s : g_lut;
                        a += LUT_LOAD(t[m * JHQ_SPLIT_LUT + (cm >> 4)])
                           + LUT_LOAD(t[m * JHQ_SPLIT_LUT + 16 + (cm & 15)]);
                    }
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
// The residual codebook is 128 KB at M=128, Kr=256, and every candidate of
// every query reads all of it -- one lookup per dimension. It is the hot set;
// list_res, read once per candidate, is what streams past it.
//
// v38 tried three ways to keep the codebook from being evicted, measured
// against each other three times on three datasets:
//
//   __ldcs on the list_res loads   +14.4% / +8.8% / +14.6%   worse
//   codebook into opt-in shared     -3.1% /  0    /  0       vogue only
//   carveout biased to L1           -2.3% / -2.2% / -2.2%    everywhere
//
// __ldcs was backwards. A candidate's code is read once, but a *cache line* of
// it is read 32 times -- byte j = 0 fetches the line that serves j = 0..31 --
// and marking it evict-first destroys that reuse. Shared memory works only
// where the codebook fits: 3072 + 98304 = 101,376 is exactly this card's
// opt-in limit at M=96, and M=128 wants 135,168.
//
// So: shared where it fits, and where it does not, the opposite carveout --
// the smallest shared partition, leaving the most of the unified block as L1.
// That is what this version does, with no flag, since the measurement above is
// what chose it.
// Templated on Br so the nibble unpack is resolved at compile time. It sat in
// the innermost loop as a runtime ternary on a kernel argument -- uniform, so
// never divergent, but an instruction and a barrier to unrolling executed
// d * ck times a query: a million on stella. Two instantiations, dispatched at
// the launch.
template<int BR>
__global__ void residual_refine_fused_kernel(
    const int*     __restrict__ topck_pos,
    const float*   __restrict__ topck_primary,
    const float*   __restrict__ q_rot,
    const float*   __restrict__ res_c1d,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int ck, int d, int Ds, int Kr, int M, int bpv, int B,
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
                const int ri = (BR == 4)
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

// v61: the same refinement with L lanes per candidate and 32-bit code loads,
// ported unchanged from jhq_v60_refine_probe, where it was measured on the
// paper's index (RTX 5090, np=128): x1.18-2.36 end to end at the sampled-rule
// alpha with recall unchanged. Selected by set_refine_lanes() / JHQ_REFINE_LANES;
// 0 keeps residual_refine_fused_kernel above.
//
// What each change buys, separately:
//   * L=1 keeps one thread per candidate but reads its code a 32-bit word at a
//     time instead of a byte at a time -- a quarter of the load instructions
//     and address computations, in the fused kernel's summation order, so its
//     distances are bit-identical to it.
//   * L>1 puts L lanes on one candidate. At the small ck the sampled rule
//     picks (40 on openai3-3072) the fused kernel leaves most of a block idle
//     while each busy thread walks d dimensions alone; L lanes cut that serial
//     walk by L, and adjacent lanes read adjacent words of one code. Each
//     candidate is then summed in a different order (lane partials, then a
//     butterfly), so distances move in the last bits and per-query results are
//     checked, not assumed.
//
// A word must not straddle two subspaces, so Ds is a multiple of the
// dimensions per word (4 at Br=8, 8 at Br=4), and bpv a multiple of 4; the
// launch checks both.
template<int BR, int L>
__global__ void residual_refine_group_kernel(
    const int*     __restrict__ topck_pos,
    const float*   __restrict__ topck_primary,
    const float*   __restrict__ q_rot,
    const float*   __restrict__ res_c1d,
    const uint8_t* __restrict__ list_res,
    const float*   __restrict__ list_corr,
    float*                      comp_dists,
    int ck, int d, int Ds, int Kr, int M, int bpv, int B,
    int cb_in_smem)
{
    static_assert(L == 1 || L == 8 || L == 16 || L == 32, "a group must tile a warp");
    constexpr int DPW = (BR == 4) ? 8 : 4;              // dimensions per 32-bit word
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
    const int lane   = threadIdx.x % L;
    const int groups = blockDim.x / L;
    const int words  = bpv >> 2;
    // Every thread makes the same number of passes, so all 32 lanes of a warp
    // reach each shuffle together.
    for (int c0 = 0; c0 < ck; c0 += groups) {
        const int c   = c0 + (int)threadIdx.x / L;
        const int pos = (c < ck) ? topck_pos[base + c] : -1;
        float d_res = 0.0f;
        if (pos >= 0) {
            const uint32_t* rw = (const uint32_t*)(list_res + (long long)pos * bpv);
            for (int w = lane; w < words; w += L) {
                const uint32_t v   = __ldg(rw + w);
                const int      j0  = w * DPW;
                const float*   cbm = cb + (long long)(j0 / Ds) * Kr;
#pragma unroll
                for (int t = 0; t < DPW; ++t) {
                    const int ri = (BR == 4) ? (int)((v >> (4 * t)) & 0x0Fu)
                                             : (int)((v >> (8 * t)) & 0xFFu);
                    const float diff = s_q[j0 + t] - cbm[ri];
                    d_res += __fmul_rn(diff, diff);
                }
            }
        }
#pragma unroll
        for (int off = L >> 1; off > 0; off >>= 1)
            d_res += __shfl_xor_sync(0xffffffffu, d_res, off);
        if (c < ck && lane == 0) {
#if JHQ_NO_RESIDUAL
            (void)d_res; (void)list_corr;
            comp_dists[base + c] = (pos < 0) ? __int_as_float(0x7F800000)
                                             : topck_primary[base + c];
#else
            comp_dists[base + c] = (pos < 0) ? __int_as_float(0x7F800000)
                                             : topck_primary[base + c] + d_res + list_corr[pos];
#endif
        }
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

// Threads for the refine launch. JHQ_REFINE_BLOCK=0 keeps the historical 256
// so the two can be A/B'd; anything else is clamped to a legal block size.
static int refine_block(int scan_block) {
    static const int v = [] {
        const char* e = std::getenv("JHQ_REFINE_BLOCK");
        return e ? std::atoi(e) : -1;
    }();
    if (v == 0) return 256;
    if (v == 128 || v == 256 || v == 512 || v == 1024) return v;
    return scan_block;
}

// v61: the refinement kernel in use (search.cuh describes the values). -2
// means not yet read from JHQ_REFINE_LANES.
static int g_refine_lanes = -2;

static int checked_lanes(int l, const char* from) {
    if (l != 0 && l != 1 && l != 8 && l != 16 && l != 32)
        throw std::runtime_error(std::string(from) + " must be 0, 1, 8, 16 or 32, got " +
                                 std::to_string(l));
    return l;
}

void set_refine_lanes(int lanes) { g_refine_lanes = checked_lanes(lanes, "set_refine_lanes"); }

int refine_lanes() {
    if (g_refine_lanes == -2) {
        const char* e = std::getenv("JHQ_REFINE_LANES");
        g_refine_lanes = checked_lanes(e ? std::atoi(e) : 0, "JHQ_REFINE_LANES");
    }
    return g_refine_lanes;
}

template<int BR>
static void set_refine_group_attr(cudaFuncAttribute attr, int val) {
    CUDA_CHECK(cudaFuncSetAttribute(residual_refine_group_kernel<BR, 1>,  attr, val));
    CUDA_CHECK(cudaFuncSetAttribute(residual_refine_group_kernel<BR, 8>,  attr, val));
    CUDA_CHECK(cudaFuncSetAttribute(residual_refine_group_kernel<BR, 16>, attr, val));
    CUDA_CHECK(cudaFuncSetAttribute(residual_refine_group_kernel<BR, 32>, attr, val));
}

template<int BR>
static void launch_refine_group(
    int L, int B, int RB, size_t smem, cudaStream_t s,
    const int* topck_pos, const float* topck_primary, const float* q_rot,
    const float* res_c1d, const uint8_t* list_res, const float* list_corr,
    float* comp_dists, int ck, int d, int Ds, int Kr, int M, int bpv, int cb_in_smem)
{
#define JHQ_REFINE_GROUP(LL)                                                   \
    residual_refine_group_kernel<BR, LL><<<B, RB, smem, s>>>(                  \
        topck_pos, topck_primary, q_rot, res_c1d, list_res, list_corr,         \
        comp_dists, ck, d, Ds, Kr, M, bpv, B, cb_in_smem)
    switch (L) {
        case 1:  JHQ_REFINE_GROUP(1);  break;
        case 8:  JHQ_REFINE_GROUP(8);  break;
        case 16: JHQ_REFINE_GROUP(16); break;
        default: JHQ_REFINE_GROUP(32); break;
    }
#undef JHQ_REFINE_GROUP
}

// v62: the primary execution in use (search.cuh describes the values). -2
// means not yet read from JHQ_PRIMARY_GROUPS.
static int    g_primary_groups      = -2;
static size_t g_group_scratch_bytes = 0;

static int checked_groups(int g, const char* from) {
    if (g != -1 && g != 0 && g != 1 && g != 2 && g != 4 && g != 8)
        throw std::runtime_error(std::string(from) + " must be -1 (scheduled), 0, 1, 2, 4 or 8, got " +
                                 std::to_string(g));
    return g;
}

void set_primary_groups(int groups) { g_primary_groups = checked_groups(groups, "set_primary_groups"); }

int primary_groups() {
    if (g_primary_groups == -2) {
        const char* e = std::getenv("JHQ_PRIMARY_GROUPS");
        g_primary_groups = checked_groups(e ? std::atoi(e) : 0, "JHQ_PRIMARY_GROUPS");
    }
    return g_primary_groups;
}

size_t primary_group_scratch_bytes() { return g_group_scratch_bytes; }

// v65: the scheduled primary execution (search.cuh). The skew is the
// coefficient of variation of the per-query candidate count at this index's
// nprobe, measured once by JHQGpuIndex::calibrate_schedule; < 0 = not measured.
static double g_sched_cv = -1.0;
void   set_schedule_skew(double cv) { g_sched_cv = cv; }
double schedule_skew() { return g_sched_cv; }

static int scan_block_threads() {
    static const int b = [] {
        const char* e = getenv("JHQ_BLOCK");
        const int   v = e ? atoi(e) : 256;
        return (v == 128 || v == 256 || v == 512 || v == 1024) ? v : 256;
    }();
    return b;
}

int scheduled_groups(int B) {
    static const long long resident = [] {
        int sms = 0, tps = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
        CUDA_CHECK(cudaDeviceGetAttribute(&tps, cudaDevAttrMaxThreadsPerMultiProcessor, 0));
        return (long long)sms * tps;
    }();
    // Fill: the largest G whose B*G scan blocks hold at most 0.6 of the
    // threads the device can keep resident. Skew: when per-query candidate
    // counts vary (CV >= 0.3), the largest query of a batch outlasts the rest
    // under one block per query, so at least 4 groups. Both constants were
    // fitted on pipeline_v62's four workloads alone, before any other ran.
    const long long T = scan_block_threads();
    int g_fill = 1;
    for (int g = 2; g <= 8; g <<= 1)
        if ((long long)B * g * T <= (long long)(0.6 * resident)) g_fill = g;
    const int g_skew = (g_sched_cv >= 0.3) ? 4 : 1;
    const int g = std::max(g_fill, g_skew);
    return g == 1 ? 0 : g;      // one group: v61's query-centric scan is the faster kernel
}

int effective_groups(int B) {
    const int g = primary_groups();
    return g == -1 ? scheduled_groups(B) : g;
}

// v63: the shared bound (search.cuh). -1 means not yet read from JHQ_SHARED_BOUND.
static int g_shared_bound = -1;

void set_shared_bound(bool on) { g_shared_bound = on ? 1 : 0; }

bool shared_bound() {
    if (g_shared_bound == -1) {
        const char* e = std::getenv("JHQ_SHARED_BOUND");
        g_shared_bound = (e && e[0] == '0') ? 0 : 1;
    }
    return g_shared_bound == 1;
}

// The list-group scratch, sized for batch_cap queries before any capture (a
// graph capture permits no allocation). It grows and never shrinks, like the
// top-ck buffers; G = 1 writes straight into the top-ck arrays and needs none.
static void realloc_group_buffers(SearchWorkspace& ws, int batch_cap, int ck, int G) {
    if (G <= 1) return;
    const long long ng = (long long)batch_cap * G * ck;
    const long long nm = (long long)batch_cap * (G / 2) * ck;
    const long long nc = (long long)batch_cap * G;
    if (ng > ws.grp_cap_elems) {
        cudaFree(ws.d_grp_val); cudaFree(ws.d_grp_pos);
        CUDA_CHECK(cudaMalloc(&ws.d_grp_val, ng * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_grp_pos, ng * sizeof(int)));
        ws.grp_cap_elems = ng;
    }
    if (nm > ws.mrg_cap_elems) {
        cudaFree(ws.d_mrg_val); cudaFree(ws.d_mrg_pos);
        CUDA_CHECK(cudaMalloc(&ws.d_mrg_val, nm * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.d_mrg_pos, nm * sizeof(int)));
        ws.mrg_cap_elems = nm;
    }
    if (nc > ws.grp_cnt_cap) {
        cudaFree(ws.d_grp_cnt);
        CUDA_CHECK(cudaMalloc(&ws.d_grp_cnt, nc * sizeof(int)));
        ws.grp_cnt_cap = nc;
    }
    if (batch_cap > ws.qkey_cap) {
        cudaFree(ws.d_qkey);
        CUDA_CHECK(cudaMalloc(&ws.d_qkey, (size_t)batch_cap * sizeof(unsigned long long)));
        ws.qkey_cap = batch_cap;
    }
    g_group_scratch_bytes = (size_t)(ws.grp_cap_elems + ws.mrg_cap_elems) * (sizeof(float) + sizeof(int))
                          + (size_t)ws.grp_cnt_cap * sizeof(int);
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
    int ck, int k, int ntotal,
    // Calibration walks a grid of alpha, and ck changes with every step, so
    // every step re-captured the graph. Measured: cal_ms was flat at ~99 ms
    // across S = 8 to 128 -- sixteen times the queries, the same cost -- and
    // scaled instead with the number of grid steps, about 15 ms each, against
    // roughly 0.8 ms of actual searching. The capture was 94% of it.
    bool capture)
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
    const int   lut_bytes = M * JHQ_SPLIT_LUT * (int)sizeof(lut_t);
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
    if (capture)
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
    // 33 KB of histogram plus 12 B per probe, independent of nlist and BLOCK
    // (36 KB at nprobe=256). Past 48 KB the opt-in is needed, and has to
    // happen before the launch is captured into the graph or the failure
    // surfaces at whichever check runs next instead of here.
    {
        const int probe_smem = probe_select_smem(nprobe);
        if (probe_smem > smem_optin)
            throw std::runtime_error(
                "select_probes_kernel needs " + std::to_string(probe_smem) +
                " B of shared memory for nprobe=" + std::to_string(nprobe) +
                "; the device allows " + std::to_string(smem_optin) +
                ". Lower nprobe.");
        if (probe_smem > 48 * 1024)
            CUDA_CHECK(cudaFuncSetAttribute(
                select_probes_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, probe_smem));
        select_probes_kernel<<<B, BLOCK, probe_smem, ws.stream>>>(
            ws.d_dots, d_cent_norms, d_list_offsets,
            ws.d_probe_ids, ws.d_probe_offsets, ws.d_query_total,
            nlist, nprobe);
    }
    CUDA_CHECK(cudaGetLastError());
#if JHQ_STEP_TIMING
    CUDA_CHECK(cudaEventRecord(ws.ev_step[3], ws.stream));
#endif
    // 4. Build byte LUT
    {
        long long tot  = (long long)B * M * JHQ_SPLIT_LUT;
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
        // Grow the buffer while it and the table still fit, so the sort fires
        // less often. Each doubling halves the number of compactions far more
        // than it lengthens one: the trigger moves from cnt > cap - BLOCK to
        // cnt > 2*cap - BLOCK, and a compaction always leaves cnt at ck.
        {
            const int lut_b = M * JHQ_SPLIT_LUT * (int)sizeof(lut_t);
            for (int mult = 1; mult < JHQ_CAP_MULT; ++mult) {
                const int bigger = cap * 2;
                const int need = bigger * (int)(sizeof(float) + sizeof(int))
                               + (int)sizeof(int) + (int)sizeof(float) + lut_b;
                if (need > smem_optin) break;
                cap = bigger;
            }
        }
        const int G = effective_groups(B);
        if (G >= 1) {
            // v62: G blocks per query, each with v61's buffer plus the
            // threshold's position (key_less), then the merge for G > 1.
#if JHQ_EXIT_BOUND || JHQ_FUSE_REFINE || JHQ_REGROUP
            throw std::runtime_error("list groups (G >= 1) are implemented for the exact scan "
                                     "alone; this build sets JHQ_EXIT_BOUND, JHQ_FUSE_REFINE or "
                                     "JHQ_REGROUP. Use G = 0.");
#endif
            // cap (distance, position) pairs, count, threshold and its position,
            // then (v63) the shared bound at the next 8-byte boundary
            const int gx_base = cap * (int)(sizeof(float) + sizeof(int)) + 24;
            const bool g_lut = (gx_base + lut_bytes) <= smem_optin;
            const int  g_smem = g_lut ? (gx_base + lut_bytes) : gx_base;
            if (g_smem > smem_optin)
                throw std::runtime_error(
                    "scan_ivf_group_kernel needs " + std::to_string(g_smem) +
                    " B of shared memory for ck=" + std::to_string(ck) + " at BLOCK=" +
                    std::to_string(BLOCK) + "; the device allows " + std::to_string(smem_optin) + ".");
            CUDA_CHECK(cudaFuncSetAttribute(scan_ivf_group_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem));
            // v63: the shared bounds start at "nothing known" every search
            const bool shared = (G > 1) && shared_bound();
            if (shared)
                CUDA_CHECK(cudaMemsetAsync(ws.d_qkey, 0xFF, (size_t)B * sizeof(unsigned long long), ws.stream));
            scan_ivf_group_kernel<<<dim3(B, G), BLOCK, g_smem, ws.stream>>>(
                ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
                d_list_primary_t, ws.d_query_total,
                G == 1 ? ws.d_topck_primary : ws.d_grp_val,
                G == 1 ? ws.d_topck_pos     : ws.d_grp_pos,
                G == 1 ? nullptr            : ws.d_grp_cnt,
                shared ? ws.d_qkey : nullptr,
                nprobe, M, ntotal, ck, cap, g_lut, tile_m, G);
            CUDA_CHECK(cudaGetLastError());
            if (G > 1) {
                merge_group_topk_kernel<<<B, 256, 0, ws.stream>>>(
                    ws.d_grp_val, ws.d_grp_pos, ws.d_grp_cnt, ws.d_mrg_val, ws.d_mrg_pos,
                    ws.d_topck_primary, ws.d_topck_pos, ck, G);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
        const int ex_base = cap * (int)(sizeof(float) + sizeof(int))
                          + (int)sizeof(int) + (int)sizeof(float);
        // JHQ_EXIT_BOUND needs M+1 floats for the suffix minima and reads the
        // table to build them, so it needs the table resident too.
        const int  suf_bytes = JHQ_EXIT_BOUND ? (M + 1) * (int)sizeof(float) : 0;
        // The fused refinement stages the rotated query; the residual codebook
        // stays in global, since M*Kr*4 is 131 KB at M=128 and will not fit
        // beside the scan buffers -- the same reason the standalone kernel
        // reads it from global there.
        const int  q_bytes_f = JHQ_FUSE_REFINE ? d * (int)sizeof(float) : 0;
        const int  extra   = suf_bytes + q_bytes_f;
        const bool ex_lut = (ex_base + lut_bytes + extra) <= smem_optin;
        const int  ex_smem = ex_lut ? (ex_base + lut_bytes + extra) : (ex_base + extra);
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
            nprobe, M, ntotal, ck, cap, ex_lut, tile_m,
            ws.d_q_rot, d_res_c1d, d_list_res, d_list_corr, ws.d_comp_dists,
            d, Ds, Kr, Br, bpv);
        CUDA_CHECK(cudaGetLastError());
        }   // v62: end of G == 0, v61's query-centric scan

    } else {
    if (effective_groups(B) != 0)
        throw std::runtime_error("list groups (G >= 1) need the exact top-ck scan "
                                 "(the paper-faithful path); this build selects the coalesced one.");
    scan_ivf_coalesced_kernel<<<B, BLOCK, scan_smem, ws.stream>>>(
        ws.d_byte_lut, ws.d_probe_ids, ws.d_probe_offsets, d_list_offsets,
        d_list_primary_t, ws.d_query_total,
        ws.d_topck_primary, ws.d_topck_pos,
        nprobe, M, ntotal, ck, lut_in_smem, pfx_num, pfx_den, tile_m);
    }
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
#if JHQ_FUSE_REFINE
    if (resid_lut_materialised())
        throw std::runtime_error(
            "JHQ_RESID_LUT and JHQ_FUSE_REFINE are two different refinement "
            "paths and this build has the fused one compiled in: the scan no "
            "longer writes topck_primary, which the materialised path reads. "
            "Use a build without JHQ_FUSE_REFINE to measure JHQ_RESID_LUT.");
#endif
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
        // 40960 is what a kernel gets without asking. Opting in raises it to
        // 101,376 on this card, which the vogue configuration needs exactly:
        // 3072 + 98304 = 101,376. M=128 at Kr=256 wants 135,168 and still does
        // not fit. Asking also costs: 99 KB of a 100 KB SM is one block
        // resident, so the occupancy that hides L2 latency goes away. Which
        // way that lands is measured, not assumed.
        // JHQ_REFINE_BLOCK picks the launch width. It was hard-coded at 256
        // while every other search kernel took JHQ_BLOCK, which is why the
        // refine share rose from 5-6% to 9% when the rest went to 1024: the
        // loop is `for (c = tid; c < ck; c += blockDim.x)` over ck=1000, so
        // 256 threads make four passes where 1024 make one.
#if JHQ_FUSE_REFINE
        // Already done inside the scan kernel.
#else
        const int RB = refine_block(BLOCK);
        const int RL = refine_lanes();
        static int cb_smem = -1;
        if (cb_smem < 0) {
            int optin = 40960;
            cudaDeviceGetAttribute(&optin,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
            cb_smem = (q_bytes + cb_bytes <= (size_t)optin) ? 1 : 0;
            const auto attr = cb_smem ? cudaFuncAttributeMaxDynamicSharedMemorySize
                                      : cudaFuncAttributePreferredSharedMemoryCarveout;
            const int  val  = cb_smem ? optin : 0;
            CUDA_CHECK(cudaFuncSetAttribute(residual_refine_fused_kernel<4>, attr, val));
            CUDA_CHECK(cudaFuncSetAttribute(residual_refine_fused_kernel<8>, attr, val));
            set_refine_group_attr<4>(attr, val);
            set_refine_group_attr<8>(attr, val);
        }
        const size_t rsmem = q_bytes + (cb_smem ? cb_bytes : 0);
        if (RL == 0) {
            if (Br == 4)
                residual_refine_fused_kernel<4><<<B, RB, rsmem, ws.stream>>>(
                    ws.d_topck_pos, ws.d_topck_primary, ws.d_q_rot, d_res_c1d,
                    d_list_res, d_list_corr, ws.d_comp_dists,
                    ck, d, Ds, Kr, M, bpv, B, cb_smem);
            else
                residual_refine_fused_kernel<8><<<B, RB, rsmem, ws.stream>>>(
                    ws.d_topck_pos, ws.d_topck_primary, ws.d_q_rot, d_res_c1d,
                    d_list_res, d_list_corr, ws.d_comp_dists,
                    ck, d, Ds, Kr, M, bpv, B, cb_smem);
        } else {
            const int dpw = (Br == 4) ? 8 : 4;
            if (Ds % dpw != 0 || bpv % 4 != 0)
                throw std::runtime_error(
                    "refine lanes " + std::to_string(RL) + " read 32-bit words, which needs Ds "
                    "a multiple of " + std::to_string(dpw) + " and bpv a multiple of 4; this "
                    "index has Ds=" + std::to_string(Ds) + ", bpv=" + std::to_string(bpv) +
                    ". Lanes 0 is the byte kernel.");
            if (Br == 4)
                launch_refine_group<4>(RL, B, RB, rsmem, ws.stream,
                    ws.d_topck_pos, ws.d_topck_primary, ws.d_q_rot, d_res_c1d,
                    d_list_res, d_list_corr, ws.d_comp_dists,
                    ck, d, Ds, Kr, M, bpv, cb_smem);
            else
                launch_refine_group<8>(RL, B, RB, rsmem, ws.stream,
                    ws.d_topck_pos, ws.d_topck_primary, ws.d_q_rot, d_res_c1d,
                    d_list_res, d_list_corr, ws.d_comp_dists,
                    ck, d, Ds, Kr, M, bpv, cb_smem);
        }
        CUDA_CHECK(cudaGetLastError());
#endif
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
    if (capture) {
        CUDA_CHECK(cudaStreamEndCapture(ws.stream, &ws.graph));
        CUDA_CHECK(cudaGraphInstantiate(&ws.graph_exec, ws.graph, nullptr, nullptr, 0));
    }
#endif

    ws.graph_ck     = ck;
    ws.graph_nprobe = nprobe;
    ws.graph_B      = B;
    ws.graph_lanes  = refine_lanes();
    ws.graph_groups = effective_groups(B);
    ws.graph_shared = shared_bound() ? 1 : 0;
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
    realloc_group_buffers(ws, ws.batch_cap, ck, primary_groups() == -1 ? 8 : primary_groups());

    // The launch is sized to the queries actually present, not to the
    // workspace.
    //
    // Every search used to run batch_cap blocks and zero-pad the rest, so a
    // 32-query calibration pass did 1024 queries' work with 992 of them
    // zeros. results/front6/alpha_fast.log shows what that cost: cal_ms was
    // flat from S=8 to S=128, scaled with the number of grid steps instead,
    // and did not move when the graph capture was removed -- because the cost
    // was never the capture, it was the kernels running at full width.
    //
    // The graph is fixed-size, so B joins ck and nprobe in its key. A run of
    // equal batches captures once as before; a remainder captures a second
    // graph and reuses it.

    SearchDiag& dg = search_diag();
    dg.on = diag_on();
    if (dg.on) {
        dg.nprobe = nprobe;
        dg.cand.assign((size_t)nq, 0);
        dg.probes.assign((size_t)nq * nprobe, -1);
    }

    for (int qoff = 0; qoff < nq; qoff += batch_size) {
        const int B = std::min(batch_size, nq - qoff);
        ws.last_B = B; ws.last_ck = ck; ws.last_nprobe = nprobe;   // v62: debug_last_batch
        if (!ws.no_graph &&
            (!ws.graph_exec || ws.graph_ck != ck ||
             ws.graph_nprobe != nprobe || ws.graph_B != B ||
             ws.graph_lanes != refine_lanes() || ws.graph_groups != effective_groups(B) ||
             ws.graph_shared != (shared_bound() ? 1 : 0))) {
            capture_graph(ws, cublas,
                          d_Pi, d_cent, d_res_c1d, d_centroids, d_cent_norms,
                          d_list_offsets, d_list_ids, d_list_primary_t,
                          d_list_res, d_list_corr,
                          B, d, M, Ds, K, Kr, nlist, nprobe, Br, bpv,
                          ck, k, ntotal, true);
        }
        std::memcpy(ws.h_q_pinned,
                    h_queries + (long long)qoff * d, (long long)B * d * sizeof(float));

        CUDA_CHECK(cudaMemcpyAsync(ws.d_q_batch, ws.h_q_pinned,
                                   (long long)B * d * sizeof(float),
                                   cudaMemcpyHostToDevice, ws.stream));
#if JHQ_STEP_TIMING
        // No graph to launch in the timed build: re-run the sequence so the
        // events land on this batch.
        capture_graph(ws, cublas,
                      d_Pi, d_cent, d_res_c1d, d_centroids, d_cent_norms,
                      d_list_offsets, d_list_ids, d_list_primary_t, d_list_res,
                      d_list_corr, B, d, M, Ds, K, Kr, nlist, nprobe,
                      Br, bpv, ck, k, ntotal, false);
#else
        if (ws.no_graph)
            capture_graph(ws, cublas,
                          d_Pi, d_cent, d_res_c1d, d_centroids, d_cent_norms,
                          d_list_offsets, d_list_ids, d_list_primary_t, d_list_res,
                          d_list_corr, B, d, M, Ds, K, Kr, nlist, nprobe,
                          Br, bpv, ck, k, ntotal, false);
        else
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
        // On ws.stream, so it is ordered after this batch's graph and before
        // the next batch overwrites the buffers. Off the timed path only in
        // the sense that it is off by default: with JHQ_DIAG=1 the two extra
        // copies are inside the measured region, so a diag run's QPS is not a
        // result. That is deliberate -- a diagnostic that silently changed the
        // timing would be worse than one that visibly does.
        if (dg.on) {
            CUDA_CHECK(cudaMemcpyAsync(dg.cand.data() + qoff, ws.d_query_total,
                                       (size_t)B * sizeof(int),
                                       cudaMemcpyDeviceToHost, ws.stream));
            CUDA_CHECK(cudaMemcpyAsync(dg.probes.data() + (size_t)qoff * nprobe,
                                       ws.d_probe_ids,
                                       (size_t)B * nprobe * sizeof(int),
                                       cudaMemcpyDeviceToHost, ws.stream));
        }
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
    report_step_timing(ws, std::min(batch_size, nq), nprobe, ck);
#endif
}

} // namespace jhq_gpu
