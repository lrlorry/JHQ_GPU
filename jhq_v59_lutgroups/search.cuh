#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

namespace jhq_gpu {

// ── Search diagnostics (JHQ_DIAG=1) ──────────────────────────────────────────
// Two numbers this project has been quoting rather than measuring.
//
// `cand` is what the scan actually read. Every table here has so far given
// N*nprobe/nlist instead, which assumes the lists are the same length; they
// are not, and a query that probes the four fattest lists scans far more than
// the mean. select_probes_kernel already sums the exact figure into
// query_total to bound its own loop, so the measurement costs one copy.
//
// `probes` is the list each query actually opened, which with the vector->list
// map gives IVF candidate recall: the fraction of the true top-k that coarse
// routing brought into range at all. It splits a low recall into two causes
// that are otherwise indistinguishable -- neighbours never routed in, versus
// neighbours routed in and then lost by the primary filter or the refinement.
// Raising nprobe fixes the first and does nothing for the second.
struct SearchDiag {
    bool               on = false;
    int                nprobe = 0;
    std::vector<int>   cand;     // [nq]        candidates the scan read
    std::vector<int>   probes;   // [nq*nprobe] list ids opened, in probe order
};
// One per process; filled by the last search_gpu() call when JHQ_DIAG=1.
SearchDiag& search_diag();


// v12_transposed: transpose list_primary from [N, M] to [M, N] so that
// the scan inner loop accesses list_primary_t[m * N + abs_pos].
//
// With the original [N, M] layout, 32 warp threads each read the m-th byte
// of their own candidate: addresses differ by N_stride × M = 24 576 bytes →
// 32 separate cache lines loaded, 4 bytes used per line (98% waste).
//
// With [M, N] layout, 32 warp threads read 32 consecutive abs_pos values at
// the same m: 32 consecutive bytes = 1 cache line, 100% utilisation.
// This reduces list_primary L2/HBM traffic by 32× per warp transaction.
//
// All other changes from v10_bytelut are preserved:
//   - byte_lut [B, M, 256] (no bank conflict, v10)
//   - CUDA graph (v5+)
//   - spin-wait sync (v7)

// The primary lookup table is factorised, and that removes the reason it was
// ever a __half.
//
// Equation 4 builds each subspace's K codewords as the Cartesian product of L
// per-dimension levels, so a code byte is Ds base-L digits and the distance
//
//     T_m[c] = sum_j (q_j - level[digit_j(c)])^2
//
// splits along any partition of the Ds dimensions. Splitting it in half:
//
//     T_m[c] = Thi_m[c >> 4] + Tlo_m[c & 15]
//
// which holds exactly whenever K = 256 and Ds is even, because then the high
// nibble carries the first Ds/2 digits and the low nibble the rest. The table
// per subspace goes from 256 entries to 32 -- 8x smaller -- and building it
// costs B*M*32 entries of Ds/2 terms instead of B*M*256 of Ds, which is 16x
// less work.
//
// 32 floats a subspace also fixes the bank pattern. With the 256-entry table
// the addresses are m*256 + c over all 32 banks with 8 addresses per bank, so
// a warp reading 32 data-dependent codes takes up to an 8-way conflict. With
// 32 entries the addresses are m*32 + h with h in 0..15, which lands in banks
// 0..15 with one address each: two threads either share an address and
// broadcast, or sit in different banks. No conflict.
//
// So it stays float. The half table existed to fit 256 entries a subspace into
// shared memory (results/lut_policy/); 32 fit in fp32 with room to spare --
// M*32*4 is 12 KB at M=96 and 49 KB at M=384 -- and there is nothing left to
// buy by giving up mantissa.
typedef float jhq_lut_t;

// ── the split granularity, as a compile-time constant ─────────────────────
//
// Splitting in half is one choice out of several. The identity holds for any
// partition of the Ds base-L digits, so at Ds = B = 8 and L = 2 the code's
// eight one-bit digits can be cut into G groups of 8/G bits each:
//
//   G   tables/subspace   entries/subspace   loads+adds per candidate
//   1   1 x 256           256                1
//   2   2 x 16             32                2      <- the shipped default
//   4   4 x 4              16                4
//   8   8 x 2              16                8
//
// G=8 is dominated: the same table as G=4 for twice the work. The question
// the sweep answers is G=2 against G=4, and it is not obvious which wins.
// G=4 halves shared memory again -- at M=384 that is 48 KiB a block against
// 24 KiB, inside the range where occupancy moves on this card -- and doubles
// the per-candidate work in exchange. So the prediction is that G=2 wins at
// small M and G=4 at large M, and if that is what happens then the choice is
// a design principle rather than an implementation detail.
//
// G is a macro and not a runtime parameter on purpose. Every lookup site
// unrolls over it, so a runtime G would add a loop and a branch to the
// innermost line of the scan and measure the branch instead of the split.
#ifndef JHQ_LUT_GROUPS
#define JHQ_LUT_GROUPS 2
#endif
#if (8 % JHQ_LUT_GROUPS) != 0 || JHQ_LUT_GROUPS < 1 || JHQ_LUT_GROUPS > 8
#error "JHQ_LUT_GROUPS must divide 8: one of 1, 2, 4, 8"
#endif

// bits a group, entries a group, entries a subspace
#define JHQ_LUT_GBITS (8 / JHQ_LUT_GROUPS)
#define JHQ_LUT_GSIZE (1 << JHQ_LUT_GBITS)
#define JHQ_SPLIT_LUT (JHQ_LUT_GROUPS * JHQ_LUT_GSIZE)

// Group g covers the digits that group g of the code byte carries, counted
// from the most significant -- the same orientation the half-split used, where
// the high nibble stood for the first Ds/2 dimensions. Keeping that
// orientation means verify_split_lut()'s representative-centroid argument
// generalises instead of needing a second proof.
#define JHQ_LUT_DIGIT(cm, g)     (((cm) >> (8 - ((g) + 1) * JHQ_LUT_GBITS)) & (JHQ_LUT_GSIZE - 1))

// The per-subspace lookup itself lives in search.cu, next to LUT_LOAD, which
// is defined there. Only the shape of the table is fixed here.

// Subspaces between two checks of the exact exit bound, and whether the bound
// is on at all. v48 bounded the unvisited subspaces by zero and lost 1-3%;
// v49 bounds them by their smallest table entry, which the factorised table
// makes cheap to compute -- 32 reads a subspace, not 256.
#ifndef JHQ_EXIT_EVERY
#define JHQ_EXIT_EVERY 16
#endif
#ifndef JHQ_EXIT_BOUND
#define JHQ_EXIT_BOUND 0
#endif

// Two ways to spend less on the refinement stage, both of which leave the
// selection and every distance exactly as they were.
//
// JHQ_REGROUP sorts the surviving ck on their position in the inverted lists
// before their residual codes are read. The top-ck arrives ordered by
// distance, so the positions are scattered and the refinement makes ck reads
// of bpv bytes at random offsets -- 1000 scattered 1 KB reads a query at
// d=1024. In position order they run forward instead.
//
// JHQ_FUSE_REFINE moves the refinement into the scan kernel. Unfused, the scan
// writes ck positions and ck primary distances to global and the next kernel
// reads them straight back: 16 KB a query at ck=1000, plus a launch.
//
// They are independent and target costs 64x apart in size, so both are built
// alone and together.
#ifndef JHQ_REGROUP
#define JHQ_REGROUP 0
#endif
#ifndef JHQ_FUSE_REFINE
#define JHQ_FUSE_REFINE 0
#endif

// Carry the probe cursor across chunks, per thread, which is what the comment
// in the scan kernel always claimed happened. Off reproduces the old code so
// the two can be timed against each other.
// The probe cursor is on by default here. Measured at +2.5% to +148.5%
// depending on nprobe, recall identical -- results/front6/v51.log.
#ifndef JHQ_PROBE_CURSOR
#define JHQ_PROBE_CURSOR 1
#endif

// Read the primary code four subspaces at a time, out of a layout that stores
// them as 32-bit words grouped by 32 dimensions.
//
// The scan reads list_primary_t[m * N + pos]: one byte a thread, so a warp
// asks for 32 consecutive bytes and the memory system moves a 128-byte
// transaction to serve them. Three quarters of every fetch is thrown away,
// and the gather runs at 47-50% of this card's peak bandwidth against a
// roofline that says it should be near 100% -- results/block_sweep/ has the
// decomposition, gather is 53-66% of the scan.
//
// Ds = 8 and B = 8 make the primary code 1 bit a dimension, so four
// consecutive subspaces are 32 dimensions and fit one uint32. Storing those
// words grouped by dimension rather than by vector puts 32 threads on 32
// consecutive words: one 128-byte transaction, fully used. Same bytes, same
// codes, same distances -- only the arrangement moves. Needs M divisible by
// 4, which every configuration here satisfies (96, 128, 192, 384).
#ifndef JHQ_WORD_LAYOUT
#define JHQ_WORD_LAYOUT 1
#endif

// How many times larger than its minimum the selection buffer is allowed to
// be, when shared memory has the room.
//
// cap is the power of two at or above ck + BLOCK, and v45 compacts when the
// buffer cannot absorb another chunk: cnt + BLOCK > cap. At ck=1000 and
// BLOCK=1024 that makes cap 2048 and fires at cnt > 1024, while a compaction
// leaves cnt at ck = 1000 -- so twenty-five more survivors trigger the next
// one. The buffer is nominally twice what it needs and in practice has
// twenty-five slots of slack.
//
// Doubling cap to 4096 moves the trigger to cnt > 3072, which takes 2,072
// survivors instead of 25: eighty times fewer sorts, each one twice as long.
// The room is there -- 4096*8 is 32,768 B and an M=384 table is 49,152, which
// totals 81,920 of the 101,376 this card allows.
#ifndef JHQ_CAP_MULT
#define JHQ_CAP_MULT 1
#endif

struct SearchWorkspace {
    int batch_cap = 0, ck_cap = 0, k_cap = 0;

#if JHQ_STEP_TIMING
    // Per-stage timing bypasses the captured graph: events cannot be read back
    // from inside one, and the question the timing answers -- which stage owns
    // the time -- is worth losing graph launch overhead for. Enabled with
    // -DJHQ_STEP_TIMING=1; the default build is untouched.
    static constexpr int N_STEP_EVENTS = 9;
    cudaEvent_t ev_step[N_STEP_EVENTS] = {};
    cudaEvent_t ev_h2d_start = nullptr, ev_h2d_done = nullptr, ev_d2h_done = nullptr;
    int         timing_count = 0;
    double      acc_ms[N_STEP_EVENTS] = {};
#endif

    float*    h_q_pinned      = nullptr;
    float*    d_q_batch       = nullptr;
    float*    d_q_rot         = nullptr;
    float*    d_dots          = nullptr;
    jhq_lut_t* d_byte_lut     = nullptr;   // [batch_cap, M, JHQ_SPLIT_LUT]
    int*      d_probe_ids     = nullptr;
    int*      d_probe_offsets = nullptr;
    int*      d_query_total   = nullptr;
    int*      d_topck_pos     = nullptr;
    float*    d_topck_primary = nullptr;
    float*    d_lut_r         = nullptr;
    float*    d_comp_dists    = nullptr;
    int*      d_final_ids     = nullptr;
    float*    d_final_dists   = nullptr;

    // Set while a calibration pass walks the alpha grid: run the kernels
    // straight on the stream instead of capturing a graph per step. The graph
    // pays for itself over a long run at one ck; it is pure overhead when ck
    // changes every call.
    bool no_graph = false;

    cudaStream_t    stream     = nullptr;
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    // B is part of the key: the graph is fixed-width, and the launch is now
    // sized to the queries present rather than to batch_cap.
    int graph_ck = 0, graph_nprobe = 0, graph_B = 0;
};

void search_gpu(
    cublasHandle_t cublas,
    const float*   d_Pi,
    const float*   d_cent,      // [M][K][Ds] product-quantiser centroids
    const float*   d_res_c1d,   // [M][Kr] per-subspace residual codebooks
    const float*   d_centroids,
    const float*   d_cent_norms,
    const int*     d_list_offsets,
    const int*     d_list_ids,
    const uint8_t* d_list_primary_t,   // [M, N] transposed
    const uint8_t* d_list_res,
    const float*   d_list_corr,
    const float*   h_queries,
    int nq, int d, int M, int Ds, int K, int Kr,
    int nlist, int nprobe,
    int Br, int bpv,
    float alpha, int k,
    int batch_size,
    int ntotal,                         // N — needed for [M,N] index
    SearchWorkspace& ws,
    float* h_out_dists,
    int*   h_out_ids);

} // namespace jhq_gpu
