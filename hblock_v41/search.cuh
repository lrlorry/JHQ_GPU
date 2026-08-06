#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>

namespace hblock_v41 {

static constexpr int K_MAX = 256;

// Rounds a byte offset up to the next multiple of 4. Shared by
// The id sub-record must be int-aligned after the code
// sub-record within a packed region entry) and jhq_gpu_index.cu's
// build_region_layout(), which must place records using the identical rule.
__host__ __device__ __forceinline__ size_t align4(size_t x) { return (x + 3u) & ~size_t(3u); }

// v41: v38 search semantics with complete block records staged through a
// bounded GPU region pool. Routing and graph metadata remain resident.
// beam_size = ef (no cap, v35/v36 semantics).
// Output: d_leaf_sel records EXPANDED blocks in expansion order (v30 semantics).
//
// Raw base-vector storage and raw query storage are int8 (SPACEV int8
// vectors); the routing/beam-search pipeline (JL projection, L1/L2/L3
// centroid routing, block graph traversal) is unchanged and stays float —
// only the exact-rerank stage reads int8 directly and
// accumulates in int32.
//
// Sorted (query, block) pairs are processed in region-fitting waves. Each
// block record contains PQ codes, ids, and raw vectors; the fused leaf kernel
// performs the same PQ top-r -> exact L2 operation as v38.
struct SearchWorkspace {
    int batch_cap    = 0;
    int max_pairs    = 0;
    int max_leaf_sel = 0;
    int d_proj       = 0;
    int ck1 = 0, ck2 = 0, ck3 = 0;
    int beam_size    = 32;
    int per_block_r  = 16;  // PQ prefilter per block before exact L2
    int klocal       = 10;  // per-block exact top-k output

    // ── Pinned host ──────────────────────────────────────────────────────────
    float*  h_q_pinned    = nullptr;  // float-cast query batch, for routing (JL proj etc.)
    int8_t* h_q_pinned_i8 = nullptr;  // raw int8 query batch, for exact rerank
    int*    h_leaf_cnt    = nullptr;
    float*  h_final_dists = nullptr;
    int*    h_final_ids   = nullptr;

    int*   h_top1_ids  = nullptr;
    int*   h_top2_beam = nullptr;
    int*   h_top3_beam = nullptr;
    int*   h_block_sel = nullptr;

    // Host mirror of the sorted d_pair_leaf_b array (pinned, [max_pairs]).
    // Copied back after gpu_build_and_sort_pairs_v29() so the index can do
    // region fetch planning (which regions must be staged into the GPU pool
    // before a region wave runs) before touching the GPU payload.
    int*   h_pair_leaf_sorted = nullptr;

    // ── GPU block beam search buffers ────────────────────────────────────────
    int*   d_visited    = nullptr;
    int    bitmap_words = 0;

    // ── GPU routing (float, unchanged pipeline) ─────────────────────────────
    float* d_q_batch   = nullptr;  // float-cast query batch (routing)
    float* d_q_proj1   = nullptr;
    float* d_dots1     = nullptr;
    int*   d_top1_ids  = nullptr;
    float* d_r1_beam   = nullptr;
    int*   d_top2_beam = nullptr;
    int*   d_top3_beam = nullptr;
    float* d_q_r3      = nullptr;
    float* d_lut_fine  = nullptr;

    // ── Raw int8 query batch (exact rerank) ─────────────────────────────────
    int8_t* d_q_batch_i8 = nullptr;

    // ── Block selection ──────────────────────────────────────────────────────
    int*   d_leaf_sel  = nullptr;
    int*   d_leaf_cnt  = nullptr;

    // ── Pair build + sort ────────────────────────────────────────────────────
    int*   d_query_offsets = nullptr;
    int*   d_pair_leaf_a   = nullptr;
    int*   d_pair_qid_a    = nullptr;
    int*   d_pair_leaf_b   = nullptr;
    int*   d_pair_qid_b    = nullptr;

    // Per-block exact L2 results [max_pairs × klocal].
    float* d_out_dists = nullptr;
    int*   d_out_ids   = nullptr;

    float* d_final_dists = nullptr;
    int*   d_final_ids   = nullptr;

    void*  d_cub_tmp = nullptr;
    size_t cub_bytes = 0;

    cudaStream_t stream = nullptr;
};

// beam = ef (no cap); output = expanded-block log (v30 semantics).
void gpu_block_search_v35(
    int B, int n_blks, int d_proj,
    int K2, int K3, int ck1, int ck2, int ck3,
    int degree, int ef, int max_ls, int entry_per_cell,
    const int*   d_block_adj,
    const float* d_blk_proj,
    const float* d_blk_norm,
    const int*   d_pair_blk_start,
    const int*   d_pair_blk_count,
    SearchWorkspace& ws);

void route_gpu_v29(
    cublasHandle_t cublas,
    const float* d_Pi1, const float* d_Pi2, const float* d_Pi3,
    const float* d_route1_cents_proj, const float* d_route1_cents_full,
    const float* d_route1_norms,
    const float* d_route2_cents_proj, const float* d_route2_cents_full,
    const float* d_route2_norms,
    const float* d_route3_cents_proj, const float* d_route3_cents_full,
    const float* d_route3_norms,
    const float* d_fine_c1d,
    const float* h_queries,
    int nq, int d, int d_proj,
    int K1, int K2, int K3, int Kr,
    int ck1, int ck2, int ck3,
    int batch_size,
    SearchWorkspace& ws);

void gpu_build_and_sort_pairs_v29(
    int nq, int n_pairs, int n_leaf_blocks,
    int max_leaf_sel, SearchWorkspace& ws);

// Staged unified block record -> PQ top-r -> exact L2 -> block top-klocal.
void launch_leaf_fused_v41(
    const int*     d_pair_leaf_ids,
    const int*     d_pair_qids,
    const uint8_t* d_region_pool,
    const int*     d_region_slot,
    const int*     d_block_region,
    const int*     d_block_offset,
    const int*     d_leaf_sizes,
    const float*   d_lut_fine,
    const int8_t*  d_q_batch_i8,
    float*         d_out_dists,
    int*           d_out_ids,
    int region_bytes,
    int n_pairs,
    int d, int Kr, int Br, int bpv, int leaf_size,
    int per_block_r, int klocal,
    cudaStream_t stream);

// Merge per-block exact results → global top-k per query
void launch_final_merge_v29(
    int nq, int n_pairs, int klocal, int k,
    SearchWorkspace& ws);

} // namespace hblock_v41
