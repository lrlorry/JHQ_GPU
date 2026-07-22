#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v37_prr/search.cuh"

namespace hblock_v37_prr {

// ── Exact-seed threshold diagnostic (additive; see
//    HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md) — one row per (ef, seed_per_block).
struct SeedDiagRow {
    int ef = 0, spb = 0;
    double visited_blocks_q = 0, valid_cands_q = 0, seeds_q = 0;
    double tau_u_mean = 0, tau_seed_mean = 0, tau_ratio_mean = 0;
    double surv_block_avg = 0, surv_block_p50 = 0, surv_block_p90 = 0, surv_block_p99 = 0;
    double blocks_over16_pct = 0, nonseed_surv_q = 0, total_exact_q = 0, baseline_exact_q = 0;
    double exact_ratio_mean = 0, exact_ratio_p50 = 0, exact_ratio_p90 = 0;
    double better_than_baseline_pct = 0;
    int insufficient_seed_queries = 0;
};

// HBlock v37_prr: v36 + certified PQ racing (PRR).
// Three search modes:
//   FIXED_PER_BLOCK  — unchanged v36 LUT path
//   CORRECTED_FIXED  — per-block correct PQ using block's own cell centroid
//   CERTIFIED_PRR    — corrected PQ + epsilon bounds + L2/U2 pruning + exact rerank
class HBlockIndex {
public:
    struct Params {
        int K1           = 16;
        int K2           = 16;
        int K3           = 16;
        int Kr           = 16;
        int Br           = 4;
        int leaf_size    = 128;
        int ck1          = 2;
        int ck2          = 2;
        int ck3          = 4;
        int d_proj       = 64;
        int per_block_r  = 16;
        int klocal       = 10;
        int km_iters     = 30;
        int batch_size   = 1024;
        int graph_degree   = 32;
        int max_ef         = 256;
        int entry_per_cell = 4;
        int n_c2_nbrs      = 4;
        int n_c1_nbrs      = 2;
        int max_cand_blocks = 2048;
        int mini_km_iters  = 5;

        // v37_prr: search mode and epsilon granularity
        enum SearchMode { FIXED_PER_BLOCK = 0, CORRECTED_FIXED = 1, CERTIFIED_PRR = 2 };
        enum EpsilonMode { BLOCK_128 = 0, SUBBLOCK_32 = 1, VECTOR_LEVEL = 2 };
        SearchMode search_mode = FIXED_PER_BLOCK;
        EpsilonMode eps_mode   = BLOCK_128;
    };

    HBlockIndex(int d, Params p);
    ~HBlockIndex();

    void train (const float* h_x, int n_train);
    void add   (const float* h_x, int n);

    void search(const float* h_q, int nq, int k,
                float* h_dists, int* h_ids, int ef = 64) const;

    double oracle_recall(const float* h_q, int nq, int k,
                         const float* h_base, const int* h_gt, int gt_k) const;

    void diagnose_missed_gt(const float* h_q, int nq, int k,
                            const int* h_gt, int gt_k) const;

    // Flip the search mode on an already-built index (no rebuild, no workspace
    // realloc). PRR workspace buffers were allocated at build time iff the index
    // was built with CERTIFIED_PRR; switching to FIXED_PER_BLOCK/CORRECTED_FIXED
    // simply doesn't use them, and switching back to CERTIFIED_PRR is safe only
    // if the index was originally built with that mode (see seed_diagnostic()).
    void set_search_mode(Params::SearchMode m) { p_.search_mode = m; }
    Params::SearchMode search_mode() const { return p_.search_mode; }

    // ── Exact-seed threshold diagnostic ─────────────────────────────────────
    // Reuses the current Tree+Graph visited blocks and corrected candidate-cell
    // PQ distance unchanged (index must be built with CERTIFIED_PRR so the PRR
    // workspace buffers exist). Does not alter FIXED_PER_BLOCK/CORRECTED_FIXED/
    // CERTIFIED_PRR result paths — purely additive. See
    // HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md for the full spec.
    //
    // For seed_per_block in {1,2,4,8} (all as PREFIXES of one max_seed_per_block=8
    // seed list, sorted ascending by U2 — guarantees identical visited blocks and
    // bounds across the sweep), appends one SeedDiagRow to out_rows and one
    // agreement fraction to out_agreement[0..3].
    //
    // validate_nq (>=100 expected): number of leading queries on which phase-4
    // exact candidate-set validation runs (CPU exact-scan of the visited
    // candidate set vs. the two-wave union, top-k IDs compared with reordering
    // allowed and a 1e-4 relative tie tolerance at the k-th boundary distance).
    // h_base is the full host base-vector array (n x d_) used only for that
    // validation and for ground truth is never used to pick seeds/thresholds.
    void seed_diagnostic(const float* h_q, int nq, int k, int ef,
                         std::vector<SeedDiagRow>& out_rows,
                         int validate_nq,
                         const float* h_base,
                         double* out_agreement) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int per_block_r_, klocal_, km_iters_, batch_size_;
    int graph_degree_, max_ef_, entry_per_cell_;
    int n_c2_nbrs_, n_c1_nbrs_, max_cand_blocks_;
    int mini_km_iters_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    // Store full params for search mode / eps mode access
    Params p_;

    float* d_Pi1_ = nullptr;
    float* d_Pi2_ = nullptr;
    float* d_Pi3_ = nullptr;

    float* d_route1_cents_proj_ = nullptr;
    float* d_route1_cents_full_ = nullptr;
    float* d_route1_norms_      = nullptr;
    float* d_route2_cents_proj_ = nullptr;
    float* d_route2_cents_full_ = nullptr;
    float* d_route2_norms_      = nullptr;
    float* d_route3_cents_proj_ = nullptr;
    float* d_route3_cents_full_ = nullptr;
    float* d_route3_norms_      = nullptr;

    float* d_fine_c1d_ = nullptr;

    int*     d_pair_blk_start_ = nullptr;
    int*     d_pair_blk_count_ = nullptr;
    uint8_t* d_leaf_codes_     = nullptr;
    int*     d_leaf_ids_       = nullptr;
    int*     d_leaf_sizes_     = nullptr;

    std::vector<float> h_block_cent_;
    std::vector<float> h_block_cent_proj_;
    std::vector<float> h_block_cent_norm_;
    std::vector<float> h_Pi_blk_;
    std::vector<int>   h_block_cell_id_;
    std::vector<int>   h_block_adj_;
    std::vector<float> h_leaf_abs_cents_;
    std::vector<int>   h_pair_blk_start_cpu_;
    std::vector<int>   h_pair_blk_count_cpu_;

    std::vector<int>   h_leaf_ids_cpu_;
    std::vector<int>   h_leaf_sizes_cpu_;

    int*   d_block_adj_gpu_  = nullptr;
    float* d_blk_proj_gpu_   = nullptr;
    float* d_blk_norm_gpu_   = nullptr;

    float* d_base_vecs_ = nullptr;

    // ── v37_prr: PRR index data (built during add()) ────────────────────────
    int*   d_block_cell_id_prr_ = nullptr;  // [total_blocks] block→cell id
    float* d_abs_cents_prr_     = nullptr;  // [n_cells * d]  C1+C2+C3 per cell
    float* d_block_eps_blk_     = nullptr;  // [total_blocks]        max eps per block
    float* d_block_eps_sub_     = nullptr;  // [total_blocks * 4]    max eps per 32-vec subgroup
    float* d_block_eps_vec_     = nullptr;  // [total_blocks * leaf_size] per-vector eps

    // ── v37_prr: PRR search workspace (allocated in alloc_workspace) ─────────
    float* d_prr_l2_     = nullptr;  // [max_pairs * leaf_size]
    float* d_prr_u2topk_ = nullptr;  // [max_pairs * klocal]
    float* d_prr_tau2_   = nullptr;  // [batch_size]
    int*   d_prr_perm_   = nullptr;  // [max_pairs] qid-major pair permutation
    unsigned long long* d_prr_diag_ = nullptr;  // [3] {survivors, blocks>r, blocks}

    // ── v37_prr: host codewords for epsilon computation ───────────────────────
    std::vector<float> h_fine_c1d_;  // [Kr] downloaded once during add()

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();

    static void init_jl_proj(int d, int d_proj, int seed, std::vector<float>& Pi);

    void gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                    int n, int K,
                    std::vector<float>& h_cents_proj,
                    std::vector<float>& h_cents_full,
                    std::vector<int>&   h_assigns);

    void upload_cents(const std::vector<float>& h_proj, const std::vector<float>& h_full,
                      const std::vector<bool>& h_valid, int K,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);

    void build_block_graph(const std::vector<int>& h_pair_blk_start,
                           const std::vector<int>& h_pair_blk_count);
};

} // namespace hblock_v37_prr
