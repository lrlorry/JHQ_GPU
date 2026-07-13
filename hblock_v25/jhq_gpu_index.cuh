#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v25/search.cuh"

namespace hblock_v25 {

// HBlock v25: v24 + semantic block construction via mini k-means within each L3 cell.
//
// v24 vs v25:
//   v24: blocks = sequential 128-vector chunks within L3 cell (not semantic)
//   v25: within each L3 cell, run mini k-means (k=ceil(N/leaf_size)) to form
//        local clusters before packing blocks — block centroids become meaningful,
//        graph edges become reliable, beam can navigate to true neighbors
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
        int rerank_r     = 128;
        int km_iters     = 30;
        int batch_size   = 1024;
        int graph_degree   = 32;
        int graph_budget   = 32;
        int entry_per_cell = 4;    // top-k entry blocks per selected L3 cell
        int n_c2_nbrs      = 4;    // c2 sibling groups for candidate cell generation
        int n_c1_nbrs      = 2;    // c1 sibling groups for candidate cell generation
        int max_cand_blocks = 2048; // cap on candidate blocks per node before edge sort
        int top_p          = 4;    // PQ candidates kept per visited block (runtime sweep)
        int mini_km_iters  = 5;   // k-means iters within each L3 cell for semantic blocks (0=off)
    };

    HBlockIndex(int d, Params p);
    ~HBlockIndex();

    void train (const float* h_x, int n_train);
    void add   (const float* h_x, int n);
    void search(const float* h_q, int nq, int k,
                float* h_dists, int* h_ids) const;

    // Oracle: exact scan of all visited blocks — separates graph quality from PQ truncation.
    // Returns recall@k. h_base must be the full base set (n × d, host).
    double oracle_recall(const float* h_q, int nq, int k,
                         const float* h_base, const int* h_gt, int gt_k) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int rerank_r_, km_iters_, batch_size_;
    int graph_degree_, graph_budget_, entry_per_cell_;
    int n_c2_nbrs_, n_c1_nbrs_, max_cand_blocks_;
    int top_p_;
    int mini_km_iters_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    // ── JL projection matrices (GPU) ─────────────────────────────────────────
    float* d_Pi1_ = nullptr;
    float* d_Pi2_ = nullptr;
    float* d_Pi3_ = nullptr;

    // ── L1/L2/L3 centroids (GPU) ─────────────────────────────────────────────
    float* d_route1_cents_proj_ = nullptr;
    float* d_route1_cents_full_ = nullptr;
    float* d_route1_norms_      = nullptr;
    float* d_route2_cents_proj_ = nullptr;
    float* d_route2_cents_full_ = nullptr;
    float* d_route2_norms_      = nullptr;
    float* d_route3_cents_proj_ = nullptr;
    float* d_route3_cents_full_ = nullptr;
    float* d_route3_norms_      = nullptr;

    // ── Fine PQ codebook ─────────────────────────────────────────────────────
    float* d_fine_c1d_ = nullptr;

    // ── Leaf block storage (GPU) ──────────────────────────────────────────────
    int*     d_pair_blk_start_ = nullptr;
    int*     d_pair_blk_count_ = nullptr;
    uint8_t* d_leaf_codes_     = nullptr;
    int*     d_leaf_ids_       = nullptr;
    int*     d_leaf_sizes_     = nullptr;

    // ── CPU block graph (build time) ─────────────────────────────────────────
    std::vector<float> h_block_cent_;          // [n_blocks, d]
    std::vector<float> h_block_cent_proj_;     // [n_blocks, d_proj]
    std::vector<float> h_block_cent_norm_;     // [n_blocks]
    std::vector<float> h_Pi_blk_;             // [d_proj, d]  CPU Pi1
    std::vector<int>   h_block_cell_id_;
    std::vector<int>   h_block_adj_;           // [n_blocks × degree]
    std::vector<float> h_leaf_abs_cents_;
    std::vector<int>   h_pair_blk_start_cpu_;
    std::vector<int>   h_pair_blk_count_cpu_;

    // ── CPU leaf id table (kept for oracle_recall) ────────────────────────────
    std::vector<int>   h_leaf_ids_cpu_;        // [n_blocks × leaf_size]
    std::vector<int>   h_leaf_sizes_cpu_;      // [n_blocks]

    // ── GPU block graph (search time) ────────────────────────────────────────
    int*   d_block_adj_gpu_  = nullptr;        // [n_blocks × degree]
    float* d_blk_proj_gpu_   = nullptr;        // [n_blocks × d_proj]
    float* d_blk_norm_gpu_   = nullptr;        // [n_blocks]

    // ── Base vectors for exact rerank (GPU) ──────────────────────────────────
    float* d_base_vecs_ = nullptr;

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

    // Build block graph using hierarchical candidate cells + block centroid edges
    void build_block_graph(const std::vector<int>& h_pair_blk_start,
                           const std::vector<int>& h_pair_blk_count);

};

} // namespace hblock_v25
