#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v22/search.cuh"

namespace hblock_v22 {

// HBlock v22: identical to v20 EXCEPT the gather_leaf_blocks step.
//
// v20: selected L3 cells → scan ALL blocks in each cell
// v22: selected L3 cells → entry block per cell → block-level graph traversal
//      → scan only graph_budget visited blocks
//
// The graph nodes are leaf BLOCKS (~78K for 10M dataset), not cells (4096).
// Each GPU cache-line access fetches exactly one block (128 vectors, 128 bytes).
// graph_budget directly controls the number of GPU cache-line accesses per query.
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
        int graph_degree = 32;   // k-NN degree of the block graph
        int graph_budget = 32;   // max blocks visited per query (= GPU cache-line budget)
    };

    HBlockIndex(int d, Params p);
    ~HBlockIndex();

    void train (const float* h_x, int n_train);
    void add   (const float* h_x, int n);
    void search(const float* h_q, int nq, int k,
                float* h_dists, int* h_ids) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int rerank_r_, km_iters_, batch_size_;
    int graph_degree_, graph_budget_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    // ── JL projection matrices (GPU) ─────────────────────────────────────────
    float* d_Pi1_ = nullptr;
    float* d_Pi2_ = nullptr;
    float* d_Pi3_ = nullptr;

    // ── L1 centroids (GPU) ───────────────────────────────────────────────────
    float* d_route1_cents_proj_ = nullptr;
    float* d_route1_cents_full_ = nullptr;
    float* d_route1_norms_      = nullptr;

    // ── L2 centroids (GPU) ───────────────────────────────────────────────────
    float* d_route2_cents_proj_ = nullptr;
    float* d_route2_cents_full_ = nullptr;
    float* d_route2_norms_      = nullptr;

    // ── L3 centroids (GPU) ───────────────────────────────────────────────────
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

    // ── CPU-side block graph ─────────────────────────────────────────────────
    // For each block: which cell it belongs to, and its k-NN neighbors in block space
    std::vector<int>   h_block_cell_id_;     // [n_leaf_blocks]
    std::vector<int>   h_block_adj_;         // [n_leaf_blocks × graph_degree]
    // CPU abs centroids of leaf cells (for graph traversal distance computation)
    std::vector<float> h_leaf_abs_cents_;    // [K1*K2*K3, d]
    // CPU copies of block start/count (for entry-block lookup)
    std::vector<int>   h_pair_blk_start_cpu_;
    std::vector<int>   h_pair_blk_count_cpu_;

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

    // Build block-level k-NN graph using cell-level GEMM then block-level edges
    void build_block_graph(const std::vector<int>& h_pair_blk_start,
                           const std::vector<int>& h_pair_blk_count);

    // CPU priority-queue traversal over block graph (called from search())
    void cpu_block_traverse(const float* h_queries, int nq) const;
};

} // namespace hblock_v22
