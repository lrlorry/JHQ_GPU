#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v18/search.cuh"

namespace hblock_v18 {

// HBlock v18: 3-level hierarchical JL routing + JL coarse scan + exact L2 rerank.
// Replaces v17's PQ coarse scan with JL projected distances for higher recall accuracy.
//
// Leaf storage: d_leaf_proj_vecs_[lb, d_proj, leaf_size] — JL projections via Pi1.
// d_q_proj1 computed during L1 routing is reused as the leaf JL query vector.
class HBlockIndex {
public:
    struct Params {
        int K1         = 16;
        int K2         = 16;
        int K3         = 16;
        int leaf_size  = 128;
        int ck1        = 4;
        int ck2        = 4;
        int ck3        = 4;
        int d_proj     = 64;
        int rerank_r   = 64;
        int km_iters   = 30;
        int batch_size = 1024;
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
    int d_, d_proj_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int rerank_r_, km_iters_, batch_size_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    // L1: global JL + k-means
    float* d_Pi1_                = nullptr;  // [d_proj, d]
    float* d_route1_cents_proj_  = nullptr;  // [K1, d_proj]
    float* d_route1_cents_full_  = nullptr;  // [K1, d]
    float* d_route1_norms_       = nullptr;  // [K1]

    // L2: per-c1 local JL + k-means
    float* d_Pi2_                = nullptr;  // [d_proj, d]
    float* d_route2_cents_proj_  = nullptr;  // [K1*K2, d_proj]
    float* d_route2_cents_full_  = nullptr;  // [K1*K2, d]
    float* d_route2_norms_       = nullptr;  // [K1*K2]

    // L3: per-(c1,c2) local JL + k-means
    float* d_Pi3_                = nullptr;  // [d_proj, d]
    float* d_route3_cents_proj_  = nullptr;  // [K1*K2*K3, d_proj]
    float* d_route3_cents_full_  = nullptr;  // [K1*K2*K3, d]
    float* d_route3_norms_       = nullptr;  // [K1*K2*K3]

    // Leaf block index
    int*   d_pair_blk_start_ = nullptr;     // [K1*K2*K3]
    int*   d_pair_blk_count_ = nullptr;     // [K1*K2*K3]
    float* d_leaf_proj_vecs_ = nullptr;     // [n_lb, d_proj, leaf_size] JL projections
    int*   d_leaf_ids_       = nullptr;     // [n_lb, leaf_size] original vector IDs
    int*   d_leaf_sizes_     = nullptr;     // [n_lb]

    // Original vectors for re-ranking
    float* d_base_vecs_      = nullptr;     // [ntotal, d]

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
                      int K,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);
};

} // namespace hblock_v18
