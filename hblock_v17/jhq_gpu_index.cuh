#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v17/search.cuh"

namespace hblock_v17 {

// HBlock v17: true 3-level JL routing + PQ coarse scan + exact re-rank
//
// Routing: L1(q) → L2(r1) → L3(r2) → ck1×ck2×ck3 leaf cells
// Leaf:    PQ-encode r3 = r2 - C3_full[c3]
// Rerank:  exact inner product on gathered original vectors
class HBlockIndex {
public:
    struct Params {
        int K1         = 64;
        int K2         = 64;
        int K3         = 64;    // L3 centroids
        int Kr         = 16;
        int Br         = 4;
        int leaf_size  = 128;
        int ck1        = 4;
        int ck2        = 4;
        int ck3        = 4;     // L3 selections; total leaf = ck1*ck2*ck3
        int d_proj     = 64;    // JL projection dimension (shared across all levels)
        int rerank_r   = 64;    // top-R candidates for exact re-rank
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
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int rerank_r_, km_iters_, batch_size_;
    int ntotal_        = 0;
    int n_leaf_blocks_ = 0;

    // L1 JL + k-means
    float* d_Pi1_                = nullptr;  // [d_proj, d]
    float* d_route1_cents_proj_  = nullptr;  // [K1, d_proj]
    float* d_route1_cents_full_  = nullptr;  // [K1, d]
    float* d_route1_norms_       = nullptr;  // [K1]

    // L2 JL + k-means  (on r1)
    float* d_Pi2_                = nullptr;  // [d_proj, d]
    float* d_route2_cents_proj_  = nullptr;  // [K2, d_proj]
    float* d_route2_cents_full_  = nullptr;  // [K2, d]
    float* d_route2_norms_       = nullptr;  // [K2]

    // L3 JL + k-means  (on r2)
    float* d_Pi3_                = nullptr;  // [d_proj, d]
    float* d_route3_cents_proj_  = nullptr;  // [K3, d_proj]
    float* d_route3_cents_full_  = nullptr;  // [K3, d]
    float* d_route3_norms_       = nullptr;  // [K3]

    // Fine PQ codebook (on r3)
    float* d_fine_c1d_ = nullptr;  // [Kr]

    // Leaf block structures: indexed by c1*K2*K3 + c2*K3 + c3
    int*     d_pair_blk_start_ = nullptr;  // [K1*K2*K3]
    int*     d_pair_blk_count_ = nullptr;  // [K1*K2*K3]
    uint8_t* d_leaf_codes_     = nullptr;  // [n_leaf_blocks, bpv, leaf_size]
    int*     d_leaf_ids_       = nullptr;  // [n_leaf_blocks, leaf_size]
    int*     d_leaf_sizes_     = nullptr;  // [n_leaf_blocks]

    // Original vectors for re-ranking
    float*   d_base_vecs_      = nullptr;  // [ntotal, d]

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

} // namespace hblock_v17
