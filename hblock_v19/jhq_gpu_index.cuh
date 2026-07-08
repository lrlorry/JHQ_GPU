#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v19/search.cuh"

namespace hblock_v19 {

// HBlock v19: 3-level hierarchical routing + JL coarse scan + exact L2 rerank.
//
// Two independent JL projection dimensions:
//   d_proj_route — used for L1/L2/L3 routing (higher = better routing accuracy)
//   d_proj_scan  — used for leaf JL scan (controls bandwidth)
//
// Empty-centroid fix: L2/L3 cells with zero training vectors get norm=1e30f
// so they are never selected during routing.
class HBlockIndex {
public:
    struct Params {
        int K1           = 16;
        int K2           = 16;
        int K3           = 16;
        int leaf_size    = 128;
        int ck1          = 4;
        int ck2          = 4;
        int ck3          = 4;
        int d_proj_route = 128;  // routing projection dimension
        int d_proj_scan  = 64;   // leaf JL scan projection dimension
        int rerank_r     = 192;
        int km_iters     = 30;
        int batch_size   = 1024;
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
    int d_, d_proj_route_, d_proj_scan_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int rerank_r_, km_iters_, batch_size_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    // Routing projections (d_proj_route dimensional)
    float* d_Pi_route1_           = nullptr;  // [d_proj_route, d]
    float* d_Pi_route2_           = nullptr;  // [d_proj_route, d]
    float* d_Pi_route3_           = nullptr;  // [d_proj_route, d]
    float* d_route1_cents_proj_   = nullptr;  // [K1, d_proj_route]
    float* d_route1_cents_full_   = nullptr;  // [K1, d]
    float* d_route1_norms_        = nullptr;  // [K1]
    float* d_route2_cents_proj_   = nullptr;  // [K1*K2, d_proj_route]
    float* d_route2_cents_full_   = nullptr;  // [K1*K2, d]
    float* d_route2_norms_        = nullptr;  // [K1*K2]
    float* d_route3_cents_proj_   = nullptr;  // [K1*K2*K3, d_proj_route]
    float* d_route3_cents_full_   = nullptr;  // [K1*K2*K3, d]
    float* d_route3_norms_        = nullptr;  // [K1*K2*K3]

    // Leaf scan projection (d_proj_scan dimensional)
    float* d_Pi_scan_             = nullptr;  // [d_proj_scan, d]

    // Leaf block index
    int*   d_pair_blk_start_ = nullptr;      // [K1*K2*K3]
    int*   d_pair_blk_count_ = nullptr;      // [K1*K2*K3]
    float* d_leaf_proj_vecs_ = nullptr;      // [n_lb, d_proj_scan, leaf_size]
    int*   d_leaf_ids_       = nullptr;      // [n_lb, leaf_size]
    int*   d_leaf_sizes_     = nullptr;      // [n_lb]

    float* d_base_vecs_      = nullptr;      // [ntotal, d]

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();
    static void init_jl_proj(int d, int proj_dim, int seed, std::vector<float>& Pi);

    void gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                    int n, int K, int proj_dim,
                    std::vector<float>& h_cents_proj,
                    std::vector<float>& h_cents_full,
                    std::vector<int>&   h_assigns);

    // h_valid[k] = false → that centroid is zero-padded (empty cell); set norm = 1e30f
    void upload_cents(const std::vector<float>& h_proj,
                      const std::vector<float>& h_full,
                      const std::vector<bool>&  h_valid,
                      int K, int proj_dim,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);
};

} // namespace hblock_v19
