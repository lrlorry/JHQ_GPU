#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v17/search.cuh"

namespace hblock_v17 {

// HBlock v17: 3-level PCA routing + PQ coarse scan + exact re-rank
//
// Key improvements over v14:
//  1. PCA-based routing (64D) instead of random JL (6D) → much higher recall
//  2. k-means centroids in 64D PCA space → better cluster quality
//  3. ck3=ck1*ck2=16 leaf blocks (vs 256 in v14) → 16× less HBM traffic
//  4. Store original vectors (d_base_vecs_) for exact re-ranking
//  5. Re-rank top-R=64 PQ candidates with exact inner products → recall > 0.9
//  6. Per-phase timing for every search step
class HBlockIndex {
public:
    struct Params {
        int K1         = 64;
        int K2         = 128;
        int Kr         = 16;
        int Br         = 4;
        int leaf_size  = 128;
        int ck1        = 4;
        int ck2        = 4;
        int ck3        = 16;    // total leaf blocks selected = ck1 * ck2
        int d_proj     = 64;    // PCA projection dimension
        int rerank_r   = 64;    // top-R candidates for exact re-rank
        int km_iters   = 30;    // k-means iterations
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
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_, K1_, K2_, ck1_, ck2_, ck3_;
    int rerank_r_, km_iters_, batch_size_;
    int ntotal_        = 0;
    int n_leaf_blocks_ = 0;

    // L1 PCA + k-means centroids
    float* d_Pi1_                = nullptr;  // [d_proj, d] row-major
    float* d_route1_cents_proj_  = nullptr;  // [K1, d_proj]
    float* d_route1_cents_full_  = nullptr;  // [K1, d]
    float* d_route1_norms_       = nullptr;  // [K1]  ||c||^2 in d_proj space

    // L2 PCA + k-means centroids
    float* d_Pi2_                = nullptr;  // [d_proj, d]
    float* d_route2_cents_proj_  = nullptr;  // [K2, d_proj]
    float* d_route2_cents_full_  = nullptr;  // [K2, d]
    float* d_route2_norms_       = nullptr;  // [K2]

    // Fine PQ codebook
    float* d_fine_c1d_ = nullptr;  // [Kr]

    // Leaf block structures
    int*     d_pair_blk_start_ = nullptr;  // [K1*K2]
    int*     d_pair_blk_count_ = nullptr;  // [K1*K2]
    uint8_t* d_leaf_codes_     = nullptr;  // [n_leaf_blocks, bpv, leaf_size]
    int*     d_leaf_ids_       = nullptr;  // [n_leaf_blocks, leaf_size]
    int*     d_leaf_sizes_     = nullptr;  // [n_leaf_blocks]

    // Original vectors for re-ranking (stored in HBM, 6.5GB for arxiv)
    float*   d_base_vecs_      = nullptr;  // [ntotal, d]

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();

    // CPU PCA: computes top-d_proj eigenvectors of training data covariance.
    // Returns Pi [d_proj, d] (row-major), eigenvalues [d_proj].
    static void compute_pca(const float* h_x, int n, int d, int d_proj,
                             std::vector<float>& Pi, std::vector<float>& eigvals);

    // GPU k-means in d_proj-D.
    // h_x_proj: [n, d_proj], h_x_full: [n, d]
    // Returns h_cents_proj [K, d_proj], h_cents_full [K, d], h_assigns [n]
    void gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                    int n, int K,
                    std::vector<float>& h_cents_proj,
                    std::vector<float>& h_cents_full,
                    std::vector<int>&   h_assigns);

    // Upload centroid pair (proj + full) to GPU
    void upload_cents(const std::vector<float>& h_proj, const std::vector<float>& h_full,
                      int K,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);

    // Compute fine PQ codebook on residuals h_r2 [n, d]
    static std::vector<float> compute_fine_c1d(const float* h_r2, int n, int d, int Kr);
};

} // namespace hblock_v17
