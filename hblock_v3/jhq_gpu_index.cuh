#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v3/search.cuh"

namespace hblock_v3 {

// ── HBlock v3: PCA cascade routing ───────────────────────────────────────────
//
// Architecture (same as v2 but routing uses learned PCA projections):
//
//   L1 routing: project q to k1-dim subspace via P1 (top k1 eigenvectors of X^T X),
//               find nearest centroid in k1-dim space (k-means, K1 centroids).
//   L2 routing: project L1 residual to k2-dim subspace via P2 (from residual X^T X),
//               find nearest centroid in k2-dim space (k-means, K2 centroids).
//   Leaf: same fine PQ as v2 (Kr=16 scalar levels, Br=4 bits/dim).
//
// Key advantage over v2 (JL + analytical product code):
//   - P1/P2 adapt to actual data distribution (maximum variance directions).
//   - K-means centroids are data-driven, not analytical.
//   - Routing GEMM is k1×d not d×d (smaller, but still followed by K1×k1 dot).
class HBlockIndex {
public:
    struct Params {
        int   K1         = 64;
        int   K2         = 128;
        int   k1         = 32;
        int   k2         = 32;
        int   Kr         = 16;
        int   Br         = 4;
        int   leaf_size  = 128;
        int   ck1        = 8;
        int   ck2        = 32;
        int   ck3        = 256;
        int   batch_size = 1024;
        int   kmeans_iters = 50;
        int   seed       = 42;
    };

    HBlockIndex(int d, Params p);
    ~HBlockIndex();

    void train(const float* h_x, int n_train);
    void add  (const float* h_x, int n);
    void search(const float* h_q, int nq, int k,
                float* h_dists, int* h_ids) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int d_, k1_, k2_, K1_, K2_, Kr_, Br_, bpv_, leaf_size_;
    int ck1_, ck2_, ck3_, batch_size_, kmeans_iters_, seed_;
    int ntotal_ = 0;
    int n_leaf_blocks_ = 0;

    std::vector<float> fine_c1d_;

    float*   d_P1_               = nullptr;  // [k1, d]
    float*   d_P2_               = nullptr;  // [k2, d]
    float*   d_C1_proj_          = nullptr;  // [K1, k1]
    float*   d_C1_proj_norms_    = nullptr;  // [K1]
    float*   d_C1_full_          = nullptr;  // [K1, d]
    float*   d_C2_proj_          = nullptr;  // [K2, k2]
    float*   d_C2_proj_norms_    = nullptr;  // [K2]
    float*   d_C2_full_          = nullptr;  // [K2, d]
    float*   d_fine_c1d_         = nullptr;  // [Kr]
    int*     d_pair_blk_start_   = nullptr;
    int*     d_pair_blk_count_   = nullptr;
    uint8_t* d_leaf_codes_       = nullptr;
    int*     d_leaf_ids_         = nullptr;
    int*     d_leaf_sizes_       = nullptr;

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();
    void upload_proj_centroids(const std::vector<float>& C_proj,
                               const std::vector<float>& C_full,
                               int K, int k,
                               float*& d_Cp, float*& d_Cn, float*& d_Cf);
};

} // namespace hblock_v3
