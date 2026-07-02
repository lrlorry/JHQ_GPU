#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v4/search.cuh"

namespace hblock_v4 {

// ── HBlock v4: Discriminative projection (S_B cascade) ───────────────────────
//
// Same runtime architecture as v3, but P1/P2 are eigenvectors of S_B
// (between-class scatter matrix) rather than X^T X.
//
//   S_B = Σ_c n_c (μ_c - μ)(μ_c - μ)^T    (rank ≤ K1-1)
//
// Training: PCA init → k-means → S_B eigendecompose → new P → re-k-means.
//
// Key advantage over v2 (JL + product code):
//   - P1/P2 adapt to the actual data distribution.
//   - Centroids found via k-means (not analytical), maximizing L1/L2 purity.
//   - Routing GEMM is k1×d or k2×d (much smaller than d×d in v2).
class HBlockIndex {
public:
    struct Params {
        int   K1        = 64;
        int   K2        = 128;
        int   k1        = 32;    // S_B projection dims for L1 routing
        int   k2        = 32;    // S_B projection dims for L2 routing
        int   Kr        = 16;
        int   Br        = 4;
        int   leaf_size = 128;
        int   ck1       = 8;
        int   ck2       = 32;
        int   ck3       = 256;
        int   batch_size = 1024;
        int   kmeans_iters = 50;
        int   seed      = 42;
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

    float*   d_P1_               = nullptr;
    float*   d_P2_               = nullptr;
    float*   d_C1_proj_          = nullptr;
    float*   d_C1_proj_norms_    = nullptr;
    float*   d_C1_full_          = nullptr;
    float*   d_C2_proj_          = nullptr;
    float*   d_C2_proj_norms_    = nullptr;
    float*   d_C2_full_          = nullptr;
    float*   d_fine_c1d_         = nullptr;
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

} // namespace hblock_v4
