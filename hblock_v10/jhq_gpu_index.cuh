#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "cpu/jl_transform.h"
#include "hblock_v10/search.cuh"

namespace hblock_v10 {

// ── HBlock v10: one big kernel launch, GPU self-schedules all leaf blocks ─────
//
// v9 had 32 serial dispatch rounds (256 blocks each) → GPU at 31% utilization.
// v10 packs ALL non-empty leaf blocks into ONE kernel launch (~3200 blocks):
//   GPU sees all blocks at once → 10 waves at 100% SM utilization (like v5).
//
// Memory savings vs v5: no 128 MB fine_dists buffer.
//   v5: fine_dists = B×ck3×leaf_size×4B = 128 MB
//   v10: out_dists = n_pairs×TOP_P×4B   ≈ 4 MB
//
// Pipeline:
//   route_queries → CPU fan-out → pack all → ONE [H2D + kernel + D2H] → merge
class HBlockIndex {
public:
    struct Params {
        int K1         = 64;
        int K2         = 128;
        int Kr         = 16;
        int Br         = 4;
        int leaf_size  = 128;
        int ck1        = 8;
        int ck2        = 32;
        int ck3        = 256;
        int batch_size = 1024;
        int seed       = 42;
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
    int d_, Kr_, Br_, bpv_, leaf_size_, K1_, K2_, ck1_, ck2_, ck3_;
    int batch_size_;
    int ntotal_        = 0;
    int n_leaf_blocks_ = 0;

    JLTransform        jl_;
    std::vector<float> fine_c1d_, h_route1_, h_route2_;

    float*   d_Pi_             = nullptr;
    float*   d_route1_cents_   = nullptr;
    float*   d_route1_norms_   = nullptr;
    float*   d_route2_cents_   = nullptr;
    float*   d_route2_norms_   = nullptr;
    float*   d_fine_c1d_       = nullptr;
    int*     d_pair_blk_start_ = nullptr;
    int*     d_pair_blk_count_ = nullptr;
    uint8_t* d_leaf_codes_     = nullptr;
    int*     d_leaf_ids_       = nullptr;
    int*     d_leaf_sizes_     = nullptr;

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();
    void upload_centroids(std::vector<float>& cents,
                          float*& d_cents, float*& d_norms, int K);
};

} // namespace hblock_v10
