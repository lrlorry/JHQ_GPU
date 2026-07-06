#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "cpu/jl_transform.h"
#include "hblock_v11/search.cuh"

namespace hblock_v11 {

// ── HBlock v11: one block per pair, no smem for codes, 100% occupancy ─────────
//
// v10: one block per unique leaf, 50KB smem for codes → 3 blocks/SM → 12 warps/SM
//      Low occupancy → HBM/L2 latency exposed → 90ms kernel
//
// v11: one block per (query,leaf) pair, ~1KB smem (reduction only)
//      → 16 blocks/SM (SM120) → 64 warps/SM → 100% occupancy
//      → latency hidden → faster despite reading codes from HBM/L2
//      Pairs sorted by leaf_id → consecutive blocks share L2-cached codes
//
// Memory savings vs v5: no 128MB fine_dists. Output: n_pairs × TOP_P × 8B ≈ 8MB
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

} // namespace hblock_v11
