#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "cpu/jl_transform.h"
#include "hblock_v8/search.cuh"

namespace hblock_v8 {

// ── HBlock v8: Flink-like streaming search ────────────────────────────────────
//
// Pipeline:
//   Phase 1 (GPU)  : route_queries → d_leaf_sel + d_lut_fine on GPU
//   Phase 2 (CPU)  : D2H h_leaf_sel → fan-out to leaf_queues[leaf_id]
//   Phase 3 (GPU+CPU): dispatch loop
//       pack DISPATCH_BATCH leaves → H2D → leaf_flink_kernel (smem codes,
//       in-kernel top-p) → D2H partial results → CPU merge into QueryState heap
//   Phase 4 (CPU)  : partial-sort each QueryState heap → final top-k
//
// Key improvement over v7:
//   - No 128 MB fine_dists buffer (eliminated entirely)
//   - Leaf codes loaded once per dispatch per leaf → no HBM re-reads
//   - D2H per dispatch: n_pairs × TOP_P × 8 B ≪ 128 MB
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
    int ntotal_       = 0;
    int n_leaf_blocks_ = 0;

    JLTransform        jl_;
    std::vector<float> fine_c1d_;
    std::vector<float> h_route1_, h_route2_;

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

} // namespace hblock_v8
