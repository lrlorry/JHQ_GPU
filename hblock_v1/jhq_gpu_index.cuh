#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/jl_transform.h"
#include "hblock_v1/search.cuh"

namespace hblock {

// ── HBlock hierarchical block index ──────────────────────────────────────────
//
// Architecture (2 routing levels + leaf):
//
//   L1 routing  (K1 centroids, GEMM-based) → identifies coarse group
//   └─ L2 routing  (K2 centroids, GEMM-based) → identifies sub-group
//         └─ Leaf block  (leaf_size=128 vectors × bpv fine-residual bytes)
//
// Routing uses 2-level flat IVF (same GEMM mechanism as v12's IVF).
// Fine code: scalar PQ on the L1+L2 residual (Kr=16, Br=4 bits/dim → 384 B/vec).
// Data is organized physically: vectors sorted by (L1_code, L2_code), then grouped
// into leaf_size-vector blocks. Routing table maps (L1, L2) → leaf block range.
//
// Designed for massive datasets where leaf blocks are fetched from storage on demand.
// On Vogue-768 all data stays in GPU memory; the architecture is validated here.
class HBlockIndex {
public:
    struct Params {
        int   K1        = 64;    // L1 routing cells (must be power of 2; uses log2(K1) JL dims)
        int   K2        = 128;   // L2 routing cells (must be power of 2; uses log2(K2) JL dims)
        int   Kr        = 16;    // fine residual scalar PQ levels (Lloyd-Max, analytical)
        int   Br        = 4;     // fine bits per dimension (bpv = d*Br/8 = 384 for d=768)
        int   leaf_size = 128;   // vectors per leaf block (= SM fine-compute block size)
        int   ck1       = 8;     // L1 probes
        int   ck2       = 32;    // L2 probes
        int   ck3       = 256;   // leaf blocks probed per query (ck3 × 128 = candidates)
        int   batch_size = 1024; // queries per GPU batch (sized for 8 GB GPU)
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
    int d_, Kr_, Br_, bpv_, leaf_size_, K1_, K2_, ck1_, ck2_, ck3_;
    int batch_size_;
    int ntotal_ = 0;
    int n_leaf_blocks_ = 0;

    JLTransform jl_;
    std::vector<float> fine_c1d_;   // [Kr] sorted fine centroids
    std::vector<float> h_route1_;   // [K1, d] L1 routing centroids (host copy)
    std::vector<float> h_route2_;   // [K2, d] L2 routing centroids (host copy)

    // GPU index data
    float*   d_Pi_              = nullptr;
    float*   d_route1_cents_    = nullptr;  // [K1, d]
    float*   d_route1_norms_    = nullptr;  // [K1]
    float*   d_route2_cents_    = nullptr;  // [K2, d]
    float*   d_route2_norms_    = nullptr;  // [K2]
    float*   d_fine_c1d_        = nullptr;  // [Kr]
    int*     d_pair_blk_start_  = nullptr;  // [K1 * K2]
    int*     d_pair_blk_count_  = nullptr;  // [K1 * K2]
    uint8_t* d_leaf_codes_      = nullptr;  // [n_leaf_blocks, leaf_size, bpv]
    int*     d_leaf_ids_        = nullptr;  // [n_leaf_blocks, leaf_size]
    int*     d_leaf_sizes_      = nullptr;  // [n_leaf_blocks]

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    // Internal helpers
    void alloc_workspace();
    void upload_centroids(std::vector<float>& cents,
                          float*& d_cents, float*& d_norms, int K);
};

} // namespace hblock
