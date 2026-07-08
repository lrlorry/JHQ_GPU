#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v20/search.cuh"

namespace hblock_v20 {

// HBlock v20: v17 + d_proj=128 routing + empty-centroid norm=1e30f fix
class HBlockIndex {
public:
    struct Params {
        int K1         = 16;
        int K2         = 16;
        int K3         = 16;
        int Kr         = 16;
        int Br         = 4;
        int leaf_size  = 128;
        int ck1        = 4;
        int ck2        = 4;
        int ck3        = 4;
        int d_proj     = 128;   // routing projection dim (was 64 in v17)
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
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_;
    int K1_, K2_, K3_, ck1_, ck2_, ck3_;
    int rerank_r_, km_iters_, batch_size_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    float* d_Pi1_                = nullptr;
    float* d_route1_cents_proj_  = nullptr;
    float* d_route1_cents_full_  = nullptr;
    float* d_route1_norms_       = nullptr;

    float* d_Pi2_                = nullptr;
    float* d_route2_cents_proj_  = nullptr;
    float* d_route2_cents_full_  = nullptr;
    float* d_route2_norms_       = nullptr;

    float* d_Pi3_                = nullptr;
    float* d_route3_cents_proj_  = nullptr;
    float* d_route3_cents_full_  = nullptr;
    float* d_route3_norms_       = nullptr;

    float* d_fine_c1d_ = nullptr;

    int*     d_pair_blk_start_ = nullptr;
    int*     d_pair_blk_count_ = nullptr;
    uint8_t* d_leaf_codes_     = nullptr;
    int*     d_leaf_ids_       = nullptr;
    int*     d_leaf_sizes_     = nullptr;

    float*   d_base_vecs_      = nullptr;

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    void alloc_workspace();

    static void init_jl_proj(int d, int d_proj, int seed, std::vector<float>& Pi);

    void gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                    int n, int K,
                    std::vector<float>& h_cents_proj,
                    std::vector<float>& h_cents_full,
                    std::vector<int>&   h_assigns);

    // upload_cents: h_valid[k]=false → norm set to 1e30f (empty centroid fix)
    void upload_cents(const std::vector<float>& h_proj, const std::vector<float>& h_full,
                      const std::vector<bool>& h_valid,
                      int K,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);
};

} // namespace hblock_v20
