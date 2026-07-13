#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <vector>

#include "hblock_v32/search.cuh"

namespace hblock_v32 {

// HBlock v32: single beam parameter.
// Global top-ef routing at each tree level (vs. per-parent local topk in v30).
// Block graph search uses the same ef as itopk.
// HNSW-style natural termination replaces fixed depth.
class HBlockIndex {
public:
    struct Params {
        int K1           = 16;
        int K2           = 16;
        int K3           = 16;
        int Kr           = 16;
        int Br           = 4;
        int leaf_size    = 128;
        int beam         = 32;   // single beam width: routing + graph search
        int d_proj       = 64;
        int per_block_r  = 16;
        int klocal       = 10;
        int km_iters     = 30;
        int batch_size   = 1024;
        int graph_degree = 32;
        int entry_per_cell = 4;
        int n_c2_nbrs    = 4;
        int n_c1_nbrs    = 2;
        int mini_km_iters = 5;
    };

    HBlockIndex(int d, Params p);
    ~HBlockIndex();

    void train (const float* h_x, int n_train);
    void add   (const float* h_x, int n);
    void search(const float* h_q, int nq, int k,
                float* h_dists, int* h_ids) const;

    double oracle_recall(const float* h_q, int nq, int k,
                         const float* h_base, const int* h_gt, int gt_k) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int d_, d_proj_, Kr_, Br_, bpv_, leaf_size_;
    int K1_, K2_, K3_;
    int per_block_r_, klocal_, km_iters_, batch_size_;
    int graph_degree_, entry_per_cell_;
    int n_c2_nbrs_, n_c1_nbrs_;
    int beam_, mini_km_iters_;
    int ntotal_           = 0;
    int n_leaf_blocks_    = 0;
    int max_blk_per_cell_ = 1;

    float* d_Pi1_ = nullptr;
    float* d_Pi2_ = nullptr;
    float* d_Pi3_ = nullptr;

    float* d_route1_cents_proj_ = nullptr;
    float* d_route1_cents_full_ = nullptr;
    float* d_route1_norms_      = nullptr;
    float* d_route2_cents_proj_ = nullptr;
    float* d_route2_cents_full_ = nullptr;
    float* d_route2_norms_      = nullptr;
    float* d_route3_cents_proj_ = nullptr;
    float* d_route3_cents_full_ = nullptr;
    float* d_route3_norms_      = nullptr;

    float* d_fine_c1d_ = nullptr;

    int*     d_pair_blk_start_ = nullptr;
    int*     d_pair_blk_count_ = nullptr;
    uint8_t* d_leaf_codes_     = nullptr;
    int*     d_leaf_ids_       = nullptr;
    int*     d_leaf_sizes_     = nullptr;

    std::vector<float> h_block_cent_;
    std::vector<float> h_block_cent_proj_;
    std::vector<float> h_block_cent_norm_;
    std::vector<float> h_Pi_blk_;
    std::vector<int>   h_block_cell_id_;
    std::vector<int>   h_block_adj_;
    std::vector<float> h_leaf_abs_cents_;
    std::vector<int>   h_pair_blk_start_cpu_;
    std::vector<int>   h_pair_blk_count_cpu_;

    std::vector<int>   h_leaf_ids_cpu_;
    std::vector<int>   h_leaf_sizes_cpu_;

    int*   d_block_adj_gpu_  = nullptr;
    float* d_blk_proj_gpu_   = nullptr;
    float* d_blk_norm_gpu_   = nullptr;

    float* d_base_vecs_ = nullptr;

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
                      const std::vector<bool>& h_valid, int K,
                      float*& d_proj_out, float*& d_full_out, float*& d_norms_out);

    void build_block_graph(const std::vector<int>& h_pair_blk_start,
                           const std::vector<int>& h_pair_blk_count);
};

} // namespace hblock_v32
