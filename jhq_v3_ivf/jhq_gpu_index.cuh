#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/codebook.h"
#include "cpu/jl_transform.h"
#include "jhq_v3_ivf/search.cuh"

namespace jhq_gpu {

class JHQGpuIndex {
public:
    struct Params {
        int   M     = 128;
        int   B     = 8;
        int   Br    = 4;
        float alpha = 4.0f;
        int   nlist = 1024;
        int   nprobe = 8;
        int   ivf_iters = 8;
        int   seed  = 42;
    };

    JHQGpuIndex(int d, Params p);
    ~JHQGpuIndex();

    void train(const float* h_x, int n_train);
    void add  (const float* h_x, int n);
    void search(const float* h_q, int nq, int k,
                float* h_dists, int* h_labels) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int   d_, M_, B_, Br_, Ds_, K1D_, bpv_, bits_per_dim_;
    int   nlist_, nprobe_, ivf_iters_;
    float alpha_;
    int   Kr_;
    int   ntotal_ = 0;

    JLTransform jl_;
    std::unique_ptr<LloydMaxCodebook> cb_;
    std::vector<float> res_c1d_;
    std::vector<float> centroids_;     // [nlist * d], row-major

    float*   d_Pi_            = nullptr;
    float*   d_c1d_           = nullptr;
    float*   d_res_c1d_       = nullptr;
    float*   d_centroids_     = nullptr;  // [nlist * d], row-major
    float*   d_cent_norms_    = nullptr;  // [nlist]

    // IVF list storage, concatenated by list.
    int*     d_list_offsets_  = nullptr;  // [nlist + 1]
    int*     d_list_ids_      = nullptr;  // [N], original vector id
    uint8_t* d_list_primary_  = nullptr;  // [N * M]
    uint8_t* d_list_res_      = nullptr;  // [N * bpv]
    float*   d_list_corr_     = nullptr;  // [N]

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    float* rotate_on_gpu(const float* h_x, int n) const;
    void   train_residual_codebook(const float* d_y_train,
                                   const uint8_t* d_codes_train, int n_train);
    void   train_ivf_centroids(const float* h_y_train,
                               const float* d_y_train,
                               int n_train);
    int*   assign_on_gpu(const float* d_y, int n) const;
    void   alloc_workspace(long long N);
};

} // namespace jhq_gpu
