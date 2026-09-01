#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cpu/codebook.h"      // train_1d_kmeans, for the residual level
#include "cpu/pq_codebook.h"
#include "cpu/jl_transform.h"
#include "jhq_v19_tiled_scan/search.cuh"

namespace jhq_gpu {

class JHQGpuIndex {
public:
    struct Params {
        int   M          = 96;
        int   B          = 8;   // bits per subspace; K = 2^B, B <= 8
        int   Br         = 4;
        float alpha      = 4.0f;
        int   nlist      = 1024;
        int   nprobe     = 8;
        int   ivf_iters  = 8;
        int   batch_size = 256;
        int   seed       = 42;
        // Lloyd iterations refining the analytically placed primary
        // centroids. The official IndexJHQ defaults to 5; 0 keeps the
        // purely analytical codebook, which is the training-free variant.
        int   kmeans_iters = 5;
        // add()-time streaming batch: max rows of full-precision float
        // vectors (raw + JL-rotated) held on GPU at once while encoding.
        // Unlike v12_transposed's add() (rotate_on_gpu() over the whole n
        // at once -- two full n*d float buffers, ~145GB for stella-trec24's
        // 17.8M rows), this version never materializes more than
        // add_batch rows of float vectors simultaneously. 65536 keeps that
        // well under 1GB even at d=3072, far below the ~12-14GB of
        // compressed (uint8 code) accumulators that stay resident for the
        // whole n regardless of batch size -- see jhq_gpu_index.cu's add().
        int   add_batch   = 65536;
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
    int   d_, M_, B_, Br_, Ds_, K_, bpv_;
    int   nlist_, nprobe_, ivf_iters_, batch_size_, add_batch_;
    int   kmeans_iters_, seed_;
    float alpha_;
    int   Kr_;
    int   ntotal_ = 0;

    JLTransform jl_;
    // Primary level is a product quantiser (cpu/pq_codebook.h), matching the
    // official IndexJHQ's primary_pq_. Ds = d/M is unconstrained, so the
    // primary code can be far shorter than one bit per dimension.
    std::unique_ptr<PQCodebook> cb_;
    // One scalar residual codebook per subspace ([M][Kr]), as in the
    // official get_scalar_codebook_ptr(subspace_idx, level).
    std::vector<float> res_c1d_;
    std::vector<float> centroids_;

    float*   d_Pi_             = nullptr;
    float*   d_cent_           = nullptr;  // [M][K][Ds]
    float*   d_res_c1d_        = nullptr;
    float*   d_centroids_      = nullptr;
    float*   d_cent_norms_     = nullptr;

    int*     d_list_offsets_   = nullptr;
    int*     d_list_ids_       = nullptr;
    uint8_t* d_list_primary_t_ = nullptr;  // [M, N] transposed — replaces [N, M]
    uint8_t* d_list_res_       = nullptr;
    float*   d_list_corr_      = nullptr;

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    float* rotate_on_gpu(const float* h_x, int n) const;
    void   train_residual_codebook(const float* d_y_train,
                                   const uint8_t* d_codes_train, int n_train);
    void   train_ivf_centroids(const float* h_y_train,
                               const float* d_y_train, int n_train);
    int*   assign_on_gpu(const float* d_y, int n) const;
    void   alloc_workspace(int batch_size);

    // Trained-state cache. Every run of a parameter sweep retrains an
    // identical index -- ~29 s on Vogue, against ~1 s of search -- because
    // training depends only on the sample and the seed, not on any search
    // parameter. Enabled by setting JHQ_INDEX_CACHE to a directory.
    void        upload_trained();
    std::string cache_path(const char* dir, const float* h_x, int n_train) const;
    bool        load_trained(const std::string& path);
    void        save_trained(const std::string& path) const;
};

} // namespace jhq_gpu
