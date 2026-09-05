#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cpu/codebook.h"      // train_1d_kmeans, for the residual level
#include "cpu/pq_codebook.h"
#include "cpu/jl_transform.h"
#include "jhq_v27_cache_key/search.cuh"

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

    // h_res_x / n_res_train train the residual codebook alone. Everything the
    // primary level needs -- sigma, the equation 4 codebook, the IVF centroids
    // -- still comes from h_x and n_train, so a sweep over n_res_train varies
    // the residual training set and nothing else. Defaults reproduce the
    // single-sample behaviour exactly.
    void train(const float* h_x, int n_train,
               const float* h_res_x = nullptr, int n_res_train = 0);
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
    // Size of the residual training set, recorded so the trained-state cache
    // key moves with it; 0 means the same set the primary level used.
    int   res_train_n_ = 0;
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
    // Per-dimension level table of the paper's product codebook, and its size.
    // Present only when that construction was used; it is what makes the
    // separable encode possible.
    float*   d_levels_         = nullptr;
    int      n_levels_         = 0;
    float*   d_centroids_      = nullptr;
    float*   d_cent_norms_     = nullptr;
    mutable __half* d_centroids16_ = nullptr;   // fp16 copy, built on demand
    mutable int8_t*  d_centroids8_ = nullptr;   // int8 copy on one scale, built on demand
    mutable unsigned* d_cstats8_   = nullptr;   // its scale and largest error, as float bits
    // The same centroids carried back through the rotation, R^T c, with their
    // own fp16 and int8 copies: the rotation is orthogonal, so the nearest
    // centroid to Rx among the c is the nearest to x among the R^T c, and a
    // pass that only assigns need not rotate. Built on demand, dropped with
    // the others whenever the centroids change.
    mutable float*    d_centroids_x_   = nullptr;
    mutable __half*   d_centroids16_x_ = nullptr;
    mutable int8_t*   d_centroids8_x_  = nullptr;
    mutable unsigned* d_cstats8_x_     = nullptr;

    int*     d_list_offsets_   = nullptr;
    int*     d_list_ids_       = nullptr;
    uint8_t* d_list_primary_t_ = nullptr;  // [M, N] transposed — replaces [N, M]
    uint8_t* d_list_res_       = nullptr;
    float*   d_list_corr_      = nullptr;

    mutable SearchWorkspace ws_;
    mutable cublasHandle_t  cublas_;

    float* rotate_on_gpu(const float* h_x, int n, double* sum_sq = nullptr) const;
    // Streams the rotation and primary encoding a batch at a time, so the
    // device holds one batch rather than the whole set. The residuals
    // themselves are O(N*Ds) per subspace -- that is what equation 5 collects
    // and it cannot be avoided without approximating -- but they belong in
    // host memory, not in VRAM.
    void   train_residual_codebook_from(const float* all_resid, int n);
    void   train_residual_codebook_streamed(const float* h_x, int n);
    void   train_residual_codebook(const float* d_y_train,
                                   const uint8_t* d_codes_train, int n_train);
    void   train_ivf_centroids(const float* h_y_train,
                               const float* d_y_train, int n_train);
    int*   assign_on_gpu(const float* d_y, int n) const;
    // Same coarse assignment, but writing into caller-owned memory on a caller-
    // owned stream. assign_on_gpu allocates two buffers, synchronises the whole
    // device and frees on every call; inside add()'s batch loop that is a
    // malloc, a device-wide barrier and a free per batch -- 272 of them on
    // stella-trec24 -- which is what stops the loop from pipelining.
    // x_domain: d_y holds unrotated input rows and the R^T c set is used.
    void   assign_into(const float* d_y, int n, int* d_out, float* d_dots,
                       int y_transposed, cudaStream_t stream,
                       __half* d_y16 = nullptr, int x_domain = 0) const;
    int    assign_precision() const;
    bool   assign_int8() const;
    void   ensure_assign_centroids(cudaStream_t stream, int x_domain = 0) const;
    void   requantize_centroids8(cudaStream_t stream, int x_domain = 0) const;
    void   drop_assign_centroids() const;
    // True when the paper's equation 4 construction is in force; see the .cu.
    bool   paper_codebook_selected() const;
    void   sort_residual_codebook();
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
