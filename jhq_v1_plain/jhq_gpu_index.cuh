#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/codebook.h"
#include "cpu/jl_transform.h"
#include "jhq_v1_plain/search.cuh"

namespace jhq_gpu {

class JHQGpuIndex {
public:
    struct Params {
        int   M     = 128;   // number of primary subspaces
        int   B     = 8;     // bits per primary subspace
        int   Br    = 4;     // bits per residual dimension (4 or 8)
        float alpha = 4.0f;  // candidate oversampling for Phase 1→2
        int   seed  = 42;
    };

    JHQGpuIndex(int d, Params p);
    ~JHQGpuIndex();

    // CPU: QR + sigma + codebook + residual k-means.
    // GPU: rotate training sample + encode to get residuals for k-means.
    void train(const float* h_x, int n_train);

    // GPU: rotate + encode all vectors; append to index.
    void add(const float* h_x, int n);

    // GPU: full JHQ search (JL rotate queries, ADC scan, residual refine).
    // h_q       : [nq × d] host query vectors
    // h_dists   : [nq × k] output distances
    // h_labels  : [nq × k] output vector ids
    void search(const float* h_q, int nq, int k,
                float* h_dists, int* h_labels) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    // ── params ────────────────────────────────────────────────────────────────
    int   d_, M_, B_, Br_, Ds_, K1D_, bpv_, bits_per_dim_;
    float alpha_;
    int   Kr_;
    int   ntotal_ = 0;

    // ── CPU objects (for train) ───────────────────────────────────────────────
    JLTransform jl_;
    std::unique_ptr<LloydMaxCodebook> cb_;
    std::vector<float> res_c1d_;   // (Kr,)

    // ── Permanent GPU buffers ─────────────────────────────────────────────────
    float*   d_Pi_            = nullptr;  // [d × d]   rotation matrix (col-major)
    float*   d_c1d_           = nullptr;  // [K1D]     primary 1D codewords
    float*   d_res_c1d_       = nullptr;  // [Kr]      residual 1D codewords
    uint8_t* d_primary_codes_ = nullptr;  // [N × M]
    uint8_t* d_res_codes_     = nullptr;  // [N × bpv]
    float*   d_corrections_   = nullptr;  // [N]

    // ── Search workspace (preallocated, reused across search() calls) ─────────
    // Reallocated whenever ntotal_ grows in add().
    mutable SearchWorkspace ws_;

    mutable cublasHandle_t cublas_;

    // ── helpers ───────────────────────────────────────────────────────────────
    float* rotate_on_gpu(const float* h_x, int n) const;
    void   train_residual_codebook(const float* d_y_train,
                                   const uint8_t* d_codes_train, int n_train);
    void   alloc_workspace(long long N);  // (re)allocate ws_ for N vectors
};

} // namespace jhq_gpu
