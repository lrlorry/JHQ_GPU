#include "v2_topk/jhq_gpu_index.cuh"
#include "v2_topk/encode.cuh"
#include "v2_topk/search.cuh"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace jhq_gpu {

JHQGpuIndex::JHQGpuIndex(int d, Params p)
    : d_(d), M_(p.M), B_(p.B), Br_(p.Br), alpha_(p.alpha),
      jl_(d, p.seed)
{
    if (d % p.M != 0)
        throw std::invalid_argument("d must be divisible by M");
    if (p.B % (d / p.M) != 0)
        throw std::invalid_argument("B must be divisible by Ds = d/M");
    if (p.B > 8)
        throw std::invalid_argument("B must be <= 8");
    if (p.Br != 4 && p.Br != 8)
        throw std::invalid_argument("Br must be 4 or 8");

    Ds_           = d_ / M_;
    bits_per_dim_ = B_ / Ds_;
    K1D_          = 1 << bits_per_dim_;
    Kr_           = 1 << Br_;
    bpv_          = (d_ * Br_ + 7) / 8;

    CUBLAS_CHECK(cublasCreate(&cublas_));
}

JHQGpuIndex::~JHQGpuIndex() {
    cublasDestroy(cublas_);
    cudaFree(d_Pi_);
    cudaFree(d_c1d_);
    cudaFree(d_res_c1d_);
    cudaFree(d_primary_codes_);
    cudaFree(d_res_codes_);
    cudaFree(d_corrections_);
    cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_lut);
    cudaFree(ws_.d_dists);
    cudaFree(ws_.d_indices);
    cudaFree(ws_.d_lut_r);
    cudaFree(ws_.d_comp_dists);
    cudaFree(ws_.d_packed);
}

void JHQGpuIndex::alloc_workspace(long long N) {
    cudaFree(ws_.d_q_rot);      ws_.d_q_rot      = nullptr;
    cudaFree(ws_.d_lut);        ws_.d_lut         = nullptr;
    cudaFree(ws_.d_dists);      ws_.d_dists       = nullptr;
    cudaFree(ws_.d_indices);    ws_.d_indices     = nullptr;
    cudaFree(ws_.d_lut_r);      ws_.d_lut_r       = nullptr;
    cudaFree(ws_.d_comp_dists); ws_.d_comp_dists  = nullptr;
    cudaFree(ws_.d_packed);     ws_.d_packed      = nullptr;

    CUDA_CHECK(cudaMalloc(&ws_.d_q_rot,      (long long)d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut,        (long long)M_ * Ds_ * K1D_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dists,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_indices,    N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_r,      (long long)d_ * Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_comp_dists, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_packed,     N * sizeof(uint64_t)));
}

float* JHQGpuIndex::rotate_on_gpu(const float* h_x, int n) const {
    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, (long long)n * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (long long)n * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, (long long)n * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSgemm(cublas_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             d_, n, d_,
                             &one, d_Pi_, d_,
                                   d_x,  d_,
                             &zero, d_y, d_));
    cudaFree(d_x);
    return d_y;
}

void JHQGpuIndex::train_residual_codebook(
    const float* d_y_train, const uint8_t* d_codes_train, int n_train)
{
    std::vector<float>   h_y    ((long long)n_train * d_);
    std::vector<uint8_t> h_codes((long long)n_train * M_);

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y_train,
                          (long long)n_train * d_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_codes.data(), d_codes_train,
                          (long long)n_train * M_ * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));

    std::vector<float> residuals;
    residuals.reserve((long long)n_train * d_);
    std::vector<float> yhat(d_);

    for (int i = 0; i < n_train; i++) {
        cb_->reconstruct(h_codes.data() + (long long)i * M_, yhat.data());
        const float* yi = h_y.data() + (long long)i * d_;
        for (int j = 0; j < d_; j++)
            residuals.push_back(yi[j] - yhat[j]);
    }

    res_c1d_ = train_1d_kmeans(residuals.data(), (int)residuals.size(), Kr_);

    CUDA_CHECK(cudaMalloc(&d_res_c1d_, Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_res_c1d_, res_c1d_.data(), Kr_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void JHQGpuIndex::train(const float* h_x, int n_train) {
    jl_.estimate_sigma(h_x, n_train);
    cb_ = std::make_unique<LloydMaxCodebook>(d_, M_, B_, jl_.sigma());

    CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(),
                          (long long)d_ * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    const std::vector<float>& c1d = cb_->c1d();
    CUDA_CHECK(cudaMalloc(&d_c1d_, K1D_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_c1d_, c1d.data(), K1D_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    float*   d_y_train     = rotate_on_gpu(h_x, n_train);
    uint8_t* d_codes_train = nullptr;
    CUDA_CHECK(cudaMalloc(&d_codes_train, (long long)n_train * M_));

    launch_primary_encode(d_y_train, d_codes_train, d_c1d_,
                          n_train, d_, M_, Ds_, K1D_, bits_per_dim_);
    CUDA_CHECK(cudaDeviceSynchronize());

    train_residual_codebook(d_y_train, d_codes_train, n_train);

    cudaFree(d_y_train);
    cudaFree(d_codes_train);
}

void JHQGpuIndex::add(const float* h_x, int n) {
    if (!cb_) throw std::runtime_error("call train() before add()");

    const long long new_total = ntotal_ + n;
    auto grow = [&](auto** ptr, size_t elem_bytes, long long old_n, long long new_n) {
        void* tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, new_n * elem_bytes));
        if (*ptr && old_n > 0)
            CUDA_CHECK(cudaMemcpy(tmp, *ptr, old_n * elem_bytes, cudaMemcpyDeviceToDevice));
        cudaFree(*ptr);
        *ptr = (decltype(*ptr))tmp;
    };

    grow((void**)&d_primary_codes_, sizeof(uint8_t), (long long)ntotal_ * M_,   new_total * M_);
    grow((void**)&d_res_codes_,     sizeof(uint8_t), (long long)ntotal_ * bpv_, new_total * bpv_);
    grow((void**)&d_corrections_,   sizeof(float),   (long long)ntotal_,        new_total);

    float* d_y = rotate_on_gpu(h_x, n);

    uint8_t* d_new_pc = d_primary_codes_ + (long long)ntotal_ * M_;
    uint8_t* d_new_rc = d_res_codes_     + (long long)ntotal_ * bpv_;
    float*   d_new_co = d_corrections_   + (long long)ntotal_;

    launch_primary_encode(d_y, d_new_pc, d_c1d_,
                          n, d_, M_, Ds_, K1D_, bits_per_dim_);
    launch_residual_encode(d_y, d_new_pc, d_new_rc, d_new_co,
                           d_c1d_, d_res_c1d_,
                           n, d_, M_, Ds_, K1D_, Kr_, Br_, bpv_, bits_per_dim_);

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_y);

    ntotal_ += n;
    alloc_workspace(ntotal_);
}

void JHQGpuIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_labels) const {
    if (ntotal_ == 0) throw std::runtime_error("index is empty");

    float* d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, (long long)nq * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, (long long)nq * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    search_gpu(cublas_,
               d_Pi_, d_c1d_, d_res_c1d_,
               d_primary_codes_, d_res_codes_, d_corrections_,
               d_q,
               nq, ntotal_, d_, M_, Ds_, K1D_, Kr_,
               Br_, bpv_, bits_per_dim_,
               alpha_, k,
               ws_,
               h_dists, h_labels);

    cudaFree(d_q);
}

} // namespace jhq_gpu
