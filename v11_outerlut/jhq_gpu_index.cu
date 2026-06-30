#include "v11_outerlut/jhq_gpu_index.cuh"
#include "v11_outerlut/encode.cuh"
#include "v11_outerlut/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace jhq_gpu {

namespace {

__global__ void assign_from_dots_kernel(
    const float* __restrict__ dots,
    const float* __restrict__ cent_norms,
    int*                  assigns,
    int nlist, int nb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const float* col = dots + (long long)i * nlist;
    float best = cent_norms[0] - 2.0f * col[0];
    int best_id = 0;
    for (int c = 1; c < nlist; ++c) {
        float dist = cent_norms[c] - 2.0f * col[c];
        if (dist < best) { best = dist; best_id = c; }
    }
    assigns[i] = best_id;
}

__global__ void gather_list_storage_kernel(
    const int*     __restrict__ sorted_ids,
    const uint8_t* __restrict__ primary,
    const uint8_t* __restrict__ residual,
    const float*   __restrict__ corrections,
    int*                        list_ids,
    uint8_t*                    list_primary,
    uint8_t*                    list_res,
    float*                      list_corr,
    int n, int M, int bpv)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n) return;
    int id = sorted_ids[pos];
    list_ids[pos] = id;
    const uint8_t* pc = primary + (long long)id * M;
    uint8_t* out_pc = list_primary + (long long)pos * M;
    for (int m = 0; m < M; ++m) out_pc[m] = pc[m];
    const uint8_t* rc = residual + (long long)id * bpv;
    uint8_t* out_rc = list_res + (long long)pos * bpv;
    for (int b = 0; b < bpv; ++b) out_rc[b] = rc[b];
    list_corr[pos] = corrections[id];
}

} // namespace

JHQGpuIndex::JHQGpuIndex(int d, Params p)
    : d_(d), M_(p.M), B_(p.B), Br_(p.Br),
      nlist_(p.nlist), nprobe_(p.nprobe), ivf_iters_(p.ivf_iters),
      batch_size_(p.batch_size),
      alpha_(p.alpha),
      jl_(d, p.seed)
{
    if (d <= 0)          throw std::invalid_argument("d must be positive");
    if (p.M <= 0)        throw std::invalid_argument("M must be positive");
    if (d % p.M != 0)   throw std::invalid_argument("d must be divisible by M");
    if (p.B % (d/p.M) != 0) throw std::invalid_argument("B must be divisible by Ds = d/M");
    if (p.B > 8)         throw std::invalid_argument("B must be <= 8");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    if (p.nlist <= 0)    throw std::invalid_argument("nlist must be positive");
    if (p.nprobe <= 0)   throw std::invalid_argument("nprobe must be positive");
    if (p.ivf_iters <= 0) throw std::invalid_argument("ivf_iters must be positive");
    if (p.batch_size <= 0) throw std::invalid_argument("batch_size must be positive");
    if (p.alpha <= 0.0f) throw std::invalid_argument("alpha must be positive");

    Ds_           = d_ / M_;
    bits_per_dim_ = B_ / Ds_;
    K1D_          = 1 << bits_per_dim_;
    Kr_           = 1 << Br_;
    bpv_          = (d_ * Br_ + 7) / 8;
    nprobe_       = std::min(nprobe_, nlist_);

    CUBLAS_CHECK(cublasCreate(&cublas_));
}

JHQGpuIndex::~JHQGpuIndex() {
    if (ws_.graph_exec) cudaGraphExecDestroy(ws_.graph_exec);
    if (ws_.graph)      cudaGraphDestroy(ws_.graph);
    if (ws_.stream)     cudaStreamDestroy(ws_.stream);

    cublasDestroy(cublas_);
    cudaFree(d_Pi_);
    cudaFree(d_c1d_);
    cudaFree(d_res_c1d_);
    cudaFree(d_centroids_);
    cudaFree(d_cent_norms_);
    cudaFree(d_list_offsets_);
    cudaFree(d_list_ids_);
    cudaFree(d_list_primary_);
    cudaFree(d_list_res_);
    cudaFree(d_list_corr_);
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch);
    cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_dots);
    cudaFree(ws_.d_byte_lut);
    cudaFree(ws_.d_probe_ids);
    cudaFree(ws_.d_probe_offsets);
    cudaFree(ws_.d_query_total);
    cudaFree(ws_.d_topck_pos);
    cudaFree(ws_.d_topck_primary);
    cudaFree(ws_.d_lut_r);
    cudaFree(ws_.d_comp_dists);
    cudaFree(ws_.d_final_ids);
    cudaFree(ws_.d_final_dists);
}

void JHQGpuIndex::alloc_workspace(int batch_size) {
    if (ws_.graph_exec) { cudaGraphExecDestroy(ws_.graph_exec); ws_.graph_exec = nullptr; }
    if (ws_.graph)      { cudaGraphDestroy(ws_.graph);          ws_.graph      = nullptr; }
    ws_.graph_ck     = 0;
    ws_.graph_nprobe = 0;

    cudaFree(ws_.d_q_batch);       ws_.d_q_batch        = nullptr;
    cudaFree(ws_.d_q_rot);         ws_.d_q_rot          = nullptr;
    cudaFree(ws_.d_dots);          ws_.d_dots           = nullptr;
    cudaFree(ws_.d_probe_ids);     ws_.d_probe_ids      = nullptr;
    cudaFree(ws_.d_probe_offsets); ws_.d_probe_offsets  = nullptr;
    cudaFree(ws_.d_query_total);   ws_.d_query_total    = nullptr;
    cudaFree(ws_.d_byte_lut);      ws_.d_byte_lut       = nullptr;
    cudaFree(ws_.d_topck_pos);     ws_.d_topck_pos      = nullptr;
    cudaFree(ws_.d_topck_primary); ws_.d_topck_primary  = nullptr;
    cudaFree(ws_.d_lut_r);         ws_.d_lut_r          = nullptr;
    cudaFree(ws_.d_comp_dists);    ws_.d_comp_dists     = nullptr;
    cudaFree(ws_.d_final_ids);     ws_.d_final_ids      = nullptr;
    cudaFree(ws_.d_final_dists);   ws_.d_final_dists    = nullptr;
    ws_.ck_cap = 0;
    ws_.k_cap  = 0;

    long long B = batch_size;
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,        B * d_               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_rot,          B * d_               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots,           B * nlist_           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_probe_ids,      B * nprobe_          * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_probe_offsets,  B * (nprobe_ + 1)   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_query_total,    B                    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_byte_lut,       B * M_ * 256         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_r,          B * d_ * Kr_        * sizeof(float)));
    ws_.batch_cap = batch_size;

    if (ws_.h_q_pinned) { cudaFreeHost(ws_.h_q_pinned); ws_.h_q_pinned = nullptr; }
    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned, B * d_ * sizeof(float)));

    if (!ws_.stream) {
        CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    }
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));
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

void JHQGpuIndex::train_ivf_centroids(
    const float* h_y_train, const float* d_y_train, int n_train)
{
    if (n_train < nlist_)
        throw std::invalid_argument("n_train must be >= nlist for v11_outerlut");

    centroids_.assign((long long)nlist_ * d_, 0.0f);
    for (int c = 0; c < nlist_; ++c) {
        int src = (int)((long long)c * n_train / nlist_);
        std::memcpy(centroids_.data() + (long long)c * d_,
                    h_y_train + (long long)src * d_,
                    (size_t)d_ * sizeof(float));
    }

    auto upload_centroids = [&]() {
        std::vector<float> cent_norms(nlist_, 0.0f);
        for (int c = 0; c < nlist_; ++c) {
            double s = 0.0;
            const float* cc = centroids_.data() + (long long)c * d_;
            for (int j = 0; j < d_; ++j) s += (double)cc[j] * cc[j];
            cent_norms[c] = (float)s;
        }
        cudaFree(d_centroids_);
        cudaFree(d_cent_norms_);
        d_centroids_ = nullptr;
        d_cent_norms_ = nullptr;
        CUDA_CHECK(cudaMalloc(&d_centroids_,  (long long)nlist_ * d_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_.data(),
                              (long long)nlist_ * d_ * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_cent_norms_, (long long)nlist_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_cent_norms_, cent_norms.data(),
                              (long long)nlist_ * sizeof(float),
                              cudaMemcpyHostToDevice));
    };

    std::vector<int>    h_assign(n_train);
    std::vector<double> sums((long long)nlist_ * d_);
    std::vector<int>    counts(nlist_);

    for (int iter = 0; iter < ivf_iters_; ++iter) {
        upload_centroids();
        int* d_assign = assign_on_gpu(d_y_train, n_train);
        CUDA_CHECK(cudaMemcpy(h_assign.data(), d_assign,
                              (long long)n_train * sizeof(int),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_assign);

        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < n_train; ++i) {
            int c = h_assign[i];
            counts[c]++;
            const float* yi = h_y_train + (long long)i * d_;
            double* sc = sums.data() + (long long)c * d_;
            for (int j = 0; j < d_; ++j) sc[j] += yi[j];
        }
        for (int c = 0; c < nlist_; ++c) {
            float* cc = centroids_.data() + (long long)c * d_;
            if (counts[c] == 0) {
                int src = (int)(((long long)c * 1103515245 + iter * 12345) % n_train);
                std::memcpy(cc, h_y_train + (long long)src * d_,
                            (size_t)d_ * sizeof(float));
                continue;
            }
            const double inv = 1.0 / (double)counts[c];
            const double* sc = sums.data() + (long long)c * d_;
            for (int j = 0; j < d_; ++j) cc[j] = (float)(sc[j] * inv);
        }
    }
    upload_centroids();
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

    std::vector<float> h_y_train((long long)n_train * d_);
    CUDA_CHECK(cudaMemcpy(h_y_train.data(), d_y_train,
                          (long long)n_train * d_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    train_ivf_centroids(h_y_train.data(), d_y_train, n_train);
    train_residual_codebook(d_y_train, d_codes_train, n_train);

    cudaFree(d_y_train);
    cudaFree(d_codes_train);
}

int* JHQGpuIndex::assign_on_gpu(const float* d_y, int n) const {
    int* d_assign = nullptr;
    CUDA_CHECK(cudaMalloc(&d_assign, (long long)n * sizeof(int)));

    const int batch = 8192;
    float* d_dots = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dots, (long long)nlist_ * batch * sizeof(float)));

    const float one = 1.f, zero = 0.f;
    for (int start = 0; start < n; start += batch) {
        int nb = std::min(batch, n - start);
        CUBLAS_CHECK(cublasSgemm(cublas_,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 nlist_, nb, d_,
                                 &one,
                                 d_centroids_, d_,
                                 d_y + (long long)start * d_, d_,
                                 &zero,
                                 d_dots, nlist_));
        assign_from_dots_kernel<<<(nb + 255) / 256, 256>>>(
            d_dots, d_cent_norms_, d_assign + start, nlist_, nb);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_dots);
    return d_assign;
}

void JHQGpuIndex::add(const float* h_x, int n) {
    if (!cb_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0)
        throw std::runtime_error("v11_outerlut currently supports one add() call");

    float* d_y = rotate_on_gpu(h_x, n);

    uint8_t* d_pc = nullptr;
    uint8_t* d_rc = nullptr;
    float*   d_co = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pc, (long long)n * M_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_rc, (long long)n * bpv_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_co, (long long)n * sizeof(float)));

    launch_primary_encode(d_y, d_pc, d_c1d_,
                          n, d_, M_, Ds_, K1D_, bits_per_dim_);
    launch_residual_encode(d_y, d_pc, d_rc, d_co,
                           d_c1d_, d_res_c1d_,
                           n, d_, M_, Ds_, K1D_, Kr_, Br_, bpv_, bits_per_dim_);
    CUDA_CHECK(cudaDeviceSynchronize());

    int* d_assign = assign_on_gpu(d_y, n);

    int* d_order = nullptr;
    CUDA_CHECK(cudaMalloc(&d_order, (long long)n * sizeof(int)));
    thrust::device_ptr<int> t_assign(d_assign);
    thrust::device_ptr<int> t_order(d_order);
    thrust::sequence(t_order, t_order + n);
    thrust::sort_by_key(t_assign, t_assign + n, t_order);

    std::vector<int> h_assign(n);
    CUDA_CHECK(cudaMemcpy(h_assign.data(), d_assign, (long long)n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<int> counts(nlist_, 0);
    for (int a : h_assign) {
        if (a < 0 || a >= nlist_) throw std::runtime_error("invalid IVF assignment");
        counts[a]++;
    }

    std::vector<int> offsets(nlist_ + 1, 0);
    for (int c = 0; c < nlist_; ++c) offsets[c + 1] = offsets[c] + counts[c];

    CUDA_CHECK(cudaMalloc(&d_list_offsets_, (long long)(nlist_ + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_list_ids_,     (long long)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_list_primary_, (long long)n * M_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_list_res_,     (long long)n * bpv_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_list_corr_,    (long long)n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_list_offsets_, offsets.data(),
                          (long long)(nlist_ + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    gather_list_storage_kernel<<<(n + 255) / 256, 256>>>(
        d_order, d_pc, d_rc, d_co,
        d_list_ids_, d_list_primary_, d_list_res_, d_list_corr_,
        n, M_, bpv_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_assign);
    cudaFree(d_order);
    cudaFree(d_y);
    cudaFree(d_pc);
    cudaFree(d_rc);
    cudaFree(d_co);

    ntotal_ = n;
    alloc_workspace(batch_size_);
}

void JHQGpuIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_labels) const {
    if (ntotal_ == 0) throw std::runtime_error("index is empty");

    search_gpu(cublas_,
               d_Pi_, d_c1d_, d_res_c1d_,
               d_centroids_, d_cent_norms_,
               d_list_offsets_, d_list_ids_,
               d_list_primary_, d_list_res_, d_list_corr_,
               h_q,
               nq, d_, M_, Ds_, K1D_, Kr_, nlist_, nprobe_,
               Br_, bpv_, bits_per_dim_,
               alpha_, k,
               batch_size_,
               ntotal_,
               ws_,
               h_dists, h_labels);
}

} // namespace jhq_gpu
