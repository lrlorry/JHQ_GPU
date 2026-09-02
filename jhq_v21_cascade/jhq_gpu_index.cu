#include "jhq_v21_cascade/jhq_gpu_index.cuh"

// Ablation switch for the residual codebook layout. The official
// get_scalar_codebook_ptr(subspace_idx, level) keeps one scalar codebook per
// subspace; v15 and earlier kept a single global one. Setting this to 1 trains
// the global codebook and replicates it into all M slots, so the device
// buffer, the kernels and the indexing stay byte-identical and the only thing
// that differs is the codebook contents -- which is what makes the comparison
// clean.  -DJHQ_GLOBAL_RESIDUAL_CB=1
#ifndef JHQ_GLOBAL_RESIDUAL_CB
#define JHQ_GLOBAL_RESIDUAL_CB 0
#endif
#include "jhq_v21_cascade/encode.cuh"
#include "jhq_v21_cascade/train_pq_gpu.cuh"
#include "jhq_v21_cascade/train_res_gpu.cuh"
#include "jhq_v21_cascade/search.cuh"
#include "common/cuda_utils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <numeric>
#include <chrono>
#include <string>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace jhq_gpu {

namespace {

// ── Assign kernel ─────────────────────────────────────────────────────────────
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

// ── Gather kernel ─────────────────────────────────────────────────────────────
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

// ── Transpose kernel: [N, M] → [M, N] with shared-memory tiling ──────────────
// TILE=32 avoids bank conflicts (padding +1 on shared dim).
// Grid: (ceil(N/TILE), ceil(M/TILE)) -- N (millions of vectors) MUST be the
// grid.x axis, not grid.y: CUDA caps grid.y/grid.z at 65535 regardless of
// compute capability, while grid.x goes up to 2^31-1. The original version
// put N on grid.y (ceil(N/TILE) with TILE=32 exceeds 65535 once N exceeds
// ~2.1M), which failed with "invalid argument" at the kernel launch on
// arxiv-abstracts-768 (2,253,000 vectors) -- never caught before since every
// dataset this was tested against until now (Vogue-768, openai3-1536) had
// fewer than ~2.1M vectors. M (subspace count, order 100s) safely stays
// under 65535 either way, so it's the one that belongs on grid.y.
// Each block transposes a TILE×TILE sub-block.
template <int TILE>
__global__ void transpose_uint8_kernel(
    const uint8_t* __restrict__ src,   // [N, M]
    uint8_t*                    dst,   // [M, N]
    long long N, int M)
{
    __shared__ uint8_t tile[TILE][TILE + 1];  // +1 avoids bank conflicts

    // Read: block reads tile[threadIdx.y][threadIdx.x] from src[row_src, col_src]
    long long col_src = (long long)blockIdx.y * TILE + threadIdx.x;  // m-axis
    long long row_src = (long long)blockIdx.x * TILE + threadIdx.y;  // n-axis
    if (col_src < M && row_src < N)
        tile[threadIdx.y][threadIdx.x] = src[row_src * M + col_src];
    __syncthreads();

    // Write: transposed — dst[row_dst, col_dst] where row is m-axis, col is n-axis
    long long col_dst = (long long)blockIdx.x * TILE + threadIdx.x;  // n-axis
    long long row_dst = (long long)blockIdx.y * TILE + threadIdx.y;  // m-axis
    if (col_dst < N && row_dst < M)
        dst[row_dst * N + col_dst] = tile[threadIdx.x][threadIdx.y];
}

} // namespace

// ── Constructor / Destructor ──────────────────────────────────────────────────
JHQGpuIndex::JHQGpuIndex(int d, Params p)
    : d_(d), M_(p.M), B_(p.B), Br_(p.Br),
      nlist_(p.nlist), nprobe_(p.nprobe), ivf_iters_(p.ivf_iters),
      batch_size_(p.batch_size), add_batch_(p.add_batch),
      kmeans_iters_(p.kmeans_iters), seed_(p.seed),
      alpha_(p.alpha),
      jl_(d, p.seed)
{
    if (d <= 0)          throw std::invalid_argument("d must be positive");
    if (p.M <= 0)        throw std::invalid_argument("M must be positive");
    if (d % p.M != 0)   throw std::invalid_argument("d must be divisible by M");
    // No B % Ds constraint: a subspace codeword is one of K = 2^B free
    // centroids in Ds dims, not a packed tuple of per-dimension indices, so
    // Ds may exceed B. That is what allows a primary code below one bit per
    // dimension -- see cpu/pq_codebook.h.
    if (p.B > 8)         throw std::invalid_argument("B must be <= 8");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    if (p.nlist <= 0)    throw std::invalid_argument("nlist must be positive");
    if (p.nprobe <= 0)   throw std::invalid_argument("nprobe must be positive");
    if (p.ivf_iters <= 0) throw std::invalid_argument("ivf_iters must be positive");
    if (p.batch_size <= 0) throw std::invalid_argument("batch_size must be positive");
    if (p.add_batch <= 0) throw std::invalid_argument("add_batch must be positive");
    if (p.kmeans_iters < 0) throw std::invalid_argument("kmeans_iters must be >= 0");
    if (p.alpha <= 0.0f) throw std::invalid_argument("alpha must be positive");

    Ds_           = d_ / M_;
    K_            = 1 << B_;
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
    cudaFree(d_cent_);
    cudaFree(d_res_c1d_);
    cudaFree(d_centroids_);
    cudaFree(d_cent_norms_);
    cudaFree(d_list_offsets_);
    cudaFree(d_list_ids_);
    cudaFree(d_list_primary_t_);
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

// ── Workspace allocation ──────────────────────────────────────────────────────
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
    CUDA_CHECK(cudaMalloc(&ws_.d_byte_lut,       B * M_ * 256         * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_r,          B * d_ * Kr_        * sizeof(float)));
    ws_.batch_cap = batch_size;

    if (ws_.h_q_pinned) { cudaFreeHost(ws_.h_q_pinned); ws_.h_q_pinned = nullptr; }
    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned, B * d_ * sizeof(float)));

    if (!ws_.stream) {
        CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    }
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));
}

// ── GPU rotation helper ───────────────────────────────────────────────────────
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

// ── IVF centroid training ─────────────────────────────────────────────────────
void JHQGpuIndex::train_ivf_centroids(
    const float* h_y_train, const float* d_y_train, int n_train)
{
    if (n_train < nlist_)
        throw std::invalid_argument("n_train must be >= nlist for v12_transposed");

    // Seed every list from an evenly spaced training vector. With the rotated
    // set no longer copied to the host, those rows are fetched one at a time
    // from the device -- nlist of them, against the whole set.
    centroids_.assign((long long)nlist_ * d_, 0.0f);
    for (int c = 0; c < nlist_; ++c) {
        const int src = (int)((long long)c * n_train / nlist_);
        if (h_y_train) {
            std::memcpy(centroids_.data() + (long long)c * d_,
                        h_y_train + (long long)src * d_,
                        (size_t)d_ * sizeof(float));
        } else {
            CUDA_CHECK(cudaMemcpy(centroids_.data() + (long long)c * d_,
                                  d_y_train + (long long)src * d_,
                                  (size_t)d_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
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

    // With the codebooks on the device the assignment no longer has to come back:
    // eight round trips and eight single-threaded passes over the training set
    // were 397 ms of the build, and only the assignment step was on the GPU.
    const bool gpu_ivf = std::getenv("JHQ_GPU_CODEBOOK") != nullptr;
    if (gpu_ivf) {
        for (int iter = 0; iter < ivf_iters_; ++iter) {
            upload_centroids();
            int* d_assign = assign_on_gpu(d_y_train, n_train);
            launch_ivf_accumulate(d_y_train, d_assign, d_centroids_,
                                  n_train, d_, nlist_, iter);
            cudaFree(d_assign);
            CUDA_CHECK(cudaMemcpy(centroids_.data(), d_centroids_,
                                  (long long)nlist_ * d_ * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        upload_centroids();
        return;
    }

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

// ── Residual codebook training ────────────────────────────────────────────────
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

    // One codebook per subspace, not one shared by all d dimensions. The
    // official get_scalar_codebook_ptr() is indexed by (subspace, level) for a
    // reason: the primary quantiser leaves a different residual scale in each
    // subspace, and a single global codebook has to straddle all of them.
    res_c1d_.assign((size_t)M_ * Kr_, 0.f);

    std::vector<float> yhat(d_);

    // Reconstruct once per vector, then slice per subspace, rather than
    // reconstructing M times.
    std::vector<float> all_resid((size_t)n_train * d_);
    for (int i = 0; i < n_train; i++) {
        cb_->reconstruct(h_codes.data() + (long long)i * M_, yhat.data());
        const float* yi = h_y.data() + (long long)i * d_;
        float* ri = all_resid.data() + (size_t)i * d_;
        for (int j = 0; j < d_; j++) ri[j] = yi[j] - yhat[j];
    }

#if JHQ_GLOBAL_RESIDUAL_CB
    {
        // One codebook over every dimension of every subspace, then replicated
        // so the layout the kernels see is unchanged.
        std::vector<float> cb =
            train_1d_kmeans(all_resid.data(), (int)all_resid.size(), Kr_);
        for (int m = 0; m < M_; ++m)
            std::copy(cb.begin(), cb.end(), res_c1d_.begin() + (size_t)m * Kr_);
    }
#else
    // This loop, not the primary codebook, is the index build: phase timing put
    // it at 23.5 s of a 28.5 s train against the PQ k-means's 4.2 s. Each
    // subspace reads its own stride of all_resid and writes its own slice of
    // res_c1d_, so the axis parallelises the same way -- resid moves inside so
    // each thread has its own scratch.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < M_; ++m) {
        std::vector<float> resid_m;
        resid_m.reserve((size_t)n_train * Ds_);
        for (int i = 0; i < n_train; i++) {
            const float* ri = all_resid.data() + (size_t)i * d_ + (size_t)m * Ds_;
            resid_m.insert(resid_m.end(), ri, ri + Ds_);
        }
        std::vector<float> cb = train_1d_kmeans(resid_m.data(), (int)resid_m.size(), Kr_);
        std::copy(cb.begin(), cb.end(), res_c1d_.begin() + (size_t)m * Kr_);
    }
#endif

    CUDA_CHECK(cudaMalloc(&d_res_c1d_, (size_t)M_ * Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_res_c1d_, res_c1d_.data(),
                          (size_t)M_ * Kr_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// ── Train ─────────────────────────────────────────────────────────────────────
// ── Trained-state cache ───────────────────────────────────────────────────────
//
// Training reads only the sample and the seed, so every point in a parameter
// sweep rebuilds the same codebooks: ~29 s on Vogue against ~1 s of search,
// which was three quarters of the wall time of a sweep. Keyed on everything
// train() consumes, plus a checksum of the sample, so a changed dataset or
// parameter misses rather than silently reusing the wrong codebooks.

void JHQGpuIndex::upload_trained() {
    if (!d_Pi_) CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(),
                          (long long)d_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

    if (!d_cent_) CUDA_CHECK(cudaMalloc(&d_cent_, cb_->size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cent_, cb_->data(), cb_->size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (!d_res_c1d_)
        CUDA_CHECK(cudaMalloc(&d_res_c1d_, (size_t)M_ * Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_res_c1d_, res_c1d_.data(),
                          (size_t)M_ * Kr_ * sizeof(float), cudaMemcpyHostToDevice));

    // Same work as the upload_centroids lambda inside train_ivf_centroids;
    // that one is local to the function, so it cannot be reused from here.
    std::vector<float> cent_norms(nlist_, 0.0f);
    for (int c = 0; c < nlist_; ++c) {
        double sn = 0.0;
        const float* cc = centroids_.data() + (long long)c * d_;
        for (int j = 0; j < d_; ++j) sn += (double)cc[j] * cc[j];
        cent_norms[c] = (float)sn;
    }
    cudaFree(d_centroids_);  d_centroids_  = nullptr;
    cudaFree(d_cent_norms_); d_cent_norms_ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_centroids_, (long long)nlist_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_.data(),
                          (long long)nlist_ * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cent_norms_, (long long)nlist_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cent_norms_, cent_norms.data(),
                          (long long)nlist_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

std::string JHQGpuIndex::cache_path(const char* dir, const float* h_x,
                                    int n_train) const {
    // FNV-1a over the parameters and a strided sample of the training data.
    // Striding keeps the hash O(1) in n_train while still covering the file.
    unsigned long long h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    const int params[] = { d_, M_, B_, Br_, K_, Kr_, nlist_,
                           ivf_iters_, kmeans_iters_, seed_, n_train };
    mix(params, sizeof params);
    const long long total = (long long)n_train * d_;
    const long long step  = total > 4096 ? total / 4096 : 1;
    for (long long i = 0; i < total; i += step) mix(&h_x[i], sizeof(float));

    char buf[64];
    std::snprintf(buf, sizeof buf, "/jhq_trained_%016llx.bin", h);
    return std::string(dir) + buf;
}

bool JHQGpuIndex::load_trained(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    if (!jl_.read_state(f)) return false;
    cb_ = std::make_unique<PQCodebook>(d_, M_, B_);
    if (!cb_->read_state(f)) return false;

    long long nr = 0, nc = 0;
    f.read(reinterpret_cast<char*>(&nr), sizeof nr);
    f.read(reinterpret_cast<char*>(&nc), sizeof nc);
    if (!f || nr != (long long)M_ * Kr_ || nc != (long long)nlist_ * d_) return false;
    res_c1d_.resize((size_t)nr);
    centroids_.resize((size_t)nc);
    f.read(reinterpret_cast<char*>(res_c1d_.data()),   (std::streamsize)nr * sizeof(float));
    f.read(reinterpret_cast<char*>(centroids_.data()), (std::streamsize)nc * sizeof(float));
    return (bool)f;
}

void JHQGpuIndex::save_trained(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    { std::ofstream f(tmp, std::ios::binary);
      if (!f) return;
      jl_.write_state(f);
      cb_->write_state(f);
      const long long nr = (long long)res_c1d_.size(), nc = (long long)centroids_.size();
      f.write(reinterpret_cast<const char*>(&nr), sizeof nr);
      f.write(reinterpret_cast<const char*>(&nc), sizeof nc);
      f.write(reinterpret_cast<const char*>(res_c1d_.data()),   (std::streamsize)nr * sizeof(float));
      f.write(reinterpret_cast<const char*>(centroids_.data()), (std::streamsize)nc * sizeof(float));
      if (!f) { std::remove(tmp.c_str()); return; }
    }
    // Rename last so a run killed mid-write never leaves a half file behind
    // that the next run would load as if it were complete.
    if (std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
}

// Phase timing for train(). The 29 s build against cuVS's 4.5 s was attributed
// to the PQ codebook k-means on the strength of an nsys trace showing only
// ~700 ms of GPU work, but that only says the time is on the host, not which
// host loop owns it -- parallelising the codebook changed the total by nothing.
#define JHQ_TRAIN_PHASE(label)                                                \
    do { if (phase_timing) {                                                  \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
        const double now = std::chrono::duration<double, std::milli>(         \
            std::chrono::high_resolution_clock::now() - t_phase).count();     \
        std::printf("  [train] %-22s %8.1f ms\n", label, now);                \
        t_phase = std::chrono::high_resolution_clock::now();                   \
    } } while (0)

void JHQGpuIndex::train(const float* h_x, int n_train) {
    const bool phase_timing = std::getenv("JHQ_TRAIN_PHASES") != nullptr;
    auto t_phase = std::chrono::high_resolution_clock::now();

    jl_.estimate_sigma(h_x, n_train);
    JHQ_TRAIN_PHASE("estimate_sigma");

    const char* cache_dir = std::getenv("JHQ_INDEX_CACHE");
    std::string cpath;
    if (cache_dir) {
        cpath = cache_path(cache_dir, h_x, n_train);
        if (load_trained(cpath)) {
            upload_trained();
            JHQ_TRAIN_PHASE("loaded from cache");
            return;
        }
    }
    cb_ = std::make_unique<PQCodebook>(d_, M_, B_);

    CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(),
                          (long long)d_ * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    // The PQ centroids need the rotated training data, so rotate first and
    // pull it back once -- train() already copies it to the host below for
    // the IVF, so this costs one extra D2H at build time, not per query.
    const bool gpu_codebook = std::getenv("JHQ_GPU_CODEBOOK") != nullptr;

    float* d_y_train = rotate_on_gpu(h_x, n_train);
    // The rotated training set is 2.9 GB on Vogue and every host-side consumer
    // of it has moved to the device, except the analytical initialisation --
    // and that reads only the per-subspace mean and variance. Computing those
    // here brings back M*Ds*2 floats instead.
    std::vector<float> h_y_train;
    if (!gpu_codebook) {
        h_y_train.resize((long long)n_train * d_);
        CUDA_CHECK(cudaMemcpy(h_y_train.data(), d_y_train,
                              (long long)n_train * d_ * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    JHQ_TRAIN_PHASE(gpu_codebook ? "rotate" : "rotate + D2H");

    // The primary codebook is 96 independent subspaces of 100k points against
    // 256 centroids in 8 dimensions -- about 98 GFLOP over five iterations, and
    // 3.5 s of the 8.3 s build even with every core busy. Setting
    // JHQ_GPU_CODEBOOK runs the Lloyd iterations on the device instead; the
    // analytical placement stays on the host so both paths start identically.
    if (gpu_codebook) {
        const int cols = M_ * Ds_;
        float *d_mean = nullptr, *d_var = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mean, (size_t)cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_var,  (size_t)cols * sizeof(float)));
        launch_subspace_stats(d_y_train, n_train, d_, M_, Ds_, d_mean, d_var);
        std::vector<float> h_mean(cols), h_var(cols);
        CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, (size_t)cols * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_var.data(), d_var, (size_t)cols * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_mean); cudaFree(d_var);
        cb_->init_from_stats(h_mean.data(), h_var.data(), seed_);
    } else {
        cb_->train(h_y_train.data(), n_train, kmeans_iters_, seed_);
    }
    JHQ_TRAIN_PHASE(gpu_codebook ? "pq codebook init (moments on device)"
                                 : "pq codebook kmeans");

    CUDA_CHECK(cudaMalloc(&d_cent_, cb_->size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cent_, cb_->data(), cb_->size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (gpu_codebook && kmeans_iters_ > 0) {
        launch_pq_kmeans(d_y_train, n_train, d_, M_, Ds_, K_,
                         d_cent_, kmeans_iters_);
        // hand the result back: the residual level trains against these, and
        // the trained-state cache stores them
        CUDA_CHECK(cudaMemcpy(cb_->mutable_data(), d_cent_,
                              cb_->size() * sizeof(float), cudaMemcpyDeviceToHost));
        JHQ_TRAIN_PHASE("pq codebook kmeans (device)");
    }

    uint8_t* d_codes_train = nullptr;
    CUDA_CHECK(cudaMalloc(&d_codes_train, (long long)n_train * M_));

    launch_primary_encode(d_y_train, d_codes_train, d_cent_,
                          n_train, d_, M_, Ds_, K_);
    CUDA_CHECK(cudaDeviceSynchronize());
    JHQ_TRAIN_PHASE("primary encode");

    train_ivf_centroids(h_y_train.empty() ? nullptr : h_y_train.data(),
                        d_y_train, n_train);
    JHQ_TRAIN_PHASE("ivf centroids");

    if (gpu_codebook) {
        // The largest phase of the build, and almost all of it a sort: one per
        // subspace over n_train*Ds values. On the device that is a single
        // segmented radix sort, and the residual is written straight into the
        // segment-major layout the sort wants, so the host's gather disappears.
        res_c1d_.assign((size_t)M_ * Kr_, 0.f);
        if (!d_res_c1d_)
            CUDA_CHECK(cudaMalloc(&d_res_c1d_, (size_t)M_ * Kr_ * sizeof(float)));
        launch_residual_codebook(d_y_train, d_codes_train, d_cent_,
                                 n_train, d_, M_, Ds_, K_, Kr_, 25, d_res_c1d_);
        CUDA_CHECK(cudaMemcpy(res_c1d_.data(), d_res_c1d_,
                              (size_t)M_ * Kr_ * sizeof(float), cudaMemcpyDeviceToHost));
        // The correction term per vector still comes from the host encode path
        // in add(); only the codebook moved.
        JHQ_TRAIN_PHASE("residual codebook (device)");
    } else {
        train_residual_codebook(d_y_train, d_codes_train, n_train);
        JHQ_TRAIN_PHASE("residual codebook");
    }

    if (cache_dir) { save_trained(cpath); JHQ_TRAIN_PHASE("cache write"); }

    cudaFree(d_y_train);
    cudaFree(d_codes_train);
}

// ── GPU assignment helper ─────────────────────────────────────────────────────
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

void JHQGpuIndex::assign_into(const float* d_y, int n, int* d_out,
                              float* d_dots, cudaStream_t stream) const {
    const int   batch = 8192;
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_, stream));
    for (int start = 0; start < n; start += batch) {
        int nb = std::min(batch, n - start);
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 nlist_, nb, d_, &one,
                                 d_centroids_, d_,
                                 d_y + (long long)start * d_, d_,
                                 &zero, d_dots, nlist_));
        assign_from_dots_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(
            d_dots, d_cent_norms_, d_out + start, nlist_, nb);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ── Add ───────────────────────────────────────────────────────────────────────
// v12_transposed's add() called rotate_on_gpu(h_x, n) on the WHOLE n at
// once -- two full n*d float device buffers (raw + JL-rotated). That's
// fine up to a few million rows, but at bge-m3 (10.09M x 1024) or
// stella-trec24 (17.8M x 1024) scale it needs ~83GB / ~145GB of VRAM
// before a single kernel runs, far past any single GPU.
//
// Every stage AFTER rotation, though, already only touches compressed
// per-vector data -- uint8 codes (d_pc, d_rc), scalars (d_co), or ints
// (d_assign) -- which for the whole n is one to two orders of magnitude
// smaller than the raw float vectors (e.g. stella-trec24: d_pc+d_rc+d_co+
// d_assign is ~10.7GB total vs. ~145GB for two full float buffers). So
// the fix isn't a full two-pass IVF-offset rewrite (unlike JHQ_official's
// IndexIVFJHQ::add_core(), which needs one because FAISS's InvertedLists
// storage is filled incrementally list-by-list) -- it's simpler: only the
// rotate+encode+assign step needs to stream, exactly mirroring how
// JHQ_repro's CPU add() chunks its own expensive stage (batch=32768)
// while accumulating into a single set of outputs. Everything from the
// thrust::sort_by_key onward is untouched from v12_transposed: it already
// only allocates n-sized compressed arrays, which comfortably fit (see
// jhq_v21_cascade/jhq_gpu_index.cuh's Params::add_batch comment for
// the full VRAM accounting).
void JHQGpuIndex::add(const float* h_x, int n) {
    if (!cb_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0)
        throw std::runtime_error("v14_streaming_add currently supports one add() call");

    uint8_t* d_pc = nullptr;      // [n, M]   primary codes
    uint8_t* d_rc = nullptr;      // [n, bpv] residual codes
    float*   d_co = nullptr;      // [n]      residual corrections
    int*     d_assign = nullptr;  // [n]      IVF cluster assignment
    CUDA_CHECK(cudaMalloc(&d_pc, (long long)n * M_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_rc, (long long)n * bpv_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_co, (long long)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assign, (long long)n * sizeof(int)));

    const long long AB = add_batch_;

    // add() is the part of the build that scales with N: train is capped at
    // 100k sample vectors whatever the dataset, so at 17.8M rows add is 96% of
    // the build. Timing the phases rather than guessing which one costs,
    // reported when JHQ_ADD_PHASES is set.
    const bool add_phases = std::getenv("JHQ_ADD_PHASES") != nullptr;
    double t_h2d = 0, t_rot = 0, t_penc = 0, t_renc = 0, t_asg = 0;
    auto tick = [&]() {
        if (add_phases) CUDA_CHECK(cudaDeviceSynchronize());
        return std::chrono::steady_clock::now();
    };
    auto lap = [&](std::chrono::steady_clock::time_point& t0, double& acc) {
        if (!add_phases) return;
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    };

    // Two-deep pipeline. The loop it replaces did a pageable copy, a device-wide
    // synchronise and a pair of mallocs per batch, so the link sat idle while the
    // GPU worked and vice versa. Two pinned staging buffers, two streams and one
    // hoisted scratch let the host copy for batch i+1 run against the GPU work for
    // batch i.
    //
    // Phase timing forces a synchronise after every step, so JHQ_ADD_PHASES
    // degenerates this to the serial order on purpose: attribution needs the
    // steps separated, throughput needs them overlapped.
    const int NBUF = add_phases ? 1 : 2;
    float*       h_stage[2] = {nullptr, nullptr};
    float*       d_xb[2]    = {nullptr, nullptr};
    float*       d_yb[2]    = {nullptr, nullptr};
    cudaStream_t st[2]      = {nullptr, nullptr};
    cudaEvent_t  ev[2]      = {nullptr, nullptr};
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaHostAlloc(&h_stage[b], (size_t)AB * d_ * sizeof(float),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&d_xb[b], (long long)AB * d_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_yb[b], (long long)AB * d_ * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&st[b]));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming));
    }
    // One assignment scratch per stream, not one for the loop. Hoisting it out
    // of the batch loop removes a malloc and a free per batch, but a single
    // buffer is written by both streams at once: the coarse assignment then
    // lands in the wrong list for whichever batch loses, and recall falls from
    // 0.9452 to 0.80. Single-threaded staging hid it -- the copy was slow enough
    // that the previous batch's assignment had finished -- so it only appeared
    // once the staging copy got fast.
    float* d_dots[2] = {nullptr, nullptr};
    for (int b = 0; b < NBUF; ++b)
        CUDA_CHECK(cudaMalloc(&d_dots[b], (long long)nlist_ * 8192 * sizeof(float)));

    const float one = 1.f, zero = 0.f;
    int nbatch = 0;
    for (long long start = 0; start < n; start += AB, ++nbatch) {
        const int nb  = (int)std::min(AB, (long long)n - start);
        const int cur = nbatch % NBUF;
        if (nbatch >= NBUF) CUDA_CHECK(cudaEventSynchronize(ev[cur]));

        auto t0 = tick();
        // The source is an mmap of the base file, so it cannot be handed to the
        // DMA engine directly and has to pass through the pinned buffer. That
        // copy, single-threaded, was the wall: 12.3 GB at a few GB/s on
        // openai3-3072, more than the GPU work it was supposed to hide behind.
        // Splitting it across the host's cores is the cheapest way to make it
        // small enough to overlap.
        {
            const size_t bytes = (size_t)nb * d_ * sizeof(float);
            const char*  src   = (const char*)(h_x + start * (long long)d_);
            char*        dst   = (char*)h_stage[cur];
#ifdef _OPENMP
            // Eight threads copied this fastest; 208 took three times as long as
            // one, the fork, join and the cores taken from the CUDA driver
            // costing more than the copy they split.
            const int nthr = omp_get_max_threads() < 8 ? omp_get_max_threads() : 8;
#pragma omp parallel num_threads(nthr)
            {
                const int nt = omp_get_num_threads(), ti = omp_get_thread_num();
                const size_t chunk = (bytes + nt - 1) / nt;
                const size_t lo = (size_t)ti * chunk;
                const size_t hi = lo + chunk < bytes ? lo + chunk : bytes;
                if (lo < bytes) std::memcpy(dst + lo, src + lo, hi - lo);
            }
#else
            std::memcpy(dst, src, bytes);
#endif
        }
        CUDA_CHECK(cudaMemcpyAsync(d_xb[cur], h_stage[cur],
                                   (long long)nb * d_ * sizeof(float),
                                   cudaMemcpyHostToDevice, st[cur]));
        lap(t0, t_h2d);

        CUBLAS_CHECK(cublasSetStream(cublas_, st[cur]));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 d_, nb, d_, &one, d_Pi_, d_,
                                 d_xb[cur], d_, &zero, d_yb[cur], d_));
        lap(t0, t_rot);

        launch_primary_encode(d_yb[cur], d_pc + start * M_, d_cent_,
                              nb, d_, M_, Ds_, K_, st[cur]);
        lap(t0, t_penc);

        launch_residual_encode(d_yb[cur], d_pc + start * M_,
                               d_rc + start * bpv_, d_co + start,
                               d_cent_, d_res_c1d_,
                               nb, d_, M_, Ds_, K_, Kr_, Br_, bpv_, st[cur]);
        lap(t0, t_renc);

        assign_into(d_yb[cur], nb, d_assign + start, d_dots[cur], st[cur]);
        CUDA_CHECK(cudaEventRecord(ev[cur], st[cur]));
        lap(t0, t_asg);
    }
    for (int b = 0; b < NBUF; ++b) CUDA_CHECK(cudaStreamSynchronize(st[b]));
    CUBLAS_CHECK(cublasSetStream(cublas_, 0));

    if (add_phases)
        std::fprintf(stderr,
            "  [add] h2d %.1f ms | rotate %.1f | primary encode %.1f | "
            "residual encode %.1f | ivf assign %.1f\n",
            t_h2d, t_rot, t_penc, t_renc, t_asg);

    for (int b = 0; b < NBUF; ++b) {
        cudaFree(d_dots[b]);
        cudaFreeHost(h_stage[b]); cudaFree(d_xb[b]); cudaFree(d_yb[b]);
        cudaStreamDestroy(st[b]); cudaEventDestroy(ev[b]);
    }

    // ---- Unchanged from v12_transposed from here down. ----
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
    CUDA_CHECK(cudaMalloc(&d_list_res_,     (long long)n * bpv_ * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_list_corr_,    (long long)n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_list_offsets_, offsets.data(),
                          (long long)(nlist_ + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Temporary [N, M] primary buffer for gathering, then transposed.
    uint8_t* d_list_primary_nm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_list_primary_nm, (long long)n * M_ * sizeof(uint8_t)));

    gather_list_storage_kernel<<<(n + 255) / 256, 256>>>(
        d_order, d_pc, d_rc, d_co,
        d_list_ids_, d_list_primary_nm, d_list_res_, d_list_corr_,
        n, M_, bpv_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Transpose [N, M] → [M, N] for coalesced scan access.
    // grid.x = N-tiles, grid.y = M-tiles -- see transpose_uint8_kernel's
    // comment for why N (which can be in the millions) must be on grid.x,
    // not grid.y (CUDA's 65535 cap on grid.y/z).
    constexpr int TILE = 32;
    CUDA_CHECK(cudaMalloc(&d_list_primary_t_, (long long)M_ * n * sizeof(uint8_t)));
    dim3 grid(((long long)n + TILE - 1) / TILE, (M_ + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    transpose_uint8_kernel<TILE><<<grid, block>>>(d_list_primary_nm, d_list_primary_t_,
                                                   (long long)n, M_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary [N, M] buffer — scan uses d_list_primary_t_ only.
    cudaFree(d_list_primary_nm);

    cudaFree(d_assign);
    cudaFree(d_order);
    cudaFree(d_pc);
    cudaFree(d_rc);
    cudaFree(d_co);

    ntotal_ = n;
    alloc_workspace(batch_size_);
}

// ── Search ────────────────────────────────────────────────────────────────────
void JHQGpuIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_labels) const {
    if (ntotal_ == 0) throw std::runtime_error("index is empty");

    search_gpu(cublas_,
               d_Pi_, d_cent_, d_res_c1d_,
               d_centroids_, d_cent_norms_,
               d_list_offsets_, d_list_ids_,
               d_list_primary_t_, d_list_res_, d_list_corr_,
               h_q,
               nq, d_, M_, Ds_, K_, Kr_, nlist_, nprobe_,
               Br_, bpv_,
               alpha_, k,
               batch_size_,
               ntotal_,
               ws_,
               h_dists, h_labels);
}

} // namespace jhq_gpu
