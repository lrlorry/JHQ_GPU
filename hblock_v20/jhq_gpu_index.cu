#include "hblock_v20/jhq_gpu_index.cuh"
#include "hblock_v17/encode.cuh"
#include "hblock_v20/search.cuh"
#include "cpu/erfinv.h"
#include "common/cuda_utils.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using Ms = std::chrono::duration<double, std::milli>;

namespace hblock_v20 {

static std::vector<float>
analytical_fine_c1d(int Kr, float sigma)
{
    std::vector<float> c(Kr);
    for (int i = 0; i < Kr; ++i) {
        float q = (i + 0.5f) / float(Kr);
        c[i] = sigma * float(M_SQRT2) * erfinv_f(2.f * q - 1.f);
    }
    return c;
}

HBlockIndex::HBlockIndex(int d, Params p)
    : d_(d), d_proj_(p.d_proj), Kr_(p.Kr), Br_(p.Br),
      bpv_((d * p.Br + 7) / 8),
      leaf_size_(p.leaf_size),
      K1_(p.K1), K2_(p.K2), K3_(p.K3),
      ck1_(p.ck1), ck2_(p.ck2), ck3_(p.ck3),
      rerank_r_(p.rerank_r), km_iters_(p.km_iters),
      batch_size_(p.batch_size)
{
    if (d <= 0) throw std::invalid_argument("d must be positive");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    CUBLAS_CHECK(cublasCreate(&cublas_));
}

HBlockIndex::~HBlockIndex() {
    if (ws_.stream) { cublasSetStream(cublas_, nullptr); cudaStreamDestroy(ws_.stream); }

    auto free_h = [](void* p){ if (p) cudaFreeHost(p); };
    auto free_d = [](void* p){ if (p) cudaFree(p); };

    free_h(ws_.h_q_pinned); free_h(ws_.h_leaf_cnt);
    free_h(ws_.h_final_dists); free_h(ws_.h_final_ids);
    free_d(ws_.d_q_batch);
    free_d(ws_.d_q_proj1);
    free_d(ws_.d_dots1);
    free_d(ws_.d_top1_ids);
    free_d(ws_.d_r1_beam);
    free_d(ws_.d_top2_beam);
    free_d(ws_.d_top3_beam);
    free_d(ws_.d_q_r3);
    free_d(ws_.d_leaf_sel); free_d(ws_.d_leaf_cnt);
    free_d(ws_.d_lut_fine);
    free_d(ws_.d_query_offsets);
    free_d(ws_.d_pair_leaf_a); free_d(ws_.d_pair_qid_a);
    free_d(ws_.d_pair_leaf_b); free_d(ws_.d_pair_qid_b);
    free_d(ws_.d_out_dists);   free_d(ws_.d_out_ids);
    free_d(ws_.d_pq_dists);    free_d(ws_.d_pq_ids);
    free_d(ws_.d_cand_vecs);   free_d(ws_.d_exact_dists);
    free_d(ws_.d_final_dists); free_d(ws_.d_final_ids);
    free_d(ws_.d_cub_tmp);

    free_d(d_Pi1_); free_d(d_Pi2_); free_d(d_Pi3_);
    free_d(d_route1_cents_proj_); free_d(d_route1_cents_full_); free_d(d_route1_norms_);
    free_d(d_route2_cents_proj_); free_d(d_route2_cents_full_); free_d(d_route2_norms_);
    free_d(d_route3_cents_proj_); free_d(d_route3_cents_full_); free_d(d_route3_norms_);
    free_d(d_fine_c1d_);
    free_d(d_pair_blk_start_);  free_d(d_pair_blk_count_);
    free_d(d_leaf_codes_);     free_d(d_leaf_ids_);    free_d(d_leaf_sizes_);
    free_d(d_base_vecs_);

    cublasDestroy(cublas_);
}

void HBlockIndex::init_jl_proj(int d, int d_proj, int seed, std::vector<float>& Pi)
{
    Pi.resize((long long)d_proj * d);
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f / std::sqrt((float)d_proj));
    for (float& v : Pi) v = g(rng);
}

void HBlockIndex::gpu_kmeans(const float* h_x_proj, const float* h_x_full,
                              int n, int K,
                              std::vector<float>& h_cents_proj,
                              std::vector<float>& h_cents_full,
                              std::vector<int>&   h_assigns)
{
    h_cents_proj.resize((long long)K * d_proj_);

    float *d_x_proj, *d_dots;
    int   *d_assigns, *d_counts;
    float *d_cents, *d_norms;
    CUDA_CHECK(cudaMalloc(&d_x_proj,  (long long)n * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dots,    (long long)n * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assigns, (long long)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts,  (long long)K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cents,   (long long)K * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms,   (long long)K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_proj, h_x_proj, (long long)n * d_proj_ * sizeof(float), cudaMemcpyHostToDevice));

    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> uni(0, n - 1);
        std::vector<int> idxs(K);
        for (int k = 0; k < K; k++) idxs[k] = uni(rng);
        std::vector<float> init_c((long long)K * d_proj_);
        for (int k = 0; k < K; k++)
            std::memcpy(init_c.data() + (long long)k * d_proj_,
                        h_x_proj + (long long)idxs[k] * d_proj_,
                        d_proj_ * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_cents, init_c.data(), (long long)K * d_proj_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    const float one = 1.f, zero = 0.f;
    for (int iter = 0; iter < km_iters_; iter++) {
        std::vector<float> h_norms(K);
        CUDA_CHECK(cudaMemcpy(h_cents_proj.data(), d_cents, (long long)K * d_proj_ * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = 0; k < K; k++) {
            float s = 0.f;
            for (int j = 0; j < d_proj_; j++) {
                float v = h_cents_proj[(long long)k * d_proj_ + j];
                s += v * v;
            }
            h_norms[k] = s;
        }
        CUDA_CHECK(cudaMemcpy(d_norms, h_norms.data(), K * sizeof(float), cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K, n, d_proj_, &one,
                                 d_cents, d_proj_,
                                 d_x_proj, d_proj_, &zero,
                                 d_dots, K));

        hblock_v17::launch_kmeans_assign(d_dots, d_norms, d_assigns, K, n, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_cents,  0, (long long)K * d_proj_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, K * sizeof(int)));
        hblock_v17::launch_kmeans_update(d_x_proj, d_assigns, d_cents, d_counts, n, d_proj_, K, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    h_cents_proj.resize((long long)K * d_proj_);
    h_assigns.resize(n);
    CUDA_CHECK(cudaMemcpy(h_cents_proj.data(), d_cents, (long long)K * d_proj_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_assigns.data(),    d_assigns, n * sizeof(int),                      cudaMemcpyDeviceToHost));

    h_cents_full.assign((long long)K * d_, 0.f);
    std::vector<int> cnt(K, 0);
    for (int i = 0; i < n; i++) {
        int c = h_assigns[i];
        cnt[c]++;
        for (int j = 0; j < d_; j++)
            h_cents_full[(long long)c * d_ + j] += h_x_full[(long long)i * d_ + j];
    }
    for (int k = 0; k < K; k++) {
        if (cnt[k] > 0)
            for (int j = 0; j < d_; j++)
                h_cents_full[(long long)k * d_ + j] /= (float)cnt[k];
    }

    cudaFree(d_x_proj); cudaFree(d_dots); cudaFree(d_assigns);
    cudaFree(d_counts); cudaFree(d_cents); cudaFree(d_norms);
}

void HBlockIndex::upload_cents(const std::vector<float>& h_proj, const std::vector<float>& h_full,
                                const std::vector<bool>& h_valid,
                                int K,
                                float*& d_proj_out, float*& d_full_out, float*& d_norms_out)
{
    std::vector<float> h_norms(K);
    for (int k = 0; k < K; k++) {
        if (!h_valid[k]) {
            h_norms[k] = 1e30f;  // empty centroid: INF distance keeps routing away
        } else {
            float s = 0.f;
            for (int j = 0; j < d_proj_; j++) {
                float v = h_proj[(long long)k * d_proj_ + j];
                s += v * v;
            }
            h_norms[k] = s;
        }
    }

    cudaFree(d_proj_out); cudaFree(d_full_out); cudaFree(d_norms_out);
    CUDA_CHECK(cudaMalloc(&d_proj_out,  (long long)K * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_full_out,  (long long)K * d_       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms_out, (long long)K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_proj_out,  h_proj.data(),   (long long)K * d_proj_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_full_out,  h_full.data(),   (long long)K * d_       * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norms_out, h_norms.data(),  (long long)K * sizeof(float),             cudaMemcpyHostToDevice));
}

// Local assignment kernel (identical to v17)
static __global__ void local_assign_kernel(
    const float* __restrict__ r_proj,
    const float* __restrict__ C_proj,
    const float* __restrict__ C_norms,
    const int*   __restrict__ outer_id,
    int*         assign,
    int n, int d_proj, int K_inner)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int og = outer_id[i];
    const float* r_i    = r_proj  + (long long)i  * d_proj;
    const float* c_base = C_proj  + (long long)og * K_inner * d_proj;
    const float* n_base = C_norms + (long long)og * K_inner;
    float bv = 1e30f; int bi = 0;
    for (int k = 0; k < K_inner; k++) {
        const float* c_k = c_base + (long long)k * d_proj;
        float dot = 0.f;
        for (int j = 0; j < d_proj; j++) dot += r_i[j] * c_k[j];
        float dist = n_base[k] - 2.f * dot;
        if (dist < bv) { bv = dist; bi = k; }
    }
    assign[i] = bi;
}

void HBlockIndex::train(const float* h_x, int n_train)
{
    auto T0 = std::chrono::high_resolution_clock::now();
    using Clock = std::chrono::high_resolution_clock;

    const int n_km  = std::min(n_train, 200000);

    printf("[v20 train] d=%d d_proj=%d K1=%d K2=%d K3=%d ck1=%d ck2=%d ck3=%d\n",
           d_, d_proj_, K1_, K2_, K3_, ck1_, ck2_, ck3_);
    printf("  n_train=%d n_km=%d km_iters=%d\n", n_train, n_km, km_iters_);

    // Step 1: L1 JL projection + k-means
    std::vector<float> Pi1;
    init_jl_proj(d_, d_proj_, /*seed=*/42, Pi1);
    CUDA_CHECK(cudaMalloc(&d_Pi1_, (long long)d_proj_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi1_, Pi1.data(), (long long)d_proj_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

    printf("  [L1 JL+k-means] K1=%d iters=%d ... ", K1_, km_iters_); fflush(stdout);
    auto t1 = Clock::now();
    {
        float *d_x_gpu, *d_y_proj;
        CUDA_CHECK(cudaMalloc(&d_x_gpu,  (long long)n_km * d_      * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y_proj, (long long)n_km * d_proj_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x_gpu, h_x, (long long)n_km * d_ * sizeof(float), cudaMemcpyHostToDevice));

        const float one = 1.f, zero = 0.f;
        CUBLAS_CHECK(cublasSetStream(cublas_, nullptr));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_proj_, n_km, d_, &one,
                                 d_Pi1_, d_,
                                 d_x_gpu, d_, &zero,
                                 d_y_proj, d_proj_));

        std::vector<float> h_y_proj((long long)n_km * d_proj_);
        CUDA_CHECK(cudaMemcpy(h_y_proj.data(), d_y_proj,
                              (long long)n_km * d_proj_ * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_x_km((long long)n_km * d_);
        std::memcpy(h_x_km.data(), h_x, (long long)n_km * d_ * sizeof(float));

        cudaFree(d_x_gpu); cudaFree(d_y_proj);

        std::vector<float> h_c1_proj, h_c1_full;
        std::vector<int> h_assign1;
        gpu_kmeans(h_y_proj.data(), h_x_km.data(), n_km, K1_,
                   h_c1_proj, h_c1_full, h_assign1);

        // All L1 centroids are valid (k-means++ guarantees non-empty)
        std::vector<bool> c1_valid(K1_, true);
        upload_cents(h_c1_proj, h_c1_full, c1_valid, K1_,
                     d_route1_cents_proj_, d_route1_cents_full_, d_route1_norms_);

        auto t2 = Clock::now();
        printf("%.1f ms\n", Ms(t2 - t1).count());

        // Step 2: L2 hierarchical k-means
        printf("  [L2 hierarchical k-means] K2=%d per c1 ... ", K2_); fflush(stdout);
        auto t3 = Clock::now();

        std::vector<float> h_r1((long long)n_km * d_);
        for (int i = 0; i < n_km; i++) {
            int c1 = h_assign1[i];
            for (int j = 0; j < d_; j++)
                h_r1[(long long)i * d_ + j] = h_x_km[(long long)i * d_ + j]
                                              - h_c1_full[(long long)c1 * d_ + j];
        }

        std::vector<float> Pi2;
        init_jl_proj(d_, d_proj_, /*seed=*/43, Pi2);
        CUDA_CHECK(cudaMalloc(&d_Pi2_, (long long)d_proj_ * d_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Pi2_, Pi2.data(), (long long)d_proj_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

        float *d_r1_gpu, *d_r1_proj_gpu;
        CUDA_CHECK(cudaMalloc(&d_r1_gpu,      (long long)n_km * d_      * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_r1_proj_gpu, (long long)n_km * d_proj_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_r1_gpu, h_r1.data(), (long long)n_km * d_ * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_proj_, n_km, d_, &one,
                                 d_Pi2_, d_, d_r1_gpu, d_, &zero, d_r1_proj_gpu, d_proj_));
        std::vector<float> h_r1_proj((long long)n_km * d_proj_);
        CUDA_CHECK(cudaMemcpy(h_r1_proj.data(), d_r1_proj_gpu,
                              (long long)n_km * d_proj_ * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_r1_gpu); cudaFree(d_r1_proj_gpu);

        std::vector<float> h_all_c2_proj((long long)K1_ * K2_ * d_proj_, 0.f);
        std::vector<float> h_all_c2_full((long long)K1_ * K2_ * d_,      0.f);
        std::vector<int>   h_assign2(n_km, 0);
        std::vector<bool>  h_c2_valid((long long)K1_ * K2_, false);

        for (int c1 = 0; c1 < K1_; c1++) {
            std::vector<int> idx;
            for (int i = 0; i < n_km; i++) if (h_assign1[i] == c1) idx.push_back(i);
            int nc1 = (int)idx.size();
            if (nc1 == 0) continue;
            int K2e = std::min(K2_, nc1);

            std::vector<float> grp_proj((long long)nc1 * d_proj_);
            std::vector<float> grp_full((long long)nc1 * d_);
            for (int k = 0; k < nc1; k++) {
                std::memcpy(grp_proj.data() + (long long)k * d_proj_,
                            h_r1_proj.data() + (long long)idx[k] * d_proj_, d_proj_ * sizeof(float));
                std::memcpy(grp_full.data() + (long long)k * d_,
                            h_r1.data()      + (long long)idx[k] * d_,      d_       * sizeof(float));
            }

            std::vector<float> c2p, c2f; std::vector<int> a2;
            gpu_kmeans(grp_proj.data(), grp_full.data(), nc1, K2e, c2p, c2f, a2);
            c2p.resize((long long)K2_ * d_proj_, 0.f);
            c2f.resize((long long)K2_ * d_,      0.f);

            std::memcpy(h_all_c2_proj.data() + (long long)c1 * K2_ * d_proj_,
                        c2p.data(), (long long)K2_ * d_proj_ * sizeof(float));
            std::memcpy(h_all_c2_full.data() + (long long)c1 * K2_ * d_,
                        c2f.data(), (long long)K2_ * d_       * sizeof(float));
            for (int k = 0; k < nc1; k++) h_assign2[idx[k]] = a2[k];
            for (int k = 0; k < K2e; k++) h_c2_valid[c1 * K2_ + k] = true;
        }
        upload_cents(h_all_c2_proj, h_all_c2_full, h_c2_valid, K1_ * K2_,
                     d_route2_cents_proj_, d_route2_cents_full_, d_route2_norms_);

        auto t4 = Clock::now();
        printf("%.1f ms\n", Ms(t4 - t3).count());

        // Step 3: L3 hierarchical k-means
        printf("  [L3 hierarchical k-means] K3=%d per (c1,c2) ... ", K3_); fflush(stdout);
        auto t5 = Clock::now();

        std::vector<float> h_r2((long long)n_km * d_);
        for (int i = 0; i < n_km; i++) {
            int c1 = h_assign1[i], c2 = h_assign2[i];
            long long c12 = (long long)c1 * K2_ + c2;
            for (int j = 0; j < d_; j++)
                h_r2[(long long)i * d_ + j] = h_r1[(long long)i * d_ + j]
                                              - h_all_c2_full[c12 * d_ + j];
        }

        std::vector<float> Pi3;
        init_jl_proj(d_, d_proj_, /*seed=*/44, Pi3);
        CUDA_CHECK(cudaMalloc(&d_Pi3_, (long long)d_proj_ * d_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Pi3_, Pi3.data(), (long long)d_proj_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

        float *d_r2_gpu, *d_r2_proj_gpu;
        CUDA_CHECK(cudaMalloc(&d_r2_gpu,      (long long)n_km * d_      * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_r2_proj_gpu, (long long)n_km * d_proj_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_r2_gpu, h_r2.data(), (long long)n_km * d_ * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_proj_, n_km, d_, &one,
                                 d_Pi3_, d_, d_r2_gpu, d_, &zero, d_r2_proj_gpu, d_proj_));
        std::vector<float> h_r2_proj((long long)n_km * d_proj_);
        CUDA_CHECK(cudaMemcpy(h_r2_proj.data(), d_r2_proj_gpu,
                              (long long)n_km * d_proj_ * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_r2_gpu); cudaFree(d_r2_proj_gpu);

        long long n_c12 = (long long)K1_ * K2_;
        std::vector<float> h_all_c3_proj(n_c12 * K3_ * d_proj_, 0.f);
        std::vector<float> h_all_c3_full(n_c12 * K3_ * d_,      0.f);
        std::vector<int>   h_assign3(n_km, 0);
        std::vector<bool>  h_c3_valid(n_c12 * K3_, false);

        for (int c1 = 0; c1 < K1_; c1++) {
            for (int c2 = 0; c2 < K2_; c2++) {
                long long c12 = (long long)c1 * K2_ + c2;
                std::vector<int> idx;
                for (int i = 0; i < n_km; i++)
                    if (h_assign1[i] == c1 && h_assign2[i] == c2) idx.push_back(i);
                int nc12 = (int)idx.size();
                if (nc12 == 0) continue;
                int K3e = std::min(K3_, nc12);

                std::vector<float> grp_proj((long long)nc12 * d_proj_);
                std::vector<float> grp_full((long long)nc12 * d_);
                for (int k = 0; k < nc12; k++) {
                    std::memcpy(grp_proj.data() + (long long)k * d_proj_,
                                h_r2_proj.data() + (long long)idx[k] * d_proj_, d_proj_ * sizeof(float));
                    std::memcpy(grp_full.data() + (long long)k * d_,
                                h_r2.data()      + (long long)idx[k] * d_,      d_       * sizeof(float));
                }

                std::vector<float> c3p, c3f; std::vector<int> a3;
                gpu_kmeans(grp_proj.data(), grp_full.data(), nc12, K3e, c3p, c3f, a3);
                c3p.resize((long long)K3_ * d_proj_, 0.f);
                c3f.resize((long long)K3_ * d_,      0.f);

                std::memcpy(h_all_c3_proj.data() + (c12 * K3_) * d_proj_,
                            c3p.data(), (long long)K3_ * d_proj_ * sizeof(float));
                std::memcpy(h_all_c3_full.data() + (c12 * K3_) * d_,
                            c3f.data(), (long long)K3_ * d_       * sizeof(float));
                for (int k = 0; k < nc12; k++) h_assign3[idx[k]] = a3[k];
                for (int k = 0; k < K3e; k++) h_c3_valid[c12 * K3_ + k] = true;
            }
        }
        upload_cents(h_all_c3_proj, h_all_c3_full, h_c3_valid, n_c12 * K3_,
                     d_route3_cents_proj_, d_route3_cents_full_, d_route3_norms_);

        // Step 4: Fine PQ codebook
        double sum_sq = 0.0;
        for (int i = 0; i < n_km; i++) {
            int c1 = h_assign1[i], c2 = h_assign2[i], c3 = h_assign3[i];
            long long c123 = ((long long)c1 * K2_ + c2) * K3_ + c3;
            for (int j = 0; j < d_; j++) {
                float v = h_r2[(long long)i * d_ + j] - h_all_c3_full[c123 * d_ + j];
                sum_sq += (double)v * v;
            }
        }
        float sigma = (float)std::sqrt(sum_sq / (double)((long long)n_km * d_));
        auto t6 = Clock::now();
        printf("%.1f ms  sigma_r3=%.4f\n", Ms(t6 - t5).count(), sigma);

        std::vector<float> fine_c1d = analytical_fine_c1d(Kr_, sigma);
        CUDA_CHECK(cudaMalloc(&d_fine_c1d_, Kr_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_fine_c1d_, fine_c1d.data(), Kr_ * sizeof(float), cudaMemcpyHostToDevice));
    }

    auto T1 = std::chrono::high_resolution_clock::now();
    printf("[v20 train] total=%.1f ms\n", Ms(T1 - T0).count());
}

void HBlockIndex::add(const float* h_x, int n)
{
    if (!d_Pi1_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0) throw std::runtime_error("HBlock supports one add() call");
    printf("[v20 add] n=%d ...\n", n);

    const int BATCH = 8192;
    const float one = 1.f, zero = 0.f;

    float *d_x, *d_r1, *d_r2, *d_r3, *d_proj1, *d_proj2, *d_proj3;
    int *d_c1, *d_c2, *d_c3; uint8_t *d_fc;
    CUDA_CHECK(cudaMalloc(&d_x,     (long long)BATCH * d_      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj1, (long long)BATCH * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj2, (long long)BATCH * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj3, (long long)BATCH * d_proj_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r1,    (long long)BATCH * d_      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r2,    (long long)BATCH * d_      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r3,    (long long)BATCH * d_      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1,    (long long)BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c2,    (long long)BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c3,    (long long)BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fc,    (long long)BATCH * bpv_));
    float* d_dots1;
    CUDA_CHECK(cudaMalloc(&d_dots1, (long long)K1_ * BATCH * sizeof(float)));

    std::vector<int>     h_code1(n), h_code2(n), h_code3(n);
    std::vector<uint8_t> h_fc_all((long long)n * bpv_);

    CUBLAS_CHECK(cublasSetStream(cublas_, nullptr));

    for (int s = 0; s < n; s += BATCH) {
        int nb = std::min(BATCH, n - s);
        CUDA_CHECK(cudaMemcpy(d_x, h_x + (long long)s * d_,
                              (long long)nb * d_ * sizeof(float), cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_proj_, nb, d_, &one, d_Pi1_, d_, d_x, d_, &zero, d_proj1, d_proj_));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K1_, nb, d_proj_, &one,
                                 d_route1_cents_proj_, d_proj_, d_proj1, d_proj_, &zero, d_dots1, K1_));
        hblock_v17::launch_assign_from_dots(d_dots1, d_route1_norms_, d_c1, K1_, nb, nullptr);
        hblock_v17::launch_subtract_centroid(d_x, d_c1, d_route1_cents_full_, d_r1, nb, d_, nullptr);

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_proj_, nb, d_, &one, d_Pi2_, d_, d_r1, d_, &zero, d_proj2, d_proj_));
        local_assign_kernel<<<(nb + 255) / 256, 256>>>(
            d_proj2, d_route2_cents_proj_, d_route2_norms_, d_c1, d_c2, nb, d_proj_, K2_);
        {
            int* d_c12; CUDA_CHECK(cudaMalloc(&d_c12, nb * sizeof(int)));
            std::vector<int> h_c1b(nb), h_c2b(nb);
            CUDA_CHECK(cudaMemcpy(h_c1b.data(), d_c1, nb*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_c2b.data(), d_c2, nb*sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> h_c12(nb);
            for (int i = 0; i < nb; i++) h_c12[i] = h_c1b[i] * K2_ + h_c2b[i];
            CUDA_CHECK(cudaMemcpy(d_c12, h_c12.data(), nb*sizeof(int), cudaMemcpyHostToDevice));
            hblock_v17::launch_subtract_centroid(d_r1, d_c12, d_route2_cents_full_, d_r2, nb, d_, nullptr);
            cudaFree(d_c12);
        }

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 d_proj_, nb, d_, &one, d_Pi3_, d_, d_r2, d_, &zero, d_proj3, d_proj_));
        {
            int* d_c12; CUDA_CHECK(cudaMalloc(&d_c12, nb * sizeof(int)));
            std::vector<int> h_c1b(nb), h_c2b(nb);
            CUDA_CHECK(cudaMemcpy(h_c1b.data(), d_c1, nb*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_c2b.data(), d_c2, nb*sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> h_c12(nb);
            for (int i = 0; i < nb; i++) h_c12[i] = h_c1b[i] * K2_ + h_c2b[i];
            CUDA_CHECK(cudaMemcpy(d_c12, h_c12.data(), nb*sizeof(int), cudaMemcpyHostToDevice));

            local_assign_kernel<<<(nb + 255) / 256, 256>>>(
                d_proj3, d_route3_cents_proj_, d_route3_norms_, d_c12, d_c3, nb, d_proj_, K3_);

            std::vector<int> h_c3b(nb), h_c123(nb);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_c3b.data(), d_c3, nb*sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < nb; i++) h_c123[i] = h_c12[i] * K3_ + h_c3b[i];
            int* d_c123; CUDA_CHECK(cudaMalloc(&d_c123, nb*sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_c123, h_c123.data(), nb*sizeof(int), cudaMemcpyHostToDevice));
            hblock_v17::launch_subtract_centroid(d_r2, d_c123, d_route3_cents_full_, d_r3, nb, d_, nullptr);
            cudaFree(d_c12); cudaFree(d_c123);
        }

        hblock_v17::launch_fine_encode(d_r3, d_fine_c1d_, d_fc, nb, d_, Kr_, Br_, bpv_, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_code1.data() + s, d_c1, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_code2.data() + s, d_c2, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_code3.data() + s, d_c3, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fc_all.data() + (long long)s * bpv_,
                              d_fc, (long long)nb * bpv_, cudaMemcpyDeviceToHost));
    }
    cudaFree(d_x); cudaFree(d_proj1); cudaFree(d_proj2); cudaFree(d_proj3);
    cudaFree(d_r1); cudaFree(d_r2); cudaFree(d_r3);
    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_c3); cudaFree(d_fc); cudaFree(d_dots1);

    long long K2K3 = (long long)K2_ * K3_;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        long long ka = (long long)h_code1[a] * K2K3 + h_code2[a] * K3_ + h_code3[a];
        long long kb = (long long)h_code1[b] * K2K3 + h_code2[b] * K3_ + h_code3[b];
        return ka < kb;
    });

    long long n_cells = (long long)K1_ * K2_ * K3_;
    std::vector<int> pair_cnt(n_cells, 0);
    for (int i = 0, j; i < n; i = j) {
        int c1 = h_code1[order[i]], c2 = h_code2[order[i]], c3 = h_code3[order[i]];
        for (j = i; j < n && h_code1[order[j]] == c1
                           && h_code2[order[j]] == c2
                           && h_code3[order[j]] == c3; ++j) {}
        pair_cnt[(long long)c1 * K2K3 + c2 * K3_ + c3] = (j - i + leaf_size_ - 1) / leaf_size_;
    }

    std::vector<int> pair_start(n_cells, 0);
    int total_blocks = 0;
    for (long long p = 0; p < n_cells; ++p) { pair_start[p] = total_blocks; total_blocks += pair_cnt[p]; }
    n_leaf_blocks_ = total_blocks;
    max_blk_per_cell_ = *std::max_element(pair_cnt.begin(), pair_cnt.end());

    std::vector<uint8_t> h_leaf_codes((long long)total_blocks * bpv_ * leaf_size_, 0);
    std::vector<int>     h_leaf_ids  ((long long)total_blocks * leaf_size_, -1);
    std::vector<int>     h_leaf_sizes(total_blocks, 0);

    for (int i = 0, j; i < n; i = j) {
        int c1 = h_code1[order[i]], c2 = h_code2[order[i]], c3 = h_code3[order[i]];
        for (j = i; j < n && h_code1[order[j]] == c1
                           && h_code2[order[j]] == c2
                           && h_code3[order[j]] == c3; ++j) {}
        int base_blk = pair_start[(long long)c1 * K2K3 + c2 * K3_ + c3];
        for (int vi = i; vi < j; ++vi) {
            int in_blk = vi - i, blk = base_blk + in_blk / leaf_size_, pos = in_blk % leaf_size_;
            int oid = order[vi];
            h_leaf_ids[(long long)blk * leaf_size_ + pos] = oid;
            const uint8_t* src = h_fc_all.data() + (long long)oid * bpv_;
            uint8_t* dst_base  = h_leaf_codes.data() + (long long)blk * bpv_ * leaf_size_;
            for (int b = 0; b < bpv_; b++)
                dst_base[(long long)b * leaf_size_ + pos] = src[b];
            h_leaf_sizes[blk] = std::max(h_leaf_sizes[blk], pos + 1);
        }
    }

    CUDA_CHECK(cudaMalloc(&d_pair_blk_start_, n_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_blk_count_, n_cells * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_start_, pair_start.data(), n_cells * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_count_, pair_cnt.data(),   n_cells * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_leaf_codes_, (long long)total_blocks * bpv_ * leaf_size_));
    CUDA_CHECK(cudaMalloc(&d_leaf_ids_,   (long long)total_blocks * leaf_size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leaf_sizes_, total_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_leaf_codes_, h_leaf_codes.data(), (long long)total_blocks*bpv_*leaf_size_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_ids_,   h_leaf_ids.data(),   (long long)total_blocks*leaf_size_*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_sizes_, h_leaf_sizes.data(), total_blocks*sizeof(int), cudaMemcpyHostToDevice));

    printf("  Uploading base vecs %.2f GB ... ", (double)n * d_ * 4 / 1e9); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&d_base_vecs_, (long long)n * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_base_vecs_, h_x, (long long)n * d_ * sizeof(float), cudaMemcpyHostToDevice));
    printf("done\n");

    ntotal_ = n;
    printf("  Built %d leaf blocks (leaf_size=%d, K1=%d, K2=%d)\n",
           total_blocks, leaf_size_, K1_, K2_);

    alloc_workspace();
}

void HBlockIndex::alloc_workspace()
{
    const int B            = batch_size_;
    const int total_ck     = ck1_ * ck2_ * ck3_;
    const int max_leaf_sel = total_ck * std::max(max_blk_per_cell_, 1);
    const int max_pairs    = B * max_leaf_sel;
    const int R         = rerank_r_;

    if (ws_.stream) { cublasSetStream(cublas_, nullptr); cudaStreamDestroy(ws_.stream); }
    CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));

#define FH(p) do { if (ws_.p) { cudaFreeHost(ws_.p); ws_.p = nullptr; } } while(0)
#define FD(p) do { if (ws_.p) { cudaFree(ws_.p);     ws_.p = nullptr; } } while(0)

    FH(h_q_pinned); FH(h_leaf_cnt); FH(h_final_dists); FH(h_final_ids);

    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned,    (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_leaf_cnt,    (long long)B * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_final_dists, (long long)B * K_MAX * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_final_ids,   (long long)B * K_MAX * sizeof(int)));

    FD(d_q_batch); FD(d_q_proj1); FD(d_dots1); FD(d_top1_ids);
    FD(d_r1_beam); FD(d_top2_beam); FD(d_top3_beam); FD(d_q_r3);
    FD(d_leaf_sel); FD(d_leaf_cnt); FD(d_lut_fine);

    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,    (long long)B * d_               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_proj1,    (long long)B * d_proj_           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots1,      (long long)B * K1_               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top1_ids,   (long long)B * ck1_              * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_r1_beam,    (long long)B * ck1_ * d_         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top2_beam,  (long long)B * ck1_ * ck2_       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top3_beam,  (long long)B * ck1_ * ck2_ * ck3_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r3,       (long long)B * d_               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_sel,   (long long)B * max_leaf_sel      * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_cnt,   (long long)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_fine,   (long long)B * d_ * Kr_          * sizeof(float)));

    FD(d_query_offsets);
    FD(d_pair_leaf_a); FD(d_pair_qid_a);
    FD(d_pair_leaf_b); FD(d_pair_qid_b);
    CUDA_CHECK(cudaMalloc(&ws_.d_query_offsets, (long long)(B + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_a,   (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_a,    (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_b,   (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_b,    (long long)max_pairs * sizeof(int)));

    FD(d_out_dists); FD(d_out_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_out_dists, (long long)max_pairs * TOP_P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_out_ids,   (long long)max_pairs * TOP_P * sizeof(int)));

    FD(d_pq_dists); FD(d_pq_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_pq_dists, (long long)B * K_MAX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pq_ids,   (long long)B * K_MAX * sizeof(int)));

    FD(d_cand_vecs); FD(d_exact_dists);
    CUDA_CHECK(cudaMalloc(&ws_.d_cand_vecs,   (long long)B * R * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_exact_dists, (long long)B * R * sizeof(float)));

    FD(d_final_dists); FD(d_final_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_final_dists, (long long)B * K_MAX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_ids,   (long long)B * K_MAX * sizeof(int)));

    size_t scan_bytes = 0, sort_leaf_bytes = 0, sort_qid_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (int*)nullptr, (int*)nullptr, B);
    cub::DeviceRadixSort::SortPairs(nullptr, sort_leaf_bytes,
        (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, max_pairs, 0, 20);
    cub::DeviceRadixSort::SortPairs(nullptr, sort_qid_bytes,
        (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, max_pairs, 0, 10);
    ws_.cub_bytes = std::max({scan_bytes, sort_leaf_bytes, sort_qid_bytes});
    FD(d_cub_tmp);
    CUDA_CHECK(cudaMalloc(&ws_.d_cub_tmp, ws_.cub_bytes));

    ws_.batch_cap    = B;
    ws_.max_pairs    = max_pairs;
    ws_.max_leaf_sel = max_leaf_sel;
    ws_.rerank_r     = R;
    ws_.d_proj       = d_proj_;

#undef FH
#undef FD
}

void HBlockIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_ids) const
{
    if (ntotal_ == 0) throw std::runtime_error("HBlock index is empty");
    if (k > K_MAX)    throw std::runtime_error("k exceeds K_MAX");
    if (k > rerank_r_) throw std::runtime_error("k must be <= rerank_r");

    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    route_queries_v20(cublas_,
                      d_Pi1_, d_Pi2_, d_Pi3_,
                      d_route1_cents_proj_, d_route1_cents_full_, d_route1_norms_,
                      d_route2_cents_proj_, d_route2_cents_full_, d_route2_norms_,
                      d_route3_cents_proj_, d_route3_cents_full_, d_route3_norms_,
                      d_fine_c1d_, d_pair_blk_start_, d_pair_blk_count_,
                      h_q, nq, d_, d_proj_, K1_, K2_, K3_, Kr_, Br_,
                      ck1_, ck2_, ck3_, batch_size_, ws_);
    auto t1 = Clock::now();

    int n_pairs = 0;
    for (int qi = 0; qi < nq; qi++) n_pairs += ws_.h_leaf_cnt[qi];
    auto t2 = Clock::now();

    gpu_build_and_sort_pairs_v20(nq, n_pairs, n_leaf_blocks_, ws_.max_leaf_sel, ws_);
    auto t3 = Clock::now();

    cudaStream_t s = ws_.stream;
    launch_leaf_flat_v20(
        ws_.d_pair_leaf_b, ws_.d_pair_qid_b,
        d_leaf_codes_, d_leaf_ids_, d_leaf_sizes_, ws_.d_lut_fine,
        ws_.d_out_dists, ws_.d_out_ids,
        n_pairs, d_, Kr_, Br_, bpv_, leaf_size_, s);
    auto t4 = Clock::now();

    gpu_merge_pq_topk_v20(nq, n_pairs, rerank_r_, ws_);
    auto t5 = Clock::now();

    launch_gather_vecs_v20(d_base_vecs_, ws_.d_pq_ids, ws_.d_cand_vecs,
                           nq, rerank_r_, d_, s);
    auto t6 = Clock::now();

    launch_exact_rerank_v20(ws_.d_q_batch, ws_.d_cand_vecs, ws_.d_pq_ids,
                             ws_.d_exact_dists, ws_.d_final_dists, ws_.d_final_ids,
                             nq, rerank_r_, d_, k, s);
    auto t7 = Clock::now();

    CUDA_CHECK(cudaMemcpyAsync(ws_.h_final_dists, ws_.d_final_dists,
                               (long long)nq * k * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ws_.h_final_ids, ws_.d_final_ids,
                               (long long)nq * k * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto t8 = Clock::now();

    std::vector<int> idx_buf(k);
    for (int qi = 0; qi < nq; qi++) {
        const float* fd = ws_.h_final_dists + (long long)qi * k;
        const int*   fi = ws_.h_final_ids   + (long long)qi * k;
        std::iota(idx_buf.begin(), idx_buf.end(), 0);
        int valid = 0;
        for (int i = 0; i < k; i++) if (fi[i] >= 0) valid++;
        int take = std::min(valid, k);
        std::partial_sort(idx_buf.begin(), idx_buf.begin() + take, idx_buf.end(),
                          [&](int a, int b){ return fd[a] < fd[b]; });
        for (int r = 0; r < k; r++) {
            h_dists[qi * k + r] = (r < take) ? fd[idx_buf[r]] : 1e30f;
            h_ids  [qi * k + r] = (r < take) ? fi[idx_buf[r]] : -1;
        }
    }
    auto t9 = Clock::now();

    printf("  [v20] Route=%.2fms  PairCount=%.2fms  GPUSort=%.2fms (%d pairs)"
           "  LeafPQ=%.2fms  MergePQ=%.2fms  Gather=%.2fms"
           "  Rerank=%.2fms  D2H=%.2fms  Extract=%.2fms  Total=%.2fms\n",
           Ms(t1-t0).count(), Ms(t2-t1).count(), Ms(t3-t2).count(), n_pairs,
           Ms(t4-t3).count(), Ms(t5-t4).count(), Ms(t6-t5).count(),
           Ms(t7-t6).count(), Ms(t8-t7).count(), Ms(t9-t8).count(),
           Ms(t9-t0).count());
}

} // namespace hblock_v20
