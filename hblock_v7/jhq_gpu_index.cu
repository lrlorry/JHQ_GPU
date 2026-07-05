#include "hblock_v7/jhq_gpu_index.cuh"
#include "hblock_v7/encode.cuh"
#include "hblock_v7/search.cuh"
#include "cpu/erfinv.h"
#include "common/cuda_utils.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace hblock_v7 {

static std::vector<float>
analytical_route_cents(int K, int d, int dim_offset, float sigma)
{
    int M_r = 0; while ((1 << M_r) < K) ++M_r;
    const float cv = sigma * std::sqrt(2.0f / float(M_PI));
    std::vector<float> cents((long long)K * d, 0.f);
    for (int c = 0; c < K; ++c) {
        float* row = cents.data() + (long long)c * d;
        for (int m = 0; m < M_r; ++m)
            row[dim_offset + m] = ((c >> m) & 1) ? +cv : -cv;
    }
    return cents;
}

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
    : d_(d), Kr_(p.Kr), Br_(p.Br),
      bpv_((d * p.Br + 7) / 8),
      leaf_size_(p.leaf_size),
      K1_(p.K1), K2_(p.K2),
      ck1_(p.ck1), ck2_(p.ck2), ck3_(p.ck3),
      batch_size_(p.batch_size),
      jl_(d, p.seed)
{
    if (d <= 0)           throw std::invalid_argument("d must be positive");
    if (p.K1 <= 0 || (p.K1 & (p.K1-1)) != 0)
        throw std::invalid_argument("K1 must be a positive power of 2");
    if (p.K2 <= 0 || (p.K2 & (p.K2-1)) != 0)
        throw std::invalid_argument("K2 must be a positive power of 2");
    if (p.leaf_size <= 0) throw std::invalid_argument("leaf_size must be positive");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    CUBLAS_CHECK(cublasCreate(&cublas_));
}

HBlockIndex::~HBlockIndex() {
    if (ws_.stream) cudaStreamDestroy(ws_.stream);
    cublasDestroy(cublas_);
    cudaFree(d_Pi_);
    cudaFree(d_route1_cents_); cudaFree(d_route1_norms_);
    cudaFree(d_route2_cents_); cudaFree(d_route2_norms_);
    cudaFree(d_fine_c1d_);
    cudaFree(d_pair_blk_start_); cudaFree(d_pair_blk_count_);
    cudaFree(d_leaf_codes_); cudaFree(d_leaf_ids_); cudaFree(d_leaf_sizes_);
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch);  cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);     cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);    cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
    cudaFree(ws_.d_fine_dists); cudaFree(ws_.d_fine_ids);
    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);
    cudaFree(ws_.d_pair_keys);   cudaFree(ws_.d_pair_vals);
    cudaFree(ws_.d_pair_keys_s); cudaFree(ws_.d_pair_vals_s);
    cudaFree(ws_.d_cub_tmp);
    // v7 RLE buffers
    cudaFree(ws_.d_unique_leaf_ids); cudaFree(ws_.d_run_counts);
    cudaFree(ws_.d_run_starts);      cudaFree(ws_.d_num_unique);
    cudaFree(ws_.d_cub_tmp2);
}

void HBlockIndex::upload_centroids(std::vector<float>& cents,
                                    float*& d_cents, float*& d_norms, int K)
{
    std::vector<float> norms(K, 0.f);
    for (int k = 0; k < K; ++k) {
        double s = 0.0;
        const float* ck = cents.data() + (long long)k * d_;
        for (int j = 0; j < d_; ++j) s += (double)ck[j] * ck[j];
        norms[k] = (float)s;
    }
    cudaFree(d_cents); cudaFree(d_norms);
    CUDA_CHECK(cudaMalloc(&d_cents, (long long)K * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norms, (long long)K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cents, cents.data(),
                          (long long)K * d_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norms, norms.data(),
                          (long long)K * sizeof(float), cudaMemcpyHostToDevice));
}

void HBlockIndex::train(const float* h_x, int n_train) {
    int M_r1 = 0; while ((1 << M_r1) < K1_) ++M_r1;
    int M_r2 = 0; while ((1 << M_r2) < K2_) ++M_r2;
    if (M_r1 + M_r2 >= d_)
        throw std::invalid_argument("log2(K1)+log2(K2) must be < d");

    jl_.estimate_sigma(h_x, n_train);
    float sigma = jl_.sigma();
    printf("  sigma=%.4f  L1: K1=%d → M_r1=%d dims  L2: K2=%d → M_r2=%d dims\n",
           sigma, K1_, M_r1, K2_, M_r2);

    CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(),
                          (long long)d_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

    h_route1_ = analytical_route_cents(K1_, d_, 0,    sigma);
    h_route2_ = analytical_route_cents(K2_, d_, M_r1, sigma);
    upload_centroids(h_route1_, d_route1_cents_, d_route1_norms_, K1_);
    upload_centroids(h_route2_, d_route2_cents_, d_route2_norms_, K2_);

    fine_c1d_ = analytical_fine_c1d(Kr_, sigma);
    CUDA_CHECK(cudaMalloc(&d_fine_c1d_, Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fine_c1d_, fine_c1d_.data(),
                          Kr_ * sizeof(float), cudaMemcpyHostToDevice));
}

void HBlockIndex::add(const float* h_x, int n) {
    if (!d_Pi_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0) throw std::runtime_error("HBlock currently supports one add() call");

    const int BATCH = 8192;
    const float one = 1.f, zero = 0.f;
    float *d_x, *d_y, *d_r1, *d_r2, *d_dots;
    int *d_c1, *d_c2; uint8_t *d_fc;
    CUDA_CHECK(cudaMalloc(&d_x,    (long long)BATCH * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,    (long long)BATCH * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r1,   (long long)BATCH * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r2,   (long long)BATCH * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1,   (long long)BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c2,   (long long)BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fc,   (long long)BATCH * bpv_));
    CUDA_CHECK(cudaMalloc(&d_dots, (long long)std::max(K1_, K2_) * BATCH * sizeof(float)));

    std::vector<int>     h_code1(n), h_code2(n);
    std::vector<uint8_t> h_fc_all((long long)n * bpv_);

    for (int s = 0; s < n; s += BATCH) {
        int nb = std::min(BATCH, n - s);
        CUDA_CHECK(cudaMemcpy(d_x, h_x + (long long)s * d_,
                              (long long)nb * d_ * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 d_, nb, d_, &one, d_Pi_, d_, d_x, d_, &zero, d_y, d_));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K1_, nb, d_, &one, d_route1_cents_, d_, d_y, d_, &zero, d_dots, K1_));
        launch_assign_from_dots(d_dots, d_route1_norms_, d_c1, K1_, nb, nullptr);
        launch_subtract_centroid(d_y, d_c1, d_route1_cents_, d_r1, nb, d_, nullptr);
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K2_, nb, d_, &one, d_route2_cents_, d_, d_r1, d_, &zero, d_dots, K2_));
        launch_assign_from_dots(d_dots, d_route2_norms_, d_c2, K2_, nb, nullptr);
        launch_subtract_centroid(d_r1, d_c2, d_route2_cents_, d_r2, nb, d_, nullptr);
        launch_fine_encode(d_r2, d_fine_c1d_, d_fc, nb, d_, Kr_, Br_, bpv_, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_code1.data() + s, d_c1, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_code2.data() + s, d_c2, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fc_all.data() + (long long)s * bpv_,
                              d_fc, (long long)nb * bpv_, cudaMemcpyDeviceToHost));
    }
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_r1); cudaFree(d_r2);
    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_fc); cudaFree(d_dots);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        long long ka = (long long)h_code1[a] * K2_ + h_code2[a];
        long long kb = (long long)h_code1[b] * K2_ + h_code2[b];
        return ka < kb;
    });

    std::vector<int> pair_cnt(K1_ * K2_, 0);
    {
        int i = 0;
        while (i < n) {
            int c1 = h_code1[order[i]], c2 = h_code2[order[i]], j = i;
            while (j < n && h_code1[order[j]] == c1 && h_code2[order[j]] == c2) ++j;
            pair_cnt[c1 * K2_ + c2] = (j - i + leaf_size_ - 1) / leaf_size_;
            i = j;
        }
    }

    std::vector<int> pair_start(K1_ * K2_, 0);
    int total_blocks = 0;
    for (int p = 0; p < K1_ * K2_; ++p) {
        pair_start[p] = total_blocks;
        total_blocks  += pair_cnt[p];
    }
    n_leaf_blocks_ = total_blocks;

    std::vector<uint8_t> h_leaf_codes((long long)total_blocks * leaf_size_ * bpv_, 0);
    std::vector<int>     h_leaf_ids  ((long long)total_blocks * leaf_size_, -1);
    std::vector<int>     h_leaf_sizes(total_blocks, 0);

    {
        int i = 0;
        while (i < n) {
            int c1 = h_code1[order[i]], c2 = h_code2[order[i]], j = i;
            while (j < n && h_code1[order[j]] == c1 && h_code2[order[j]] == c2) ++j;
            int base_blk = pair_start[c1 * K2_ + c2];
            for (int vi = i; vi < j; ++vi) {
                int in_blk = vi - i, blk_idx = base_blk + in_blk / leaf_size_;
                int pos = in_blk % leaf_size_, orig_id = order[vi];
                h_leaf_ids[(long long)blk_idx * leaf_size_ + pos] = orig_id;
                std::memcpy(h_leaf_codes.data() + ((long long)blk_idx * leaf_size_ + pos) * bpv_,
                            h_fc_all.data()     + (long long)orig_id * bpv_, bpv_);
                h_leaf_sizes[blk_idx] = std::max(h_leaf_sizes[blk_idx], pos + 1);
            }
            i = j;
        }
    }

    CUDA_CHECK(cudaMalloc(&d_pair_blk_start_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_blk_count_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_start_, pair_start.data(),
                          (long long)K1_ * K2_ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_count_, pair_cnt.data(),
                          (long long)K1_ * K2_ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_leaf_codes_, (long long)total_blocks * leaf_size_ * bpv_));
    CUDA_CHECK(cudaMalloc(&d_leaf_ids_,   (long long)total_blocks * leaf_size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leaf_sizes_, (long long)total_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_leaf_codes_, h_leaf_codes.data(),
                          (long long)total_blocks * leaf_size_ * bpv_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_ids_,   h_leaf_ids.data(),
                          (long long)total_blocks * leaf_size_ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_sizes_, h_leaf_sizes.data(),
                          (long long)total_blocks * sizeof(int), cudaMemcpyHostToDevice));

    ntotal_ = n;
    printf("  Built %d leaf blocks (leaf_size=%d, K1=%d, K2=%d)\n",
           total_blocks, leaf_size_, K1_, K2_);

    alloc_workspace();
}

void HBlockIndex::alloc_workspace() {
    const int B       = batch_size_;
    const int nc      = ck3_ * leaf_size_;
    const int n_pairs = B * ck3_;

    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch);  cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);     cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);    cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
    cudaFree(ws_.d_fine_dists); cudaFree(ws_.d_fine_ids);
    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);
    cudaFree(ws_.d_pair_keys);   cudaFree(ws_.d_pair_vals);
    cudaFree(ws_.d_pair_keys_s); cudaFree(ws_.d_pair_vals_s);
    cudaFree(ws_.d_cub_tmp);
    cudaFree(ws_.d_unique_leaf_ids); cudaFree(ws_.d_run_counts);
    cudaFree(ws_.d_run_starts);      cudaFree(ws_.d_num_unique);
    cudaFree(ws_.d_cub_tmp2);

    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned,    (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,          (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_rot,            (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r1,             (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r2,             (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots1,            (long long)B * K1_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots2,            (long long)B * K2_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top1_ids,         (long long)B * ck1_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top2_ids,         (long long)B * ck2_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_sel,         (long long)B * ck3_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_cnt,         (long long)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_fine,         (long long)B * d_ * Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_fine_dists,       (long long)B * nc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_fine_ids,         (long long)B * nc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_dists,      (long long)B * 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_ids,        (long long)B * 1024 * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&ws_.d_pair_keys,   (long long)n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_vals,   (long long)n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_keys_s, (long long)n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_vals_s, (long long)n_pairs * sizeof(int)));

    // CUB sort temp
    cub::DeviceRadixSort::SortPairs(
        nullptr, ws_.cub_bytes,
        (const int*)nullptr, (int*)nullptr,
        (const int*)nullptr, (int*)nullptr,
        n_pairs, 0, 30);
    CUDA_CHECK(cudaMalloc(&ws_.d_cub_tmp, ws_.cub_bytes));

    // v7 RLE buffers
    CUDA_CHECK(cudaMalloc(&ws_.d_unique_leaf_ids, (long long)n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_run_counts,      (long long)n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_run_starts,      (long long)n_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_num_unique,      sizeof(int)));

    // CUB temp for RLE and scan (reuse same buffer)
    size_t rle_bytes = 0, scan_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        nullptr, rle_bytes,
        (const int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, n_pairs);
    cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes,
        (const int*)nullptr, (int*)nullptr, n_pairs);
    ws_.cub_bytes2 = std::max(rle_bytes, scan_bytes);
    CUDA_CHECK(cudaMalloc(&ws_.d_cub_tmp2, ws_.cub_bytes2));

    ws_.batch_cap = B;
    if (!ws_.stream) CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));
}

void HBlockIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_ids) const
{
    if (ntotal_ == 0) throw std::runtime_error("HBlock index is empty");
    search_hblock(cublas_,
                  d_Pi_,
                  d_route1_cents_, d_route1_norms_,
                  d_route2_cents_, d_route2_norms_,
                  d_fine_c1d_,
                  d_pair_blk_start_, d_pair_blk_count_,
                  d_leaf_codes_, d_leaf_ids_, d_leaf_sizes_,
                  h_q,
                  nq, d_, K1_, K2_, Kr_, Br_, bpv_,
                  leaf_size_, ck1_, ck2_, ck3_, k,
                  batch_size_, n_leaf_blocks_,
                  ws_,
                  h_dists, h_ids);
}

} // namespace hblock_v7
