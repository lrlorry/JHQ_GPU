#include "hblock_v14/jhq_gpu_index.cuh"
#include "hblock_v14/encode.cuh"
#include "hblock_v14/search.cuh"
#include "cpu/erfinv.h"
#include "common/cuda_utils.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

using Ms = std::chrono::duration<double, std::milli>;

namespace hblock_v14 {

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
        throw std::invalid_argument("K1 must be power of 2");
    if (p.K2 <= 0 || (p.K2 & (p.K2-1)) != 0)
        throw std::invalid_argument("K2 must be power of 2");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    CUBLAS_CHECK(cublasCreate(&cublas_));
}

HBlockIndex::~HBlockIndex() {
    if (ws_.stream) cudaStreamDestroy(ws_.stream);
    if (ws_.h_leaf_cnt)    cudaFreeHost(ws_.h_leaf_cnt);
    if (ws_.h_final_dists) cudaFreeHost(ws_.h_final_dists);
    if (ws_.h_final_ids)   cudaFreeHost(ws_.h_final_ids);
    cudaFree(ws_.d_query_offsets);
    cudaFree(ws_.d_pair_leaf_a); cudaFree(ws_.d_pair_qid_a);
    cudaFree(ws_.d_pair_leaf_b); cudaFree(ws_.d_pair_qid_b);
    cudaFree(ws_.d_out_dists);   cudaFree(ws_.d_out_ids);
    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);
    cudaFree(ws_.d_cub_tmp);
    cublasDestroy(cublas_);
    cudaFree(d_Pi_);
    cudaFree(d_route1_cents_); cudaFree(d_route1_norms_);
    cudaFree(d_route2_cents_); cudaFree(d_route2_norms_);
    cudaFree(d_fine_c1d_);
    cudaFree(d_pair_blk_start_); cudaFree(d_pair_blk_count_);
    cudaFree(d_leaf_codes_); cudaFree(d_leaf_ids_); cudaFree(d_leaf_sizes_);
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch); cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);    cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);   cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
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
    CUDA_CHECK(cudaMemcpy(d_cents, cents.data(), (long long)K * d_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norms, norms.data(), (long long)K * sizeof(float), cudaMemcpyHostToDevice));
}

void HBlockIndex::train(const float* h_x, int n_train) {
    int M_r1 = 0; while ((1 << M_r1) < K1_) ++M_r1;
    int M_r2 = 0; while ((1 << M_r2) < K2_) ++M_r2;
    if (M_r1 + M_r2 >= d_) throw std::invalid_argument("log2(K1)+log2(K2) must be < d");

    jl_.estimate_sigma(h_x, n_train);
    float sigma = jl_.sigma();
    printf("  sigma=%.4f  L1: K1=%d M_r1=%d  L2: K2=%d M_r2=%d\n",
           sigma, K1_, M_r1, K2_, M_r2);

    CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(), (long long)d_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

    h_route1_ = analytical_route_cents(K1_, d_, 0,    sigma);
    h_route2_ = analytical_route_cents(K2_, d_, M_r1, sigma);
    upload_centroids(h_route1_, d_route1_cents_, d_route1_norms_, K1_);
    upload_centroids(h_route2_, d_route2_cents_, d_route2_norms_, K2_);

    fine_c1d_ = analytical_fine_c1d(Kr_, sigma);
    CUDA_CHECK(cudaMalloc(&d_fine_c1d_, Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fine_c1d_, fine_c1d_.data(), Kr_ * sizeof(float), cudaMemcpyHostToDevice));
}

void HBlockIndex::add(const float* h_x, int n) {
    if (!d_Pi_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0) throw std::runtime_error("HBlock supports one add() call");

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
        return (long long)h_code1[a] * K2_ + h_code2[a]
             < (long long)h_code1[b] * K2_ + h_code2[b];
    });

    std::vector<int> pair_cnt(K1_ * K2_, 0);
    for (int i = 0, j; i < n; i = j) {
        int c1 = h_code1[order[i]], c2 = h_code2[order[i]];
        for (j = i; j < n && h_code1[order[j]] == c1 && h_code2[order[j]] == c2; ++j) {}
        pair_cnt[c1 * K2_ + c2] = (j - i + leaf_size_ - 1) / leaf_size_;
    }

    std::vector<int> pair_start(K1_ * K2_, 0);
    int total_blocks = 0;
    for (int p = 0; p < K1_ * K2_; ++p) { pair_start[p] = total_blocks; total_blocks += pair_cnt[p]; }
    n_leaf_blocks_ = total_blocks;

    // Transposed code layout [blk][bpv][leaf_size] for coalesced access (v13+)
    std::vector<uint8_t> h_leaf_codes((long long)total_blocks * bpv_ * leaf_size_, 0);
    std::vector<int>     h_leaf_ids  ((long long)total_blocks * leaf_size_, -1);
    std::vector<int>     h_leaf_sizes(total_blocks, 0);

    for (int i = 0, j; i < n; i = j) {
        int c1 = h_code1[order[i]], c2 = h_code2[order[i]];
        for (j = i; j < n && h_code1[order[j]] == c1 && h_code2[order[j]] == c2; ++j) {}
        int base_blk = pair_start[c1 * K2_ + c2];
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

    CUDA_CHECK(cudaMalloc(&d_pair_blk_start_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_blk_count_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_start_, pair_start.data(), (long long)K1_*K2_*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_count_, pair_cnt.data(),   (long long)K1_*K2_*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_leaf_codes_, (long long)total_blocks * bpv_ * leaf_size_));
    CUDA_CHECK(cudaMalloc(&d_leaf_ids_,   (long long)total_blocks * leaf_size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leaf_sizes_, (long long)total_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_leaf_codes_, h_leaf_codes.data(), (long long)total_blocks*bpv_*leaf_size_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_ids_,   h_leaf_ids.data(),   (long long)total_blocks*leaf_size_*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_sizes_, h_leaf_sizes.data(), (long long)total_blocks*sizeof(int), cudaMemcpyHostToDevice));

    ntotal_ = n;
    printf("  Built %d leaf blocks (leaf_size=%d, K1=%d, K2=%d)\n",
           total_blocks, leaf_size_, K1_, K2_);
    alloc_workspace();
}

void HBlockIndex::alloc_workspace() {
    const int B         = batch_size_;
    const int max_pairs = B * ck3_;

    if (ws_.stream) cudaStreamDestroy(ws_.stream);
    CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));

    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    if (ws_.h_leaf_cnt) cudaFreeHost(ws_.h_leaf_cnt);
    cudaFree(ws_.d_q_batch); cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);    cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);   cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);

    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned, (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_leaf_cnt, (long long)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,   (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_rot,     (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r1,      (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r2,      (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots1,     (long long)B * K1_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots2,     (long long)B * K2_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top1_ids,  (long long)B * ck1_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top2_ids,  (long long)B * ck2_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_sel,  (long long)B * ck3_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_cnt,  (long long)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_fine,  (long long)B * d_ * Kr_ * sizeof(float)));

    cudaFree(ws_.d_query_offsets);
    cudaFree(ws_.d_pair_leaf_a); cudaFree(ws_.d_pair_qid_a);
    cudaFree(ws_.d_pair_leaf_b); cudaFree(ws_.d_pair_qid_b);
    CUDA_CHECK(cudaMalloc(&ws_.d_query_offsets, (long long)(B + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_a,   (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_a,    (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_leaf_b,   (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_pair_qid_b,    (long long)max_pairs * sizeof(int)));

    cudaFree(ws_.d_out_dists); cudaFree(ws_.d_out_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_out_dists, (long long)max_pairs * TOP_P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_out_ids,   (long long)max_pairs * TOP_P * sizeof(int)));

    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);
    if (ws_.h_final_dists) cudaFreeHost(ws_.h_final_dists);
    if (ws_.h_final_ids)   cudaFreeHost(ws_.h_final_ids);
    CUDA_CHECK(cudaMalloc(&ws_.d_final_dists,    (long long)B * K_MAX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_ids,      (long long)B * K_MAX * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_final_dists,(long long)B * K_MAX * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_final_ids,  (long long)B * K_MAX * sizeof(int)));

    // CUB temp storage: max of scan, leaf sort (14-bit), qid sort (10-bit)
    size_t scan_bytes = 0, sort_leaf_bytes = 0, sort_qid_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
        (int*)nullptr, (int*)nullptr, B);
    cub::DeviceRadixSort::SortPairs(nullptr, sort_leaf_bytes,
        (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, max_pairs, 0, 14);
    cub::DeviceRadixSort::SortPairs(nullptr, sort_qid_bytes,
        (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, max_pairs, 0, 10);
    ws_.cub_bytes = std::max({scan_bytes, sort_leaf_bytes, sort_qid_bytes});
    cudaFree(ws_.d_cub_tmp);
    CUDA_CHECK(cudaMalloc(&ws_.d_cub_tmp, ws_.cub_bytes));

    ws_.batch_cap = B;
    ws_.max_pairs = max_pairs;
}

void HBlockIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_ids) const
{
    if (ntotal_ == 0) throw std::runtime_error("HBlock index is empty");
    if (k > K_MAX) throw std::runtime_error("k exceeds K_MAX (64)");
    using Clock = std::chrono::high_resolution_clock;

    // ── Phase 1: GPU routing ──────────────────────────────────────────────────
    auto t0 = Clock::now();
    route_queries(cublas_, d_Pi_,
                  d_route1_cents_, d_route1_norms_,
                  d_route2_cents_, d_route2_norms_,
                  d_fine_c1d_, d_pair_blk_start_, d_pair_blk_count_,
                  h_q, nq, d_, K1_, K2_, Kr_, Br_,
                  leaf_size_, ck1_, ck2_, ck3_, batch_size_, ws_);
    auto t1 = Clock::now();

    // ── Phase 2: pair count (O(nq) CPU sum) ───────────────────────────────────
    int n_pairs = 0;
    for (int qi = 0; qi < nq; qi++) n_pairs += ws_.h_leaf_cnt[qi];

    // ── Phase 3: GPU leaf sort ────────────────────────────────────────────────
    gpu_build_and_sort_pairs(nq, n_pairs, n_leaf_blocks_, ck3_, ws_);
    auto t2 = Clock::now();

    // ── Phase 4: leaf PQ kernel + GPU merge (iota/sort/topk) + D2H ───────────
    cudaStream_t s = ws_.stream;
    launch_leaf_flat(
        ws_.d_pair_leaf_b, ws_.d_pair_qid_b,
        d_leaf_codes_, d_leaf_ids_, d_leaf_sizes_, ws_.d_lut_fine,
        ws_.d_out_dists, ws_.d_out_ids,
        n_pairs, d_, Kr_, Br_, bpv_, leaf_size_, s);

    gpu_merge_topk(nq, n_pairs, k, ws_);

    CUDA_CHECK(cudaMemcpyAsync(ws_.h_final_dists, ws_.d_final_dists,
                               (long long)nq * k * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ws_.h_final_ids,   ws_.d_final_ids,
                               (long long)nq * k * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto t3 = Clock::now();

    // ── Phase 5: CPU extract (partial_sort per query) ─────────────────────────
    std::vector<int> idx_buf(k);
    for (int qi = 0; qi < nq; qi++) {
        const float* fd = ws_.h_final_dists + (long long)qi * K_MAX;
        const int*   fi = ws_.h_final_ids   + (long long)qi * K_MAX;
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
    auto t4 = Clock::now();

    printf("  [v14] Route=%.1f ms  GPUSort=%.1f ms (%d pairs)"
           "  Kernel+Merge+DMA=%.1f ms  Extract=%.1f ms  Total=%.1f ms\n",
           Ms(t1-t0).count(), Ms(t2-t1).count(), n_pairs,
           Ms(t3-t2).count(), Ms(t4-t3).count(),
           Ms(t4-t0).count());
}

} // namespace hblock_v14
