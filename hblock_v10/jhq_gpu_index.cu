#include "hblock_v10/jhq_gpu_index.cuh"
#include "hblock_v10/encode.cuh"
#include "hblock_v10/search.cuh"
#include "cpu/erfinv.h"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

using Ms = std::chrono::duration<double, std::milli>;

namespace hblock_v10 {

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
    if (ws_.h_dispatch_leaf_ids) cudaFreeHost(ws_.h_dispatch_leaf_ids);
    if (ws_.h_dispatch_offsets)  cudaFreeHost(ws_.h_dispatch_offsets);
    if (ws_.h_dispatch_qids)     cudaFreeHost(ws_.h_dispatch_qids);
    if (ws_.h_out_dists)         cudaFreeHost(ws_.h_out_dists);
    if (ws_.h_out_ids)           cudaFreeHost(ws_.h_out_ids);
    cudaFree(ws_.d_dispatch_leaf_ids);
    cudaFree(ws_.d_dispatch_offsets);
    cudaFree(ws_.d_dispatch_qids);
    cudaFree(ws_.d_out_dists);
    cudaFree(ws_.d_out_ids);
    cublasDestroy(cublas_);
    cudaFree(d_Pi_);
    cudaFree(d_route1_cents_); cudaFree(d_route1_norms_);
    cudaFree(d_route2_cents_); cudaFree(d_route2_norms_);
    cudaFree(d_fine_c1d_);
    cudaFree(d_pair_blk_start_); cudaFree(d_pair_blk_count_);
    cudaFree(d_leaf_codes_); cudaFree(d_leaf_ids_); cudaFree(d_leaf_sizes_);
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    if (ws_.h_leaf_sel) cudaFreeHost(ws_.h_leaf_sel);
    if (ws_.h_leaf_cnt) cudaFreeHost(ws_.h_leaf_cnt);
    cudaFree(ws_.d_q_batch); cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);    cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);   cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
    cudaFree(ws_.d_cub_tmp);
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

    std::vector<uint8_t> h_leaf_codes((long long)total_blocks * leaf_size_ * bpv_, 0);
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
            std::memcpy(h_leaf_codes.data() + ((long long)blk * leaf_size_ + pos) * bpv_,
                        h_fc_all.data()     + (long long)oid * bpv_, bpv_);
            h_leaf_sizes[blk] = std::max(h_leaf_sizes[blk], pos + 1);
        }
    }

    CUDA_CHECK(cudaMalloc(&d_pair_blk_start_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_blk_count_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_start_, pair_start.data(), (long long)K1_*K2_*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_count_, pair_cnt.data(),   (long long)K1_*K2_*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_leaf_codes_, (long long)total_blocks * leaf_size_ * bpv_));
    CUDA_CHECK(cudaMalloc(&d_leaf_ids_,   (long long)total_blocks * leaf_size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leaf_sizes_, (long long)total_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_leaf_codes_, h_leaf_codes.data(), (long long)total_blocks*leaf_size_*bpv_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_ids_,   h_leaf_ids.data(),   (long long)total_blocks*leaf_size_*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_sizes_, h_leaf_sizes.data(), (long long)total_blocks*sizeof(int), cudaMemcpyHostToDevice));

    ntotal_ = n;
    printf("  Built %d leaf blocks (leaf_size=%d, K1=%d, K2=%d)\n",
           total_blocks, leaf_size_, K1_, K2_);
    alloc_workspace();
}

void HBlockIndex::alloc_workspace() {
    const int B = batch_size_;
    const int max_leaves = n_leaf_blocks_;
    const int max_pairs  = B * ck3_;

    if (ws_.stream) cudaStreamDestroy(ws_.stream);
    CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));

    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    if (ws_.h_leaf_sel) cudaFreeHost(ws_.h_leaf_sel);
    if (ws_.h_leaf_cnt) cudaFreeHost(ws_.h_leaf_cnt);
    cudaFree(ws_.d_q_batch); cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);    cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);   cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);

    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned, (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_leaf_sel, (long long)B * ck3_ * sizeof(int)));
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

    // Single dispatch buffers — sized for the worst case (all leaves non-empty,
    // all queries hit all leaf slots).
    if (ws_.h_dispatch_leaf_ids) cudaFreeHost(ws_.h_dispatch_leaf_ids);
    if (ws_.h_dispatch_offsets)  cudaFreeHost(ws_.h_dispatch_offsets);
    if (ws_.h_dispatch_qids)     cudaFreeHost(ws_.h_dispatch_qids);
    if (ws_.h_out_dists)         cudaFreeHost(ws_.h_out_dists);
    if (ws_.h_out_ids)           cudaFreeHost(ws_.h_out_ids);
    cudaFree(ws_.d_dispatch_leaf_ids);
    cudaFree(ws_.d_dispatch_offsets);
    cudaFree(ws_.d_dispatch_qids);
    cudaFree(ws_.d_out_dists);
    cudaFree(ws_.d_out_ids);

    CUDA_CHECK(cudaMallocHost(&ws_.h_dispatch_leaf_ids, (long long)max_leaves * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_dispatch_offsets,  (long long)(max_leaves + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_dispatch_qids,     (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_out_dists,         (long long)max_pairs * TOP_P * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&ws_.h_out_ids,           (long long)max_pairs * TOP_P * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dispatch_leaf_ids, (long long)max_leaves * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dispatch_offsets,  (long long)(max_leaves + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dispatch_qids,     (long long)max_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_out_dists,         (long long)max_pairs * TOP_P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_out_ids,           (long long)max_pairs * TOP_P * sizeof(int)));

    ws_.batch_cap  = B;
    ws_.max_leaves = max_leaves;
    ws_.max_pairs  = max_pairs;
}

void HBlockIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_ids) const
{
    if (ntotal_ == 0) throw std::runtime_error("HBlock index is empty");
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

    // ── Phase 2: CPU fan-out ──────────────────────────────────────────────────
    std::vector<std::vector<int>> leaf_queues(n_leaf_blocks_);
    for (int qi = 0; qi < nq; qi++) {
        int cnt = ws_.h_leaf_cnt[qi];
        for (int s = 0; s < cnt; s++) {
            int lb = ws_.h_leaf_sel[qi * ck3_ + s];
            if (lb >= 0 && lb < n_leaf_blocks_)
                leaf_queues[lb].push_back(qi);
        }
    }
    auto t2 = Clock::now();

    // ── Phase 3: pack ALL non-empty leaves into ONE dispatch ──────────────────
    int n_leaves = 0, n_pairs = 0;
    for (int lb = 0; lb < n_leaf_blocks_; lb++) {
        const auto& q = leaf_queues[lb];
        if (q.empty()) continue;
        ws_.h_dispatch_leaf_ids[n_leaves] = lb;
        ws_.h_dispatch_offsets [n_leaves] = n_pairs;
        for (int qid : q)
            ws_.h_dispatch_qids[n_pairs++] = qid;
        n_leaves++;
    }
    ws_.h_dispatch_offsets[n_leaves] = n_pairs;
    auto t3 = Clock::now();

    // ── Phase 4: ONE H2D + kernel + D2H ──────────────────────────────────────
    // Kernel grid = n_leaves (~3200 blocks) — GPU sees all at once and
    // self-schedules into ~10 waves at full SM utilization.
    cudaStream_t s = ws_.stream;
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_dispatch_leaf_ids, ws_.h_dispatch_leaf_ids,
                               (long long)n_leaves * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_dispatch_offsets,  ws_.h_dispatch_offsets,
                               (long long)(n_leaves + 1) * sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(ws_.d_dispatch_qids, ws_.h_dispatch_qids,
                               (long long)n_pairs * sizeof(int), cudaMemcpyHostToDevice, s));

    launch_leaf_flink(
        ws_.d_dispatch_leaf_ids, ws_.d_dispatch_offsets, ws_.d_dispatch_qids,
        d_leaf_codes_, d_leaf_ids_, d_leaf_sizes_, ws_.d_lut_fine,
        ws_.d_out_dists, ws_.d_out_ids,
        n_leaves, d_, Kr_, Br_, bpv_, leaf_size_, s);

    CUDA_CHECK(cudaMemcpyAsync(ws_.h_out_dists, ws_.d_out_dists,
                               (long long)n_pairs * TOP_P * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(ws_.h_out_ids, ws_.d_out_ids,
                               (long long)n_pairs * TOP_P * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto t4 = Clock::now();

    // ── Phase 5: CPU merge ────────────────────────────────────────────────────
    std::vector<QueryState> qstates(nq);
    for (int pi = 0; pi < n_pairs; pi++) {
        int qid = ws_.h_dispatch_qids[pi];
        for (int r = 0; r < TOP_P; r++)
            qstates[qid].push(ws_.h_out_dists[pi * TOP_P + r],
                              ws_.h_out_ids  [pi * TOP_P + r], k);
    }
    auto t5 = Clock::now();

    // ── Phase 6: extract top-k ────────────────────────────────────────────────
    std::vector<int> idx_buf(64);
    for (int qi = 0; qi < nq; qi++) {
        QueryState& qs = qstates[qi];
        int sz   = qs.heap_size;
        int take = std::min(sz, k);
        idx_buf.resize(sz);
        std::iota(idx_buf.begin(), idx_buf.end(), 0);
        std::partial_sort(idx_buf.begin(), idx_buf.begin() + take, idx_buf.end(),
                          [&](int a, int b_){ return qs.dists[a] < qs.dists[b_]; });
        for (int r = 0; r < k; r++) {
            h_dists[qi * k + r] = (r < take) ? qs.dists[idx_buf[r]] : 1e30f;
            h_ids  [qi * k + r] = (r < take) ? qs.ids  [idx_buf[r]] : -1;
        }
    }
    auto t6 = Clock::now();

    printf("  [v10] Route=%.1f ms  FanOut=%.1f ms  Pack=%.1f ms"
           "  Kernel+DMA=%.1f ms (%d leaves, %d pairs)"
           "  Merge=%.1f ms  Extract=%.1f ms  Total=%.1f ms\n",
           Ms(t1-t0).count(), Ms(t2-t1).count(), Ms(t3-t2).count(),
           Ms(t4-t3).count(), n_leaves, n_pairs,
           Ms(t5-t4).count(), Ms(t6-t5).count(),
           Ms(t6-t0).count());
}

} // namespace hblock_v10
