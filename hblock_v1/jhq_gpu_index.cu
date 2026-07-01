#include "hblock_v1/jhq_gpu_index.cuh"
#include "hblock_v1/encode.cuh"
#include "hblock_v1/search.cuh"
#include "cpu/erfinv.h"
#include "common/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace hblock {

// ── Analytical codebook builders (JL → i.i.d. N(0,σ²) per dim) ──────────────

// Product-code routing centroids: K = 2^M_r, routes on dims [dim_offset, dim_offset+M_r).
// Centroid c, dim j: ±σ·√(2/π) if j is a routing dim, else 0.
// Optimal 2-centroid 1D Lloyd-Max centroids for N(0,σ²) are exactly ±σ√(2/π).
static std::vector<float>
analytical_route_cents(int K, int d, int dim_offset, float sigma)
{
    int M_r = 0;
    while ((1 << M_r) < K) ++M_r;
    const float cv = sigma * std::sqrt(2.0f / float(M_PI));
    std::vector<float> cents((long long)K * d, 0.f);
    for (int c = 0; c < K; ++c) {
        float* row = cents.data() + (long long)c * d;
        for (int m = 0; m < M_r; ++m)
            row[dim_offset + m] = ((c >> m) & 1) ? +cv : -cv;
    }
    return cents;
}

// Analytical Lloyd-Max scalar centroids for N(0,σ²) with Kr quantisation levels.
// c_i = σ·√2·erfinv((2i+1)/Kr − 1), i=0,…,Kr-1  (sorted ascending).
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

// ── Constructor / Destructor ──────────────────────────────────────────────────
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
    cudaFree(d_leaf_codes_);
    cudaFree(d_leaf_ids_);
    cudaFree(d_leaf_sizes_);
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch);  cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);     cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);    cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
    cudaFree(ws_.d_fine_dists); cudaFree(ws_.d_fine_ids);
    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);
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

// ── Train ─────────────────────────────────────────────────────────────────────
// After JL rotation every dimension is i.i.d. N(0,σ²), so all codebooks have
// analytical closed-form solutions — no k-means needed.
//
// Routing: product code on M_r = log2(K) routing dimensions.
//   Optimal 2-centroid 1D Lloyd-Max on N(0,σ²): ±σ√(2/π).
//   L1 routes on dims [0, M_r1), L2 routes on dims [M_r1, M_r1+M_r2).
//
// Fine code: scalar 1D Lloyd-Max on N(0,σ²) with Kr levels.
//   c_i = σ·√2·erfinv((2i+1)/Kr − 1).
void HBlockIndex::train(const float* h_x, int n_train) {
    int M_r1 = 0; while ((1 << M_r1) < K1_) ++M_r1;  // log2(K1)
    int M_r2 = 0; while ((1 << M_r2) < K2_) ++M_r2;  // log2(K2)
    if (M_r1 + M_r2 >= d_)
        throw std::invalid_argument("log2(K1)+log2(K2) must be < d");

    // σ² = E[||x||²]/d  — no rotation needed (Lemma 2 in JL paper)
    jl_.estimate_sigma(h_x, n_train);
    float sigma = jl_.sigma();
    printf("  sigma=%.4f  L1: K1=%d → M_r1=%d dims  L2: K2=%d → M_r2=%d dims\n",
           sigma, K1_, M_r1, K2_, M_r2);

    // Upload JL rotation matrix
    CUDA_CHECK(cudaMalloc(&d_Pi_, (long long)d_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Pi_, jl_.pi_data(),
                          (long long)d_ * d_ * sizeof(float), cudaMemcpyHostToDevice));

    // Analytical routing centroids — no k-means, no training data on GPU
    h_route1_ = analytical_route_cents(K1_, d_, 0,    sigma);
    h_route2_ = analytical_route_cents(K2_, d_, M_r1, sigma);
    upload_centroids(h_route1_, d_route1_cents_, d_route1_norms_, K1_);
    upload_centroids(h_route2_, d_route2_cents_, d_route2_norms_, K2_);

    // Analytical fine codebook — Lloyd-Max on N(0,σ²)
    fine_c1d_ = analytical_fine_c1d(Kr_, sigma);
    CUDA_CHECK(cudaMalloc(&d_fine_c1d_, Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fine_c1d_, fine_c1d_.data(),
                          Kr_ * sizeof(float), cudaMemcpyHostToDevice));
}

// ── Add ───────────────────────────────────────────────────────────────────────
// Batched pipeline: each batch of ADD_BATCH vectors goes through
//   H2D → rotate → L1 assign → L1 residual → L2 assign → L2 residual → fine encode → D2H
// Peak GPU temp memory: 4 × ADD_BATCH × d × 4 bytes ≈ 96 MB for ADD_BATCH=8192, d=768.
void HBlockIndex::add(const float* h_x, int n) {
    if (!d_Pi_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0) throw std::runtime_error("HBlock currently supports one add() call");

    const int BATCH = 8192;
    const float one = 1.f, zero = 0.f;

    // Temp GPU buffers for one batch
    float   *d_x, *d_y, *d_r1, *d_r2;
    int     *d_c1, *d_c2;
    uint8_t *d_fc;
    float   *d_dots;
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

        // H2D
        CUDA_CHECK(cudaMemcpy(d_x, h_x + (long long)s * d_,
                              (long long)nb * d_ * sizeof(float), cudaMemcpyHostToDevice));
        // rotate: d_y = Pi @ d_x
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 d_, nb, d_, &one, d_Pi_, d_, d_x, d_, &zero, d_y, d_));
        // L1 assign
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K1_, nb, d_, &one,
                                 d_route1_cents_, d_, d_y, d_, &zero, d_dots, K1_));
        launch_assign_from_dots(d_dots, d_route1_norms_, d_c1, K1_, nb, nullptr);
        // L1 residual
        launch_subtract_centroid(d_y, d_c1, d_route1_cents_, d_r1, nb, d_, nullptr);
        // L2 assign
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K2_, nb, d_, &one,
                                 d_route2_cents_, d_, d_r1, d_, &zero, d_dots, K2_));
        launch_assign_from_dots(d_dots, d_route2_norms_, d_c2, K2_, nb, nullptr);
        // L2 residual
        launch_subtract_centroid(d_r1, d_c2, d_route2_cents_, d_r2, nb, d_, nullptr);
        // fine encode
        launch_fine_encode(d_r2, d_fine_c1d_, d_fc, nb, d_, Kr_, Br_, bpv_, nullptr);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_code1.data() + s, d_c1, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_code2.data() + s, d_c2, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fc_all.data() + (long long)s * bpv_,
                              d_fc, (long long)nb * bpv_, cudaMemcpyDeviceToHost));
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_r1); cudaFree(d_r2);
    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_fc); cudaFree(d_dots);

    std::vector<uint8_t>& h_fc = h_fc_all;  // alias for the leaf-building code below

    // 7. Sort vectors by (K1_code, K2_code), then group into leaf blocks
    // Sort key: code1 * K2 + code2
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        long long ka = (long long)h_code1[a] * K2_ + h_code2[a];
        long long kb = (long long)h_code1[b] * K2_ + h_code2[b];
        return ka < kb;
    });

    // 8. Build leaf blocks and routing table
    // Vectors are sorted by (code1, code2). Group consecutive vectors into leaf_size blocks.
    // Each block belongs to one (code1, code2) pair.

    // First pass: count blocks per pair and compute pair offsets
    std::vector<int> pair_cnt(K1_ * K2_, 0);
    {
        int i = 0;
        while (i < n) {
            int c1 = h_code1[order[i]];
            int c2 = h_code2[order[i]];
            int j = i;
            while (j < n && h_code1[order[j]] == c1 && h_code2[order[j]] == c2) ++j;
            int pair_vecs = j - i;
            pair_cnt[c1 * K2_ + c2] = (pair_vecs + leaf_size_ - 1) / leaf_size_;
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

    // Second pass: fill leaf block arrays
    std::vector<uint8_t> h_leaf_codes((long long)total_blocks * leaf_size_ * bpv_, 0);
    std::vector<int>     h_leaf_ids  ((long long)total_blocks * leaf_size_, -1);
    std::vector<int>     h_leaf_sizes(total_blocks, 0);

    {
        int i = 0;
        while (i < n) {
            int c1 = h_code1[order[i]];
            int c2 = h_code2[order[i]];
            int j = i;
            while (j < n && h_code1[order[j]] == c1 && h_code2[order[j]] == c2) ++j;

            int base_blk = pair_start[c1 * K2_ + c2];
            for (int vi = i; vi < j; ++vi) {
                int in_blk     = vi - i;
                int blk_idx    = base_blk + in_blk / leaf_size_;
                int pos_in_blk = in_blk % leaf_size_;
                int orig_id    = order[vi];

                h_leaf_ids  [(long long)blk_idx * leaf_size_ + pos_in_blk] = orig_id;
                {
                    // Transposed layout [n_leaf_blocks, bpv, leaf_size]: byte b of vector v
                    // at dst_base[b * leaf_size + v] → warp reads rc_base[b*leaf_size+v]
                    // with stride 1 (coalesced) instead of stride bpv (uncoalesced).
                    const uint8_t* src = h_fc.data() + (long long)orig_id * bpv_;
                    uint8_t* dst_base  = h_leaf_codes.data() + (long long)blk_idx * bpv_ * leaf_size_;
                    for (int b = 0; b < bpv_; ++b)
                        dst_base[(long long)b * leaf_size_ + pos_in_blk] = src[b];
                }
                h_leaf_sizes[blk_idx] = std::max(h_leaf_sizes[blk_idx], pos_in_blk + 1);
            }
            i = j;
        }
    }

    // 9. Upload to GPU
    CUDA_CHECK(cudaMalloc(&d_pair_blk_start_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_blk_count_, (long long)K1_ * K2_ * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_start_, pair_start.data(),
                          (long long)K1_ * K2_ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pair_blk_count_, pair_cnt.data(),
                          (long long)K1_ * K2_ * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_leaf_codes_,
                          (long long)total_blocks * leaf_size_ * bpv_));
    CUDA_CHECK(cudaMalloc(&d_leaf_ids_,
                          (long long)total_blocks * leaf_size_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_leaf_sizes_, (long long)total_blocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_leaf_codes_, h_leaf_codes.data(),
                          (long long)total_blocks * leaf_size_ * bpv_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_ids_, h_leaf_ids.data(),
                          (long long)total_blocks * leaf_size_ * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leaf_sizes_, h_leaf_sizes.data(),
                          (long long)total_blocks * sizeof(int),
                          cudaMemcpyHostToDevice));

    ntotal_ = n;
    printf("  Built %d leaf blocks (leaf_size=%d, K1=%d, K2=%d)\n",
           total_blocks, leaf_size_, K1_, K2_);

    alloc_workspace();
}

// ── Workspace allocation ──────────────────────────────────────────────────────
void HBlockIndex::alloc_workspace() {
    const int B  = batch_size_;
    const int nc = ck3_ * leaf_size_;

    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch);  cudaFree(ws_.d_q_rot);
    cudaFree(ws_.d_q_r1);     cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);    cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
    cudaFree(ws_.d_fine_dists); cudaFree(ws_.d_fine_ids);
    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);

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

    ws_.batch_cap = B;
    if (!ws_.stream) {
        CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    }
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));
}

// ── Search ────────────────────────────────────────────────────────────────────
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
                  batch_size_,
                  ws_,
                  h_dists, h_ids);
}

} // namespace hblock
