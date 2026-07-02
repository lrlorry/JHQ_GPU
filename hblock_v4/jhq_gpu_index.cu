#include "hblock_v4/jhq_gpu_index.cuh"
#include "hblock_v4/encode.cuh"
#include "hblock_v4/search.cuh"
#include "cpu/erfinv.h"
#include "common/cuda_utils.cuh"

#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
  typedef __CLPK_integer lapack_int_t;
#else
  #include <cblas.h>
  typedef int lapack_int_t;
  extern "C" {
    void ssyev_(char* jobz, char* uplo, lapack_int_t* n, float* a, lapack_int_t* lda,
                float* w, float* work, lapack_int_t* lwork, lapack_int_t* info);
  }
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace hblock_v4 {

// ── Shared helpers (same as v3) ────────────────────────────────────────────────

static std::vector<float>
pca_top_k(const float* X, int n, int d, int k)
{
    std::vector<float> XtX((long long)d * d, 0.f);
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans,
                d, n, 1.0f / n, X, d, 0.0f, XtX.data(), d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < i; ++j)
            XtX[(long long)i * d + j] = XtX[(long long)j * d + i];

    std::vector<float> W(d);
    lapack_int_t n_ = d, lda_ = d, lwork_ = -1, info_ = 0;
    float work_query = 0.f;
    ssyev_((char*)"V", (char*)"U", &n_, XtX.data(), &lda_,
           W.data(), &work_query, &lwork_, &info_);
    lwork_ = (lapack_int_t)work_query;
    std::vector<float> work_buf(lwork_);
    ssyev_((char*)"V", (char*)"U", &n_, XtX.data(), &lda_,
           W.data(), work_buf.data(), &lwork_, &info_);
    if (info_ != 0)
        throw std::runtime_error("ssyev_ failed: info=" + std::to_string(info_));

    std::vector<float> P((long long)k * d);
    std::memcpy(P.data(), XtX.data() + (long long)(d - k) * d,
                (long long)k * d * sizeof(float));
    return P;
}

static std::vector<float>
project(const float* X, int n, int d, const float* P, int k)
{
    std::vector<float> Z((long long)n * k);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, k, d, 1.f, X, d, P, d, 0.f, Z.data(), k);
    return Z;
}

static void
kmeans_cpu(const float* Z, int n, int k_dim, int K, int n_iters, int seed,
           std::vector<float>& mu, std::vector<int>& assigns)
{
    mu.assign((long long)K * k_dim, 0.f);
    assigns.assign(n, 0);

    std::mt19937 rng(seed);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int c = 0; c < K; ++c)
        std::memcpy(mu.data() + (long long)c * k_dim,
                    Z + (long long)perm[c % n] * k_dim, k_dim * sizeof(float));

    std::vector<float> dots((long long)n * K), norms(K), cnt(K);

    for (int iter = 0; iter < n_iters; ++iter) {
        for (int c = 0; c < K; ++c) {
            double s = 0.0;
            const float* mc = mu.data() + (long long)c * k_dim;
            for (int j = 0; j < k_dim; ++j) s += (double)mc[j] * mc[j];
            norms[c] = (float)s;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    n, K, k_dim, 1.f, Z, k_dim, mu.data(), k_dim, 0.f, dots.data(), K);
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            const float* row = dots.data() + (long long)i * K;
            float best = norms[0] - 2.f * row[0]; int bc = 0;
            for (int c = 1; c < K; ++c) {
                float dd = norms[c] - 2.f * row[c];
                if (dd < best) { best = dd; bc = c; }
            }
            if (assigns[i] != bc) { assigns[i] = bc; changed = true; }
        }
        if (!changed && iter > 0) { printf("  K-means converged at iter %d\n", iter); break; }
        std::fill(mu.begin(), mu.end(), 0.f);
        std::fill(cnt.begin(), cnt.end(), 0.f);
        for (int i = 0; i < n; ++i) {
            int c = assigns[i]; cnt[c] += 1.f;
            float* mc = mu.data() + (long long)c * k_dim;
            const float* zi = Z + (long long)i * k_dim;
            for (int j = 0; j < k_dim; ++j) mc[j] += zi[j];
        }
        for (int c = 0; c < K; ++c) {
            if (cnt[c] == 0.f) {
                int ri = (int)(rng() % n);
                std::memcpy(mu.data() + (long long)c * k_dim,
                            Z + (long long)ri * k_dim, k_dim * sizeof(float));
            } else {
                float inv = 1.f / cnt[c];
                float* mc = mu.data() + (long long)c * k_dim;
                for (int j = 0; j < k_dim; ++j) mc[j] *= inv;
            }
        }
    }
}

static std::vector<float>
full_centroids(const float* X, const std::vector<int>& assigns, int n, int d, int K)
{
    std::vector<float> C((long long)K * d, 0.f);
    std::vector<float> cnt(K, 0.f);
    for (int i = 0; i < n; ++i) {
        int c = assigns[i]; cnt[c] += 1.f;
        float*       cc = C.data() + (long long)c * d;
        const float* xi = X + (long long)i * d;
        for (int j = 0; j < d; ++j) cc[j] += xi[j];
    }
    for (int c = 0; c < K; ++c) {
        if (cnt[c] > 0.f) {
            float inv = 1.f / cnt[c]; float* cc = C.data() + (long long)c * d;
            for (int j = 0; j < d; ++j) cc[j] *= inv;
        }
    }
    return C;
}

static std::vector<float>
proj_centroids(const std::vector<float>& C_full, const float* P, int K, int d, int k)
{
    std::vector<float> C_proj((long long)K * k);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                K, k, d, 1.f, C_full.data(), d, P, d, 0.f, C_proj.data(), k);
    return C_proj;
}

static float sigma_from_residuals(const float* R, long long total)
{
    double s = 0.0;
    for (long long i = 0; i < total; ++i) s += (double)R[i] * R[i];
    return (float)std::sqrt(s / (double)total);
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

// ── S_B (between-class scatter) top-k eigenvectors ───────────────────────────
// Given: C_full [K×d], cluster counts [K], global mean μ [d]
// S_B = Σ_c n_c (μ_c - μ)(μ_c - μ)^T  =  M^T M
//   where M[c] = sqrt(n_c) * (μ_c - μ)
// rank(S_B) ≤ K-1, so k ≤ K-1 required for non-trivial eigenvectors.
static std::vector<float>
sb_top_k(const std::vector<float>& C_full, const std::vector<int>& assigns,
         int n, int d, int K, int k)
{
    // Count vectors per cluster
    std::vector<float> cnt(K, 0.f);
    for (int i = 0; i < n; ++i) cnt[assigns[i]] += 1.f;

    // Global mean
    std::vector<float> mu(d, 0.f);
    for (int c = 0; c < K; ++c) {
        const float* cc = C_full.data() + (long long)c * d;
        for (int j = 0; j < d; ++j) mu[j] += cnt[c] * cc[j];
    }
    for (int j = 0; j < d; ++j) mu[j] /= n;

    // Build M [K×d]: M[c] = sqrt(n_c) * (μ_c - μ)
    std::vector<float> M((long long)K * d);
    for (int c = 0; c < K; ++c) {
        float scale = std::sqrt(cnt[c] / n);
        const float* cc = C_full.data() + (long long)c * d;
        float*       mc = M.data() + (long long)c * d;
        for (int j = 0; j < d; ++j) mc[j] = scale * (cc[j] - mu[j]);
    }

    // S_B = M^T M (d×d), computed via ssyrk
    std::vector<float> SB((long long)d * d, 0.f);
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans,
                d, K, 1.0f, M.data(), d, 0.0f, SB.data(), d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < i; ++j)
            SB[(long long)i * d + j] = SB[(long long)j * d + i];

    // Eigendecompose S_B
    std::vector<float> W(d);
    lapack_int_t n_ = d, lda_ = d, lwork_ = -1, info_ = 0;
    float work_query = 0.f;
    ssyev_((char*)"V", (char*)"U", &n_, SB.data(), &lda_,
           W.data(), &work_query, &lwork_, &info_);
    lwork_ = (lapack_int_t)work_query;
    std::vector<float> work_buf(lwork_);
    ssyev_((char*)"V", (char*)"U", &n_, SB.data(), &lda_,
           W.data(), work_buf.data(), &lwork_, &info_);
    if (info_ != 0)
        throw std::runtime_error("ssyev_ (S_B) failed: info=" + std::to_string(info_));

    printf("  S_B top eigenvalue: %.4f (rank≤%d)\n", W[d - 1], K - 1);

    // Top-k eigenvectors = last k rows in row-major view
    std::vector<float> P((long long)k * d);
    std::memcpy(P.data(), SB.data() + (long long)(d - k) * d,
                (long long)k * d * sizeof(float));
    return P;
}

// ── Discriminative projections: PCA init → S_B refinement ────────────────────
static void compute_projections_lda(
    const float* h_x, int n_train, int d, int K1, int k1, int K2, int k2,
    int kmeans_iters, int seed,
    std::vector<float>& P1_out,
    std::vector<float>& C1_full_out,
    std::vector<int>&   assigns1_out,
    std::vector<float>& P2_out,
    std::vector<float>& C2_full_out,
    std::vector<int>&   assigns2_out)
{
    // Pass 1: PCA init for L1
    printf("  [L1] PCA init top-%d of %d dims...\n", k1, d);
    std::vector<float> P1_pca = pca_top_k(h_x, n_train, d, k1);
    std::vector<float> Z1_pca = project(h_x, n_train, d, P1_pca.data(), k1);

    printf("  [L1] K-means K=%d (init for S_B)...\n", K1);
    std::vector<float> mu1_pca;
    kmeans_cpu(Z1_pca.data(), n_train, k1, K1, kmeans_iters, seed,
               mu1_pca, assigns1_out);
    (void)mu1_pca;
    C1_full_out = full_centroids(h_x, assigns1_out, n_train, d, K1);

    // Pass 2: S_B projection for L1
    printf("  [L1] S_B top-%d eigenvectors...\n", k1);
    P1_out = sb_top_k(C1_full_out, assigns1_out, n_train, d, K1, k1);

    // Re-run k-means with S_B projection
    printf("  [L1] K-means K=%d (with S_B projection)...\n", K1);
    std::vector<float> Z1_lda = project(h_x, n_train, d, P1_out.data(), k1);
    std::vector<float> mu1_lda;
    kmeans_cpu(Z1_lda.data(), n_train, k1, K1, kmeans_iters, seed,
               mu1_lda, assigns1_out);
    (void)mu1_lda;
    C1_full_out = full_centroids(h_x, assigns1_out, n_train, d, K1);

    // L1 residuals for L2 stage
    printf("  [L2] Computing L1 residuals...\n");
    std::vector<float> R1((long long)n_train * d);
    for (int i = 0; i < n_train; ++i) {
        int          c1  = assigns1_out[i];
        const float* xi  = h_x + (long long)i * d;
        const float* c1p = C1_full_out.data() + (long long)c1 * d;
        float*        ri = R1.data() + (long long)i * d;
        for (int j = 0; j < d; ++j) ri[j] = xi[j] - c1p[j];
    }

    // Pass 1: PCA init for L2
    printf("  [L2] PCA init top-%d of %d dims...\n", k2, d);
    std::vector<float> P2_pca = pca_top_k(R1.data(), n_train, d, k2);
    std::vector<float> Z2_pca = project(R1.data(), n_train, d, P2_pca.data(), k2);

    printf("  [L2] K-means K=%d (init for S_B)...\n", K2);
    std::vector<float> mu2_pca;
    kmeans_cpu(Z2_pca.data(), n_train, k2, K2, kmeans_iters, seed + 1,
               mu2_pca, assigns2_out);
    (void)mu2_pca;
    C2_full_out = full_centroids(R1.data(), assigns2_out, n_train, d, K2);

    // Pass 2: S_B for L2
    printf("  [L2] S_B top-%d eigenvectors...\n", k2);
    P2_out = sb_top_k(C2_full_out, assigns2_out, n_train, d, K2, k2);

    // Re-run k-means with S_B projection
    printf("  [L2] K-means K=%d (with S_B projection)...\n", K2);
    std::vector<float> Z2_lda = project(R1.data(), n_train, d, P2_out.data(), k2);
    std::vector<float> mu2_lda;
    kmeans_cpu(Z2_lda.data(), n_train, k2, K2, kmeans_iters, seed + 1,
               mu2_lda, assigns2_out);
    (void)mu2_lda;
    C2_full_out = full_centroids(R1.data(), assigns2_out, n_train, d, K2);
}

// ── Constructor / Destructor ──────────────────────────────────────────────────
HBlockIndex::HBlockIndex(int d, Params p)
    : d_(d), k1_(p.k1), k2_(p.k2),
      K1_(p.K1), K2_(p.K2),
      Kr_(p.Kr), Br_(p.Br),
      bpv_((d * p.Br + 7) / 8),
      leaf_size_(p.leaf_size),
      ck1_(p.ck1), ck2_(p.ck2), ck3_(p.ck3),
      batch_size_(p.batch_size),
      kmeans_iters_(p.kmeans_iters),
      seed_(p.seed)
{
    if (d <= 0)  throw std::invalid_argument("d must be positive");
    if (p.Br != 4 && p.Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    if (p.k1 <= 0 || p.k1 > d) throw std::invalid_argument("k1 out of range");
    if (p.k2 <= 0 || p.k2 > d) throw std::invalid_argument("k2 out of range");
    CUBLAS_CHECK(cublasCreate(&cublas_));
}

HBlockIndex::~HBlockIndex() {
    if (ws_.stream) cudaStreamDestroy(ws_.stream);
    cublasDestroy(cublas_);
    cudaFree(d_P1_);            cudaFree(d_P2_);
    cudaFree(d_C1_proj_);       cudaFree(d_C1_proj_norms_);  cudaFree(d_C1_full_);
    cudaFree(d_C2_proj_);       cudaFree(d_C2_proj_norms_);  cudaFree(d_C2_full_);
    cudaFree(d_fine_c1d_);
    cudaFree(d_pair_blk_start_); cudaFree(d_pair_blk_count_);
    cudaFree(d_leaf_codes_);     cudaFree(d_leaf_ids_);       cudaFree(d_leaf_sizes_);
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch); cudaFree(ws_.d_z1);  cudaFree(ws_.d_z2);
    cudaFree(ws_.d_q_r1);   cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);  cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
    cudaFree(ws_.d_fine_dists); cudaFree(ws_.d_fine_ids);
    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);
}

void HBlockIndex::upload_proj_centroids(
    const std::vector<float>& C_proj, const std::vector<float>& C_full,
    int K, int k, float*& d_Cp, float*& d_Cn, float*& d_Cf)
{
    std::vector<float> norms(K);
    for (int c = 0; c < K; ++c) {
        double s = 0.0;
        const float* cp = C_proj.data() + (long long)c * k;
        for (int j = 0; j < k; ++j) s += (double)cp[j] * cp[j];
        norms[c] = (float)s;
    }
    cudaFree(d_Cp); cudaFree(d_Cn); cudaFree(d_Cf);
    CUDA_CHECK(cudaMalloc(&d_Cp, (long long)K * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cn, (long long)K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Cf, (long long)K * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Cp, C_proj.data(),
                          (long long)K * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Cn, norms.data(),
                          (long long)K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Cf, C_full.data(),
                          (long long)K * d_ * sizeof(float), cudaMemcpyHostToDevice));
}

void HBlockIndex::train(const float* h_x, int n_train) {
    std::vector<float> P1, C1_full, P2, C2_full;
    std::vector<int>   assigns1, assigns2;
    compute_projections_lda(h_x, n_train, d_, K1_, k1_, K2_, k2_,
                             kmeans_iters_, seed_,
                             P1, C1_full, assigns1, P2, C2_full, assigns2);

    std::vector<float> C1_proj = proj_centroids(C1_full, P1.data(), K1_, d_, k1_);
    std::vector<float> C2_proj = proj_centroids(C2_full, P2.data(), K2_, d_, k2_);

    // Sigma from L2 residuals
    printf("  Computing L2 residuals for sigma estimation...\n");
    std::vector<float> R2((long long)n_train * d_);
    {
        std::vector<float> R1((long long)n_train * d_);
        for (int i = 0; i < n_train; ++i) {
            int c1 = assigns1[i];
            const float* xi  = h_x + (long long)i * d_;
            const float* c1p = C1_full.data() + (long long)c1 * d_;
            float*        r1 = R1.data() + (long long)i * d_;
            for (int j = 0; j < d_; ++j) r1[j] = xi[j] - c1p[j];
        }
        for (int i = 0; i < n_train; ++i) {
            int c2 = assigns2[i];
            const float* r1  = R1.data() + (long long)i * d_;
            const float* c2p = C2_full.data() + (long long)c2 * d_;
            float*        r2 = R2.data() + (long long)i * d_;
            for (int j = 0; j < d_; ++j) r2[j] = r1[j] - c2p[j];
        }
    }
    float sigma = sigma_from_residuals(R2.data(), (long long)n_train * d_);
    printf("  Fine codebook: sigma=%.4f\n", sigma);
    fine_c1d_ = analytical_fine_c1d(Kr_, sigma);

    CUDA_CHECK(cudaMalloc(&d_P1_, (long long)k1_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P2_, (long long)k2_ * d_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_P1_, P1.data(), (long long)k1_ * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P2_, P2.data(), (long long)k2_ * d_ * sizeof(float),
                          cudaMemcpyHostToDevice));

    upload_proj_centroids(C1_proj, C1_full, K1_, k1_,
                          d_C1_proj_, d_C1_proj_norms_, d_C1_full_);
    upload_proj_centroids(C2_proj, C2_full, K2_, k2_,
                          d_C2_proj_, d_C2_proj_norms_, d_C2_full_);

    CUDA_CHECK(cudaMalloc(&d_fine_c1d_, Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fine_c1d_, fine_c1d_.data(),
                          Kr_ * sizeof(float), cudaMemcpyHostToDevice));
}

void HBlockIndex::add(const float* h_x, int n) {
    if (!d_P1_) throw std::runtime_error("call train() before add()");
    if (ntotal_ != 0) throw std::runtime_error("HBlock v4 currently supports one add() call");

    const int BATCH = 8192;
    const float one = 1.f, zero = 0.f;

    float   *d_x, *d_r1, *d_r2, *d_z1, *d_z2, *d_dots;
    int     *d_c1, *d_c2;
    uint8_t *d_fc;
    CUDA_CHECK(cudaMalloc(&d_x,    (long long)BATCH * d_  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r1,   (long long)BATCH * d_  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r2,   (long long)BATCH * d_  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z1,   (long long)BATCH * k1_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z2,   (long long)BATCH * k2_ * sizeof(float)));
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

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 k1_, nb, d_, &one, d_P1_, d_, d_x, d_, &zero, d_z1, k1_));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K1_, nb, k1_, &one, d_C1_proj_, k1_, d_z1, k1_,
                                 &zero, d_dots, K1_));
        launch_assign_from_dots(d_dots, d_C1_proj_norms_, d_c1, K1_, nb, nullptr);

        launch_subtract_centroid(d_x, d_c1, d_C1_full_, d_r1, nb, d_, nullptr);

        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 k2_, nb, d_, &one, d_P2_, d_, d_r1, d_, &zero, d_z2, k2_));
        CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                                 K2_, nb, k2_, &one, d_C2_proj_, k2_, d_z2, k2_,
                                 &zero, d_dots, K2_));
        launch_assign_from_dots(d_dots, d_C2_proj_norms_, d_c2, K2_, nb, nullptr);

        launch_subtract_centroid(d_r1, d_c2, d_C2_full_, d_r2, nb, d_, nullptr);
        launch_fine_encode(d_r2, d_fine_c1d_, d_fc, nb, d_, Kr_, Br_, bpv_, nullptr);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_code1.data() + s, d_c1, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_code2.data() + s, d_c2, nb * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fc_all.data() + (long long)s * bpv_,
                              d_fc, (long long)nb * bpv_, cudaMemcpyDeviceToHost));
    }
    cudaFree(d_x); cudaFree(d_r1); cudaFree(d_r2); cudaFree(d_z1); cudaFree(d_z2);
    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_fc); cudaFree(d_dots);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return (long long)h_code1[a] * K2_ + h_code2[a] <
               (long long)h_code1[b] * K2_ + h_code2[b];
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
    for (int p = 0; p < K1_ * K2_; ++p) { pair_start[p] = total_blocks; total_blocks += pair_cnt[p]; }
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
                            h_fc_all.data() + (long long)orig_id * bpv_, bpv_);
                h_leaf_sizes[blk_idx] = std::max(h_leaf_sizes[blk_idx], pos + 1);
            }
            i = j;
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
    const int B = batch_size_, nc = ck3_ * leaf_size_;
    if (ws_.h_q_pinned) cudaFreeHost(ws_.h_q_pinned);
    cudaFree(ws_.d_q_batch); cudaFree(ws_.d_z1);  cudaFree(ws_.d_z2);
    cudaFree(ws_.d_q_r1);   cudaFree(ws_.d_q_r2);
    cudaFree(ws_.d_dots1);  cudaFree(ws_.d_dots2);
    cudaFree(ws_.d_top1_ids); cudaFree(ws_.d_top2_ids);
    cudaFree(ws_.d_leaf_sel); cudaFree(ws_.d_leaf_cnt);
    cudaFree(ws_.d_lut_fine);
    cudaFree(ws_.d_fine_dists); cudaFree(ws_.d_fine_ids);
    cudaFree(ws_.d_final_dists); cudaFree(ws_.d_final_ids);

    CUDA_CHECK(cudaMallocHost(&ws_.h_q_pinned,  (long long)B * d_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_batch,        (long long)B * d_  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_z1,             (long long)B * k1_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_z2,             (long long)B * k2_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r1,           (long long)B * d_  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_q_r2,           (long long)B * d_  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots1,          (long long)B * K1_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_dots2,          (long long)B * K2_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top1_ids,       (long long)B * ck1_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_top2_ids,       (long long)B * ck2_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_sel,       (long long)B * ck3_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_leaf_cnt,       (long long)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_lut_fine,       (long long)B * d_ * Kr_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_fine_dists,     (long long)B * nc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_fine_ids,       (long long)B * nc * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_dists,    (long long)B * 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws_.d_final_ids,      (long long)B * 1024 * sizeof(int)));

    ws_.batch_cap = B;
    if (!ws_.stream) CUDA_CHECK(cudaStreamCreate(&ws_.stream));
    CUBLAS_CHECK(cublasSetStream(cublas_, ws_.stream));
}

void HBlockIndex::search(const float* h_q, int nq, int k,
                          float* h_dists, int* h_ids) const
{
    if (ntotal_ == 0) throw std::runtime_error("HBlock v4 index is empty");

    search_hblock(cublas_,
                  d_P1_, d_P2_,
                  d_C1_proj_, d_C1_proj_norms_, d_C1_full_,
                  d_C2_proj_, d_C2_proj_norms_, d_C2_full_,
                  d_fine_c1d_,
                  d_pair_blk_start_, d_pair_blk_count_,
                  d_leaf_codes_, d_leaf_ids_, d_leaf_sizes_,
                  h_q,
                  nq, d_, K1_, K2_, k1_, k2_,
                  Kr_, Br_, bpv_,
                  leaf_size_, ck1_, ck2_, ck3_, k,
                  batch_size_,
                  ws_,
                  h_dists, h_ids);
}

} // namespace hblock_v4
