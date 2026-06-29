#include "cpu/jl_transform.h"
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
typedef __CLPK_integer lapack_int_t;
#else
#include <cblas.h>
typedef int lapack_int_t;
extern "C" {
void sgeqrf_(lapack_int_t* m, lapack_int_t* n, float* A, lapack_int_t* lda,
             float* tau, float* work, lapack_int_t* lwork, lapack_int_t* info);
void sorgqr_(lapack_int_t* m, lapack_int_t* n, lapack_int_t* k, float* A,
             lapack_int_t* lda, float* tau, float* work, lapack_int_t* lwork,
             lapack_int_t* info);
}
#endif
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

JLTransform::JLTransform(int d, int seed) : d_(d), Pi_((size_t)d * d) {
    build_rotation(seed);
}

void JLTransform::build_rotation(int seed) {
    // Step 1: fill Pi_ with i.i.d. N(0,1) entries → random Gaussian matrix G
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    for (float& v : Pi_) v = norm(rng);

    // Step 2: QR decomposition G = Q·R via LAPACK sgeqrf + sorgqr.
    //   Pi_ is (d×d) in column-major order (LAPACK default).
    //   After sorgqr, Pi_ holds Q — the random orthogonal matrix Π.
    lapack_int_t m = d_, n = d_, lda = d_, info;
    std::vector<float> tau(d_);

    // Query optimal workspace size
    lapack_int_t lwork = -1;
    float work_sz;
    sgeqrf_(&m, &n, Pi_.data(), &lda, tau.data(), &work_sz, &lwork, &info);
    lwork = (lapack_int_t)work_sz;
    std::vector<float> work(lwork);

    // Factorize
    sgeqrf_(&m, &n, Pi_.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) throw std::runtime_error("sgeqrf_ failed");

    // Extract explicit Q
    lwork = -1;
    sorgqr_(&m, &n, &n, Pi_.data(), &lda, tau.data(), &work_sz, &lwork, &info);
    lwork = (lapack_int_t)work_sz;
    work.assign(lwork, 0.0f);
    sorgqr_(&m, &n, &n, Pi_.data(), &lda, tau.data(), work.data(), &lwork, &info);
    if (info != 0) throw std::runtime_error("sorgqr_ failed");
    // Pi_ now holds Π in column-major order.
}

void JLTransform::apply(const float* x, float* y, int n) const {
    // We want: y_i = Π · x_i  for each row-vector x_i.
    // Stacked: Y(n×d) = X(n×d) · Πᵀ
    //
    // Pi_ stores Π column-major, which is identical to Πᵀ row-major.
    // So CBLAS SGEMM with NoTrans on both operands gives Y = X · (Πᵀ in row-major)
    // = X · Πᵀ as desired.
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                n, d_, d_,
                1.0f, x,       d_,
                      Pi_.data(), d_,
                0.0f, y,       d_);
}

void JLTransform::estimate_sigma(const float* x, int n_samples) {
    // Lemma 2: after rotation, Var[y_i] = ||x||²/d.
    // So σ² = E[||x||²]/d — computed directly from raw vectors in O(nd),
    // avoiding the full O(nd²) rotation that the naive approach requires.
    double sum2 = 0.0;
    for (int i = 0; i < n_samples; i++) {
        const float* xi = x + (size_t)i * d_;
        for (int j = 0; j < d_; j++) sum2 += (double)xi[j] * xi[j];
    }
    sigma_ = (float)std::sqrt(sum2 / ((double)n_samples * d_));
}
