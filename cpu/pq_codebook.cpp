#include "cpu/pq_codebook.h"

#include <istream>
#include <ostream>
#include "cpu/erfinv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

inline float l2sqr(const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

} // namespace

PQCodebook::PQCodebook(int d, int M, int B)
    : d_(d), M_(M), Ds_(d / M), K_(1 << B)
{
    if (d <= 0 || M <= 0)  throw std::invalid_argument("PQCodebook: d, M must be positive");
    if (d % M != 0)        throw std::invalid_argument("PQCodebook: d must be divisible by M");
    if (B <= 0 || B > 8)   throw std::invalid_argument("PQCodebook: B must be in 1..8 (one code byte per subspace)");
    cent_.assign((size_t)M_ * K_ * Ds_, 0.f);
}

void PQCodebook::analytical_init(const float* sub, int n, float* out, int seed) const {
    const int dim = Ds_;
    std::vector<float> mean(dim, 0.f), var(dim, 0.f);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < dim; ++j) mean[j] += sub[(size_t)i * dim + j];
    for (int j = 0; j < dim; ++j) mean[j] /= (float)n;

    if (n > 1) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < dim; ++j) {
                float t = sub[(size_t)i * dim + j] - mean[j];
                var[j] += t * t;
            }
        for (int j = 0; j < dim; ++j) var[j] /= (float)(n - 1);
    }
    analytical_init_from_stats(mean.data(), var.data(), out, seed);
}

// The construction past the first two moments touches no data: a trimmed
// variance sets the radius, the radii are Gaussian quantiles, and the
// directions are random vectors orthogonalised against each other. Splitting it
// here lets the device compute the two moments -- which is the only part that
// reads all n*d values -- and hand back M*Ds*2 floats instead of the whole
// rotated training set.
void PQCodebook::analytical_init_from_stats(const float* mean, const float* var,
                                            float* out, int seed) const {
    const int dim = Ds_, k = K_;

    // A trimmed statistic rather than the plain mean: after the JL rotation the
    // per-dimension variances are close but not identical, and a single outlier
    // dimension would otherwise set the radius for every centroid.
    float robust_var;
    {
        std::vector<float> sv(var, var + dim);
        std::sort(sv.begin(), sv.end());
        if (dim <= 8) {
            robust_var = sv[dim / 2];
        } else {
            int lo = dim / 4, hi = 3 * dim / 4;
            float trimmed = 0.f;
            for (int i = lo; i < hi; ++i) trimmed += sv[i];
            trimmed /= (float)(hi - lo);
            robust_var = (dim > 32) ? 0.8f * sv[dim / 2] + 0.2f * trimmed : trimmed;
        }
    }
    const float std_scale = std::sqrt(std::max(robust_var, 1e-12f));

    // Radii: Gaussian quantiles, widened slightly in high dimension where the
    // mass sits further from the mean than the 1-D quantile suggests.
    std::vector<float> radius(k);
    for (int i = 0; i < k; ++i) {
        float q = (float)(i + 1) / (float)(k + 1);
        radius[i] = std::sqrt(2.f) * erfinv_f(2.f * q - 1.f);
    }
    if (dim > 64) {
        float adj = std::min(1.f + 0.03f * std::log((float)dim / 64.f), 1.15f);
        for (int i = 0; i < k; ++i) radius[i] *= adj;
    }

    // Directions: random, then Gram-Schmidt against the ones already placed so
    // the first min(k, dim) centroids spread over independent axes instead of
    // clustering along one.
    std::mt19937 rng((uint32_t)seed);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::normal_distribution<float> noise(0.f, std_scale * 0.01f);
    std::vector<float> dirs((size_t)k * dim);

    for (int i = 0; i < k; ++i) {
        float* di = dirs.data() + (size_t)i * dim;
        for (int j = 0; j < dim; ++j) di[j] = gauss(rng);

        const int basis = std::min(i, dim);
        for (int pass = 0; pass < 2 && basis > 0; ++pass)
            for (int b = 0; b < basis; ++b) {
                const float* db = dirs.data() + (size_t)b * dim;
                float dot = std::inner_product(di, di + dim, db, 0.f);
                for (int j = 0; j < dim; ++j) di[j] -= dot * db[j];
            }

        float nsq = std::inner_product(di, di + dim, di, 0.f);
        if (!std::isfinite(nsq) || nsq < 1e-12f) {
            for (int j = 0; j < dim; ++j) di[j] = gauss(rng);
            nsq = std::inner_product(di, di + dim, di, 0.f);
        }
        const float inv = 1.f / std::sqrt(std::max(nsq, 1e-12f));
        for (int j = 0; j < dim; ++j) di[j] *= inv;
    }

    for (int i = 0; i < k; ++i) {
        const float r = std::fabs(radius[i]) * std_scale;
        const float* di = dirs.data() + (size_t)i * dim;
        float* ci = out + (size_t)i * dim;
        for (int j = 0; j < dim; ++j) {
            float adj = std::sqrt(var[j] / std::max(robust_var, 1e-12f));
            adj = std::clamp(adj, 0.7f, 1.5f);
            ci[j] = mean[j] + r * di[j] * adj + noise(rng);
        }
    }
}

void PQCodebook::train(const float* y, int n, int kmeans_iters, int seed) {
    if (n <= 0) throw std::invalid_argument("PQCodebook::train: n must be positive");

    // The subspaces are independent: each owns its slice of cent_, and
    // analytical_init seeds its own generator from seed + m, so nothing is
    // shared across m but the read-only training set. Training was ~26 s of the
    // 27 s index build against cuVS's 4.5 s, and all of it was this loop on one
    // core. The scratch moves inside so each thread has its own.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M_; ++m) {
        std::vector<float> sub((size_t)n * Ds_);
        std::vector<int>   assign(n);
        std::vector<float> sums((size_t)K_ * Ds_);
        std::vector<int>   counts(K_);

        for (int i = 0; i < n; ++i)
            std::memcpy(sub.data() + (size_t)i * Ds_,
                        y + (size_t)i * d_ + (size_t)m * Ds_,
                        (size_t)Ds_ * sizeof(float));

        float* cent = cent_.data() + (size_t)m * K_ * Ds_;
        analytical_init(sub.data(), n, cent, seed + m);

        for (int it = 0; it < kmeans_iters; ++it) {
            for (int i = 0; i < n; ++i) {
                const float* xi = sub.data() + (size_t)i * Ds_;
                int best = 0; float bd = std::numeric_limits<float>::max();
                for (int c = 0; c < K_; ++c) {
                    float dd = l2sqr(xi, cent + (size_t)c * Ds_, Ds_);
                    if (dd < bd) { bd = dd; best = c; }
                }
                assign[i] = best;
            }

            std::fill(sums.begin(), sums.end(), 0.f);
            std::fill(counts.begin(), counts.end(), 0);
            for (int i = 0; i < n; ++i) {
                float* s = sums.data() + (size_t)assign[i] * Ds_;
                const float* xi = sub.data() + (size_t)i * Ds_;
                for (int j = 0; j < Ds_; ++j) s[j] += xi[j];
                counts[assign[i]]++;
            }
            for (int c = 0; c < K_; ++c) {
                if (counts[c] == 0) continue;   // keep the analytical position
                float* cc = cent + (size_t)c * Ds_;
                const float* s = sums.data() + (size_t)c * Ds_;
                const float inv = 1.f / (float)counts[c];
                for (int j = 0; j < Ds_; ++j) cc[j] = s[j] * inv;
            }
        }
    }
}

int PQCodebook::encode_subspace(int m, const float* ym) const {
    const float* c = centroids(m);
    int best = 0; float bd = std::numeric_limits<float>::max();
    for (int k = 0; k < K_; ++k) {
        float dd = l2sqr(ym, c + (size_t)k * Ds_, Ds_);
        if (dd < bd) { bd = dd; best = k; }
    }
    return best;
}

void PQCodebook::reconstruct(const uint8_t* code, float* out) const {
    for (int m = 0; m < M_; ++m)
        std::memcpy(out + (size_t)m * Ds_,
                    centroids(m) + (size_t)code[m] * Ds_,
                    (size_t)Ds_ * sizeof(float));
}


// ── trained-state round-trip ──────────────────────────────────────────────────
void PQCodebook::write_state(std::ostream& os) const {
    const long long n = (long long)cent_.size();
    os.write(reinterpret_cast<const char*>(&d_),  sizeof d_);
    os.write(reinterpret_cast<const char*>(&M_),  sizeof M_);
    os.write(reinterpret_cast<const char*>(&Ds_), sizeof Ds_);
    os.write(reinterpret_cast<const char*>(&K_),  sizeof K_);
    os.write(reinterpret_cast<const char*>(&n),   sizeof n);
    os.write(reinterpret_cast<const char*>(cent_.data()), (std::streamsize)n * sizeof(float));
}

bool PQCodebook::read_state(std::istream& is) {
    int d = 0, M = 0, Ds = 0, K = 0; long long n = 0;
    is.read(reinterpret_cast<char*>(&d),  sizeof d);
    is.read(reinterpret_cast<char*>(&M),  sizeof M);
    is.read(reinterpret_cast<char*>(&Ds), sizeof Ds);
    is.read(reinterpret_cast<char*>(&K),  sizeof K);
    is.read(reinterpret_cast<char*>(&n),  sizeof n);
    if (!is || d != d_ || M != M_ || Ds != Ds_ || K != K_ ||
        n != (long long)cent_.size()) return false;
    is.read(reinterpret_cast<char*>(cent_.data()), (std::streamsize)n * sizeof(float));
    return (bool)is;
}


void PQCodebook::init_from_stats(const float* mean_all, const float* var_all, int seed) {
    for (int m = 0; m < M_; ++m)
        analytical_init_from_stats(mean_all + (size_t)m * Ds_,
                                   var_all  + (size_t)m * Ds_,
                                   cent_.data() + (size_t)m * K_ * Ds_,
                                   seed + m);
}


void PQCodebook::build_analytical_cartesian(float sigma) {
    int B = 0;
    while ((1 << B) < K_) ++B;              // K = 2^B
    if (!cartesian_admissible(B, Ds_))
        throw std::invalid_argument(
            "build_analytical_cartesian: Ds=" + std::to_string(Ds_) +
            " does not divide B=" + std::to_string(B) + ", so K^(1/Ds) is not "
            "an integer number of levels per dimension");

    // Levels per dimension, and the one scalar codebook they share.
    const int L = 1 << (B / Ds_);           // = K^(1/Ds)
    std::vector<float> level(L);
    for (int i = 1; i <= L; ++i) {
        const float q = ((float)i - 0.5f) / (float)L;
        level[i - 1] = sigma * std::sqrt(2.f) * erfinv_f(2.f * q - 1.f);
    }

    // Codeword k of a subspace is k written in base L across its Ds dimensions.
    for (int m = 0; m < M_; ++m) {
        float* cent = cent_.data() + (size_t)m * K_ * Ds_;
        for (int k = 0; k < K_; ++k) {
            int rem = k;
            for (int j = Ds_ - 1; j >= 0; --j) {
                cent[(size_t)k * Ds_ + j] = level[rem % L];
                rem /= L;
            }
        }
    }
}
